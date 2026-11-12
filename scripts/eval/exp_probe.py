#!/usr/bin/env python3
"""Sampler-variant probe of the rfmv2 refiners (no training): S guidance, confidence guidance, memory guidance and a
low-band anchor of S, each on the same positions and start noise as the baseline.

Per model and variant (``--variant NAME:attr=value,...``; ``probe_lp=σ`` is the low-band anchor below), on the
split's real walks, target positions t (every --t_every from --t_first), --n draws each:
  A     S = real frame t-1, M = rule memory of the real frames, sampling confidence, hops = window
  D     S = the other face's real frame t-1 (only with --full): keep(A) - keep(D) = cracks taken from S
  G     A with the confidence of a real photo (conf 1, hops 0) (only with --full): G - A = what the channel gates
  roll  S = the model's own --hops-step rollout from the real frame t-1-h, M = the memory along that rollout
        (sampling); frame t scored against the real frame t. S_* = the rollout's last frame against the real t-1.
Scores (uint8 gray): keep / false as rollout_vs_resample_probe.py, d_mean tone offset, sat = share of pixels >= 250.
probe_lp=σ: every generated S (rollout steps and the S of frame t) gets S - G_σ(S) + G_σ(anchor), anchor = the real
frame the rollout started from (the last real frame the sampler would have): S's low band cannot drift.

  python3 scripts/eval/exp_probe.py --model e=<run>@<ckpt> --variant base: --variant sg1:refiner_state_guidance=1 \
      --out x.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_spec = importlib.util.spec_from_file_location(
    "_rp", os.path.join(_ROOT, "scripts", "eval", "rollout_vs_resample_probe.py"))
_rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rp)
_sp = _rp._sp

from src.metrics import dino_mmd as dm  # noqa: E402
from src.model.cfm.refiner_memory import _gauss_1d  # noqa: E402

STATS = ("keep", "false", "d_mean", "sat", "hp_ratio")
_MISSING = object()


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", action="append", required=True, help="name=result_dir@checkpoint (repeatable)")
    p.add_argument("--variant", action="append", default=None, help="NAME:attr=value,... (repeatable; base: = none)")
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--hops", type=int, default=8)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--t_first", type=int, default=9)
    p.add_argument("--t_every", type=int, default=4)
    p.add_argument("--chunk", type=int, default=12)
    p.add_argument("--full", action="store_true", help="also conditions D and G")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _blur(x, sigma):
    k, r = _gauss_1d(sigma, x.device, torch.float32)
    c = x.shape[1]
    y = x.float().reshape(-1, 1, *x.shape[-2:])
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return y.reshape(x.shape[0], c, *x.shape[-2:]).to(x.dtype)


def _variants(specs):
    out = []
    for v in specs or ["base:"]:
        name, _, kv = v.partition(":")
        pairs = [x for x in kv.split(",") if x.strip()]
        out.append((name, pairs))
    return out


def _cast(val):
    for cast in (int, float):
        try:
            return cast(val)
        except ValueError:
            continue
    return val


@torch.no_grad()
def _probe(spec, variants, cli, device):
    name, cfm, ds, _dl, _lay, step = _sp._build(spec, cli.split, "mean", device)
    X, C, G, M = _rp._load_split(cfm, ds, device)
    ctx = _rp._Ctx(cfm, X, C, G, M, cli.n)
    walks = _rp._walks(ds)
    L = int(getattr(cfm, "refiner_window", 0)) or 8
    n, h = cli.n, cli.hops
    res = {}
    for vname, pairs in variants:
        saved = {}
        lp, dS, noroll = 0.0, 0.0, False
        for kv in pairs:
            k, v = kv.split("=", 1)
            if k == "probe_lp":
                lp = float(v)
                continue
            if k == "probe_dS":  # tone offset (model units) added to S of condition A: gain = d_out / d_S
                dS = float(v)
                continue
            if k == "probe_noroll":
                noroll = bool(int(v))
                continue
            saved[k] = getattr(cfm, k, _MISSING)
            setattr(cfm, k, _cast(v))
        conds = ["A"] + (["D", "G"] if cli.full else []) + ([] if noroll else ["roll"])
        rows = {k: [] for k in conds + ([] if noroll else ["S"])}
        t0 = time.time()
        for wi, walk in enumerate(walks):
            other = walks[(wi + 1) % len(walks)]
            pos = list(range(max(cli.t_first, h + 1), len(walk), cli.t_every))
            for c0 in range(0, len(pos), max(1, cli.chunk)):
                ps = pos[c0:c0 + max(1, cli.chunk)]
                P = len(ps)
                seeds = [cli.seed + 100003 * wi + p for p in ps]
                i, j = [walk[p] for p in ps], [walk[p - 1] for p in ps]
                idx = lambda r: torch.as_tensor([int(x) for x in r], device=device)
                R_u8 = dm.to_uint8_gray(cfm._decode_from_target_space(X[idx(i)]))
                Rp_u8 = dm.to_uint8_gray(cfm._decode_from_target_space(X[idx(j)]))

                def anchor(S, src_rows):
                    if lp <= 0:
                        return S
                    a = ctx.R(X, src_rows)
                    return S - _blur(S, lp) + _blur(a, lp)

                def frame_t(S, tr, conf, hops):
                    with cfm.refiner_noise_streams(seeds, [n] * P, device):
                        shared = cfm._refiner_start_noise(None, P * n, device) if cfm.refiner_shared_lowfreq_noise else None
                        z0 = cfm._refiner_start_noise(shared, P * n, device)
                    out = cfm.sample_with_ode(ctx.R(C, i), z0=z0, gru_feat=ctx.R(G, i), damage_mask=ctx.mask(i),
                                              lr_cond=ctx.hint(S, tr, conf, hops=hops), gru_delta=ctx.dg(i, j))
                    return dm.to_uint8_gray(cfm._decode_from_target_space(out))

                def score(k, o_u8, ref):
                    for b in range(P):
                        blk = o_u8[b * n:(b + 1) * n]
                        s = _rp._stats(blk, ref[b])
                        rows[k].append({"walk": wi, "pos": int(ps[b]), "keep": s["keep"], "false": s["false"],
                                        "d_mean": s["d_mean"], "sat": float((blk >= 250).mean()),
                                        "hp_ratio": s["hp_ratio"]})

                mem_real = ctx.track([walk[:p] for p in ps])
                cfm._tone_ref = ctx.R(X, j)
                score("A", frame_t(ctx.R(X, j) + dS, mem_real, ctx.conf_inf, L), R_u8)
                if cli.full:
                    ow = [min(p, len(other) - 1) for p in ps]
                    score("D", frame_t(ctx.R(X, [other[q - 1] for q in ow]), mem_real, ctx.conf_inf, L), R_u8)
                    score("G", frame_t(ctx.R(X, j), mem_real, 1.0, 0), R_u8)
                if noroll:
                    continue
                # own h-step rollout from the real frame t-1-h, memory written along it
                starts = [p - 1 - h for p in ps]
                start_rows = [walk[s] for s in starts]
                cfm._tone_ref = ctx.R(X, start_rows)
                tr = ctx.track([walk[: s + 1] for s in starts])
                state = ctx.R(X, start_rows)
                B = state.shape[0]
                with cfm.refiner_noise_streams([s + 7 for s in seeds], [n] * P, device):
                    shared = cfm._refiner_start_noise(None, B, device) if cfm.refiner_shared_lowfreq_noise else None
                    for q in range(1, h + 1):
                        ii, jj = [walk[s + q] for s in starts], [walk[s + q - 1] for s in starts]
                        z0 = cfm._refiner_start_noise(shared, B, device)
                        state = cfm.sample_with_ode(ctx.R(C, ii), z0=z0, gru_feat=ctx.R(G, ii),
                                                    damage_mask=ctx.mask(ii),
                                                    lr_cond=ctx.hint(anchor(state, start_rows) if q > 1 else state,
                                                                     tr, ctx.conf_inf, hops=q - 1),
                                                    gru_delta=ctx.dg(ii, jj))
                        if tr is not None:
                            tr.update(state)
                S = anchor(state, start_rows)
                score("S", dm.to_uint8_gray(cfm._decode_from_target_space(S)), Rp_u8)
                score("roll", frame_t(S, tr, ctx.conf_inf, min(h, L)), R_u8)
            print(f"[exp] {name} {vname} walk {wi} ({time.time() - t0:.0f} s): " + " ".join(
                f"{k} {np.mean([r['keep'] for r in rows[k]]):.3f}" for k in conds), flush=True)
        summ = {k: {x: float(np.nanmean([r[x] for r in rows[k]])) for x in STATS} for k in rows}
        res[vname] = {"set": pairs, "summary": summ, "seconds": round(time.time() - t0, 1)}
        print(f"[exp] {name} {vname}: " + " | ".join(
            f"{k} keep {summ[k]['keep']:.3f} false {summ[k]['false']:.3f} d {summ[k]['d_mean']:+.1f} "
            f"sat {summ[k]['sat']:.3f}" for k in rows), flush=True)
        for k, v in saved.items():
            if v is _MISSING:
                delattr(cfm, k)
            else:
                setattr(cfm, k, v)
    del cfm
    torch.cuda.empty_cache()
    return name, {"model": spec, "step": step, "variants": res}


def main():
    cli = _parse_args()
    device = torch.device("cuda")
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    variants = _variants(cli.variant)
    res = {}
    for spec in cli.model:
        name, r = _probe(spec, variants, cli, device)
        res[name] = r
        with open(cli.out, "w") as fh:
            json.dump({"split": cli.split, "n": cli.n, "hops": cli.hops, "full": cli.full, "models": res}, fh,
                      indent=2)
    print(f"[exp] wrote {cli.out}")


if __name__ == "__main__":
    main()
