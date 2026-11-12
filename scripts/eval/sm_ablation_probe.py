#!/usr/bin/env python3
"""Where do the refiner's cracks come from: S_{t-1}, the memory M, or neither? (No training.)

For target frames t of the split's real walks, frame t is generated with the full sampler (cfm_ode_steps) from
a controlled input, every condition on the same start noise per position (paired):
  A  S real   M real   S = real frame t-1, M = rule memory of the real frames up to t-1 (the clean input)
  B  S real   M zero   memory dropped (the "nothing written" state the model trained with)
  C  S real   M swap   memory of the other face of the column at the same position (wrong crack positions)
  D  S swap   M real   S = the other face's real frame t-1
  E  S swap   M zero   neither carries this face's cracks: the floor from the conditions alone
  F  S blur   M real   S = real frame t-1, Gaussian-blurred (σ --blur px): is a soft S distrusted?
  G  S real   M real   S confidence (1, 0 hops) = "S is a real photo" instead of the sampling value
A-F use the sampling confidence (refiner_state_conf_value) with a long chain (hops = refiner_window), as late
frames of a walk get it. Scored on the generated frame t against the real frame t (uint8 gray):
  keep / false     real cracks of frame t found (2 px) / cracks away from real ones, per real crack pixel
  keepS            share of S's own cracks (real frame t-1) redrawn in frame t: how much is copied from S
  keepM            share of the memory's structure pixels covered by a crack of frame t
  d_mean           tone offset against the real frame t
Reading: keep(A) - keep(D) = cracks that come from S; keep(A) - keep(B) = cracks that come from M; E = the floor.
F against A: a blurrier S is used less. G against A: the confidence channel gates S.

  python3 scripts/eval/sm_ablation_probe.py --model rfmv2=<run>@<ckpt> --model rfm=<run>@<ckpt> --out x.json
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
from scipy import ndimage as ndi

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_spec = importlib.util.spec_from_file_location(
    "_rp", os.path.join(_ROOT, "scripts", "eval", "rollout_vs_resample_probe.py"))
_rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rp)
_sp = _rp._sp

from src.metrics import damage_texture as dt  # noqa: E402
from src.metrics import dino_mmd as dm  # noqa: E402
from src.model.cfm.refiner_memory import _gauss_1d  # noqa: E402

CONDS = {  # name: (S source, M source, confidence)
    "A_Sreal_Mreal": ("real", "real", "inf"),
    "B_Sreal_Mzero": ("real", "zero", "inf"),
    "C_Sreal_Mswap": ("real", "swap", "inf"),
    "D_Sswap_Mreal": ("swap", "real", "inf"),
    "E_Sswap_Mzero": ("swap", "zero", "inf"),
    "F_Sblur_Mreal": ("blur", "real", "inf"),
    "G_Sreal_Mreal_conf1": ("real", "real", "real"),
}
STATS = ("keep", "false", "keepS", "keepM", "d_mean")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", action="append", required=True, help="name=result_dir@checkpoint (repeatable)")
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--n", type=int, default=8, help="draws per target frame")
    p.add_argument("--t_first", type=int, default=8)
    p.add_argument("--t_every", type=int, default=2)
    p.add_argument("--blur", type=float, default=1.0, help="σ (px) of the blurred S of condition F")
    p.add_argument("--chunk", type=int, default=12, help="target positions per batch (rows = chunk * n)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _blur(x, sigma):
    k, r = _gauss_1d(sigma, x.device, x.dtype)
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def _cover(a, b, st):
    """Share of the pixels of mask a within 2 px of mask b."""
    return float((a & ndi.binary_dilation(b, st)).sum() / max(int(a.sum()), 1))


@torch.no_grad()
def _probe(spec, cli, device):
    name, cfm, ds, _dl, _lay, step = _sp._build(spec, cli.split, "mean", device)
    X, C, G, M = _rp._load_split(cfm, ds, device)
    ctx = _rp._Ctx(cfm, X, C, G, M, cli.n)
    walks = _rp._walks(ds)
    if len(walks) < 2:
        raise ValueError("the swap conditions need at least two walks in the split")
    L = int(getattr(cfm, "refiner_window", 0)) or 8
    n = cli.n
    st = np.ones((5, 5), bool)
    rows = {k: [] for k in CONDS}
    print(f"[sm] {name} step {step} {cli.split}: {len(walks)} walks, memory {ctx.has_mem}, conf at sampling "
          f"{ctx.conf_inf} (hops {L}), ode steps {cfm.cfm_ode_steps}, solver {cfm.cfm_solver}", flush=True)
    t0 = time.time()
    for wi, walk in enumerate(walks):
        other = walks[(wi + 1) % len(walks)]
        pos = list(range(max(cli.t_first, 1), len(walk), cli.t_every))
        for c0 in range(0, len(pos), max(1, cli.chunk)):
            ps = pos[c0:c0 + max(1, cli.chunk)]
            P = len(ps)
            seeds = [cli.seed + 100003 * wi + p for p in ps]
            ow = [min(p, len(other) - 1) for p in ps]  # the other face at the same position
            i, j = [walk[p] for p in ps], [walk[p - 1] for p in ps]
            S_src = {"real": ctx.R(X, j), "swap": ctx.R(X, [other[q - 1] for q in ow])}
            S_src["blur"] = _blur(S_src["real"], cli.blur)
            M_src = {"real": ctx.track([walk[:p] for p in ps]), "zero": None,
                     "swap": ctx.track([other[:q] for q in ow])}
            R_u8 = dm.to_uint8_gray(cfm._decode_from_target_space(X[torch.as_tensor([int(x) for x in i], device=device)]))
            Sr_u8 = dm.to_uint8_gray(cfm._decode_from_target_space(X[torch.as_tensor([int(x) for x in j], device=device)]))
            cr = dt.crack_map(R_u8.astype(np.uint8))
            cs = dt.crack_map(Sr_u8.astype(np.uint8))
            mem_real = M_src["real"].value[::n, 0].float().cpu().numpy() < -0.03 if ctx.has_mem else None
            for k, (s_src, m_src, conf_src) in CONDS.items():
                if m_src != "zero" and not ctx.has_mem:
                    m_src = "zero"
                conf, hops = (ctx.conf_inf, L) if conf_src == "inf" else (1.0, 0)
                with cfm.refiner_noise_streams(seeds, [n] * P, device):
                    shared = cfm._refiner_start_noise(None, P * n, device) if cfm.refiner_shared_lowfreq_noise else None
                    z0 = cfm._refiner_start_noise(shared, P * n, device)
                out = cfm.sample_with_ode(ctx.R(C, i), z0=z0, gru_feat=ctx.R(G, i), damage_mask=ctx.mask(i),
                                          lr_cond=ctx.hint(S_src[s_src], M_src[m_src], conf, hops=hops),
                                          gru_delta=ctx.dg(i, j))
                o_u8 = dm.to_uint8_gray(cfm._decode_from_target_space(out))
                co = dt.crack_map(o_u8.astype(np.uint8))
                for b, p in enumerate(ps):
                    blk = slice(b * n, (b + 1) * n)
                    s = _rp._stats(o_u8[blk], R_u8[b])
                    s["keepS"] = float(np.mean([_cover(cs[b], c, st) for c in co[blk]]))
                    s["keepM"] = (float(np.mean([_cover(mem_real[b], c, st) for c in co[blk]]))
                                  if mem_real is not None else float("nan"))
                    rows[k].append({"walk": wi, "pos": int(p), **{x: s[x] for x in STATS}})
            print(f"[sm] {name} walk {wi} t={ps[0]}..{ps[-1]}: " + " ".join(
                f"{k[:1]} keep {np.mean([r['keep'] for r in rows[k][-P:]]):.2f}" for k in CONDS), flush=True)
    summary = {k: {x: float(np.nanmean([r[x] for r in rows[k]])) for x in STATS} for k in CONDS}
    print(f"\n[sm] {name} ({time.time() - t0:.0f} s)  " + " ".join(f"{x:>8}" for x in STATS))
    for k in CONDS:
        print(f"[sm]   {k:22s} " + " ".join(f"{summary[k][x]:8.3f}" for x in STATS))
    a = summary["A_Sreal_Mreal"]
    print(f"[sm]   cracks from S   keep(A)-keep(D) = {a['keep'] - summary['D_Sswap_Mreal']['keep']:+.3f}")
    print(f"[sm]   cracks from M   keep(A)-keep(B) = {a['keep'] - summary['B_Sreal_Mzero']['keep']:+.3f}")
    print(f"[sm]   floor           keep(E)         = {summary['E_Sswap_Mzero']['keep']:.3f}")
    print(f"[sm]   blur S          keep(F)-keep(A) = {summary['F_Sblur_Mreal']['keep'] - a['keep']:+.3f}")
    print(f"[sm]   conf = real     keep(G)-keep(A) = {summary['G_Sreal_Mreal_conf1']['keep'] - a['keep']:+.3f}",
          flush=True)
    del cfm
    torch.cuda.empty_cache()
    return name, {"model": spec, "step": step, "summary": summary, "rows": rows}


def main():
    cli = _parse_args()
    device = torch.device("cuda")
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    res = {}
    for spec in cli.model:
        name, r = _probe(spec, cli, device)
        res[name] = r
    with open(cli.out, "w") as fh:
        json.dump({"split": cli.split, "n": cli.n, "blur": cli.blur, "conds": CONDS, "models": res}, fh, indent=2)
    print("\n[sm] keep by condition (rows) and model (columns)")
    names = list(res)
    print("[sm]   " + " " * 22 + " ".join(f"{m:>10}" for m in names))
    for k in CONDS:
        print(f"[sm]   {k:22s} " + " ".join(f"{res[m]['summary'][k]['keep']:10.3f}" for m in names))
    print(f"[sm] wrote {cli.out}")


if __name__ == "__main__":
    main()
