#!/usr/bin/env python3
"""Does the memory help once S has degraded? (No training.)

scripts/eval/sm_ablation_probe.py found that zeroing the memory changes nothing when S is the real frame t-1. The
memory is meant for the other case: after a long rollout S has lost cracks, and the memory should bring them back.
Here S_{t-1} is the model's own h-step rollout from the real frame t-1-h (as roll_h of
rollout_vs_resample_probe.py, h = --hops), and frame t is generated from it with the memory
  real   rule memory of the real frames up to t-1 (a clean memory: the upper bound)
  self   the memory as sampling has it: real frames up to t-1-h, then the h generated frames
  zero   memory dropped
  swap   the other face's real memory at the same position (wrong crack positions)
plus S real / M real and S real / M zero as the clean reference. All conditions of a position share the start noise
of frame t (and the rollout noise). Scored on frame t against the real frame t (uint8 gray):
  keep / false   real cracks of frame t found (2 px) / cracks away from real ones, per real crack pixel
  recover        real cracks of frame t that S has lost (none of S's cracks within 2 px): share redrawn in frame t.
                 This is what the memory is for; recover(self) - recover(zero) = what the memory adds at sampling
  keepM          share of the self memory's structure pixels covered by a crack of frame t
  d_mean         tone offset against the real frame t
S statistics (keep / false of S against the real frame t-1) are reported per h, for the degradation level.

  python3 scripts/eval/memory_degraded_probe.py --model rfm=<run>@<ckpt> --model rfmv2=<run>@<ckpt> --out x.json
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

MEMS = ("real", "self", "zero", "swap")
STATS = ("keep", "false", "recover", "keepM", "d_mean")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", action="append", required=True, help="name=result_dir@checkpoint (repeatable)")
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--hops", default="1,4,8", help="rollout lengths h of S")
    p.add_argument("--n", type=int, default=8, help="draws per target frame")
    p.add_argument("--t_every", type=int, default=2)
    p.add_argument("--chunk", type=int, default=12, help="target positions per batch (rows = chunk * n)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conf", type=float, default=None,
                   help="S confidence fed at every step instead of the model's refiner_state_conf_value")
    p.add_argument("--out", required=True)
    return p.parse_args()


@torch.no_grad()
def _rollout(ctx, walk, ps, h, seeds):
    """roll_h of rollout_vs_resample_probe.py, also returning the memory accumulated along the rollout."""
    cfm, n = ctx.cfm, ctx.n
    starts = [p - 1 - h for p in ps]
    tr = ctx.track([walk[: s + 1] for s in starts])
    state = ctx.R(ctx.X, [walk[s] for s in starts])
    B = state.shape[0]
    with cfm.refiner_noise_streams([s + 7 for s in seeds], [n] * len(ps), state.device):
        shared = cfm._refiner_start_noise(None, B, state.device) if cfm.refiner_shared_lowfreq_noise else None
        for q in range(1, h + 1):
            i, j = [walk[s + q] for s in starts], [walk[s + q - 1] for s in starts]
            z0 = cfm._refiner_start_noise(shared, B, state.device)
            state = cfm.sample_with_ode(ctx.R(ctx.C, i), z0=z0, gru_feat=ctx.R(ctx.G, i), damage_mask=ctx.mask(i),
                                        lr_cond=ctx.hint(state, tr, ctx.conf_inf, hops=q - 1),
                                        gru_delta=ctx.dg(i, j))
            if tr is not None:
                tr.update(state)
    return state, tr


def _dil(m, st):
    return ndi.binary_dilation(m, st)


@torch.no_grad()
def _probe(spec, cli, device):
    name, cfm, ds, _dl, _lay, step = _sp._build(spec, cli.split, "mean", device)
    X, C, G, M = _rp._load_split(cfm, ds, device)
    ctx = _rp._Ctx(cfm, X, C, G, M, cli.n)
    if cli.conf is not None:
        ctx.conf_inf = float(cli.conf)
    if not ctx.has_mem:
        raise ValueError(f"{name}: the model has no memory")
    walks = _rp._walks(ds)
    if len(walks) < 2:
        raise ValueError("the swap memory needs at least two walks in the split")
    hops = [int(h) for h in cli.hops.split(",") if h.strip()]
    L = int(getattr(cfm, "refiner_window", 0)) or 8
    n = cli.n
    st = np.ones((5, 5), bool)
    conds = ["Sreal_Mreal", "Sreal_Mzero"] + [f"roll{h}_M{m}" for h in hops for m in MEMS]
    rows = {k: [] for k in conds}
    s_rows = {h: [] for h in hops}
    print(f"[memdeg] {name} step {step} {cli.split}: {len(walks)} walks, hops {hops}, conf at sampling "
          f"{ctx.conf_inf}, ode steps {cfm.cfm_ode_steps}", flush=True)
    t0 = time.time()
    for wi, walk in enumerate(walks):
        other = walks[(wi + 1) % len(walks)]
        pos = list(range(max(hops) + 1, len(walk), cli.t_every))
        for c0 in range(0, len(pos), max(1, cli.chunk)):
            ps = pos[c0:c0 + max(1, cli.chunk)]
            P = len(ps)
            seeds = [cli.seed + 100003 * wi + p for p in ps]
            ow = [min(p, len(other) - 1) for p in ps]
            i, j = [walk[p] for p in ps], [walk[p - 1] for p in ps]
            idx = lambda r: torch.as_tensor([int(x) for x in r], device=device)
            R_u8 = dm.to_uint8_gray(cfm._decode_from_target_space(X[idx(i)]))
            Rp_u8 = dm.to_uint8_gray(cfm._decode_from_target_space(X[idx(j)]))
            cr = dt.crack_map(R_u8.astype(np.uint8))
            mem_real = ctx.track([walk[:p] for p in ps])
            mem_swap = ctx.track([other[:q] for q in ow])

            def frame_t(S, tr, hop):
                with cfm.refiner_noise_streams(seeds, [n] * P, device):
                    shared = cfm._refiner_start_noise(None, P * n, device) if cfm.refiner_shared_lowfreq_noise else None
                    z0 = cfm._refiner_start_noise(shared, P * n, device)
                out = cfm.sample_with_ode(ctx.R(C, i), z0=z0, gru_feat=ctx.R(G, i), damage_mask=ctx.mask(i),
                                          lr_cond=ctx.hint(S, tr, ctx.conf_inf, hops=hop), gru_delta=ctx.dg(i, j))
                return dm.to_uint8_gray(cfm._decode_from_target_space(out))

            def score(k, o_u8, S_u8, self_struct):
                co = dt.crack_map(o_u8.astype(np.uint8))
                cs = dt.crack_map(S_u8.astype(np.uint8))
                for b, p in enumerate(ps):
                    blk = range(b * n, (b + 1) * n)
                    s = _rp._stats(o_u8[b * n:(b + 1) * n], R_u8[b])
                    rec, km = [], []
                    for r in blk:
                        lost = cr[b] & ~_dil(cs[r], st)
                        rec.append((lost & _dil(co[r], st)).sum() / max(int(lost.sum()), 1) if lost.any() else np.nan)
                        sm = self_struct[r]
                        km.append((sm & _dil(co[r], st)).sum() / max(int(sm.sum()), 1) if sm.any() else np.nan)
                    rows[k].append({"walk": wi, "pos": int(p), "keep": s["keep"], "false": s["false"],
                                    "recover": float(np.nanmean(rec)) if np.isfinite(rec).any() else float("nan"),
                                    "keepM": float(np.nanmean(km)) if np.isfinite(km).any() else float("nan"),
                                    "d_mean": s["d_mean"]})

            S_real = ctx.R(X, j)
            Sr_u8 = np.repeat(Rp_u8, n, axis=0)
            real_struct = mem_real.value[:, 0].float().cpu().numpy() < -0.03
            score("Sreal_Mreal", frame_t(S_real, mem_real, L), Sr_u8, real_struct)
            score("Sreal_Mzero", frame_t(S_real, None, L), Sr_u8, real_struct)
            for h in hops:
                S, mem_self = _rollout(ctx, walk, ps, h, seeds)
                S_u8 = dm.to_uint8_gray(cfm._decode_from_target_space(S))
                for b, p in enumerate(ps):
                    s = _rp._stats(S_u8[b * n:(b + 1) * n], Rp_u8[b])
                    s_rows[h].append({"keep": s["keep"], "false": s["false"], "d_mean": s["d_mean"]})
                self_struct = mem_self.value[:, 0].float().cpu().numpy() < -0.03
                for m, tr in (("real", mem_real), ("self", mem_self), ("zero", None), ("swap", mem_swap)):
                    score(f"roll{h}_M{m}", frame_t(S, tr, min(h, L)), S_u8, self_struct)
            print(f"[memdeg] {name} walk {wi} t={ps[0]}..{ps[-1]} ({time.time() - t0:.0f} s): " + " ".join(
                f"roll{h} rec self {np.nanmean([r['recover'] for r in rows[f'roll{h}_Mself'][-P:]]):.2f} "
                f"zero {np.nanmean([r['recover'] for r in rows[f'roll{h}_Mzero'][-P:]]):.2f}" for h in hops), flush=True)
    summary = {k: {x: float(np.nanmean([r[x] for r in rows[k]])) for x in STATS} for k in conds}
    s_summary = {str(h): {x: float(np.mean([r[x] for r in s_rows[h]])) for x in ("keep", "false", "d_mean")}
                 for h in hops}
    print(f"\n[memdeg] {name} ({time.time() - t0:.0f} s)")
    for h in hops:
        s = s_summary[str(h)]
        print(f"[memdeg]   S after {h} generated frames vs the real t-1: keep {s['keep']:.3f} false {s['false']:.3f} "
              f"d_mean {s['d_mean']:+.2f}")
    print(f"[memdeg]   {'condition':16s} " + " ".join(f"{x:>8}" for x in STATS))
    for k in conds:
        print(f"[memdeg]   {k:16s} " + " ".join(f"{summary[k][x]:8.3f}" for x in STATS))
    for h in hops:
        g = lambda m, x: summary[f"roll{h}_M{m}"][x]
        print(f"[memdeg]   h={h}: memory adds at sampling  recover {g('self', 'recover') - g('zero', 'recover'):+.3f}"
              f"  keep {g('self', 'keep') - g('zero', 'keep'):+.3f} | a clean memory would add  recover "
              f"{g('real', 'recover') - g('zero', 'recover'):+.3f}  keep {g('real', 'keep') - g('zero', 'keep'):+.3f}"
              f" | wrong memory  keep {g('swap', 'keep') - g('zero', 'keep'):+.3f}  false "
              f"{g('swap', 'false') - g('zero', 'false'):+.3f}", flush=True)
    del cfm
    torch.cuda.empty_cache()
    return name, {"model": spec, "step": step, "summary": summary, "S": s_summary, "rows": rows}


def main():
    cli = _parse_args()
    device = torch.device("cuda")
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    res = {}
    for spec in cli.model:
        name, r = _probe(spec, cli, device)
        res[name] = r
        with open(cli.out, "w") as fh:  # after every model, so a late failure keeps the earlier ones
            json.dump({"split": cli.split, "n": cli.n, "hops": cli.hops, "conf": cli.conf, "models": res}, fh,
                      indent=2)
    names = list(res)
    print("\n[memdeg] recover (memory: self - zero | real - zero) by model")
    for h in [int(x) for x in cli.hops.split(",") if x.strip()]:
        print(f"[memdeg]   h={h}: " + "   ".join(
            f"{m} {res[m]['summary'][f'roll{h}_Mself']['recover'] - res[m]['summary'][f'roll{h}_Mzero']['recover']:+.3f}"
            f" | {res[m]['summary'][f'roll{h}_Mreal']['recover'] - res[m]['summary'][f'roll{h}_Mzero']['recover']:+.3f}"
            for m in names))
    print(f"[memdeg] wrote {cli.out}")


if __name__ == "__main__":
    main()
