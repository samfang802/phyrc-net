#!/usr/bin/env python3
"""Do the late ODE steps copy a washed-out S over cracks the early steps had drawn? (No training.)

Hypothesis: after h generated frames S is faded / gray. Early in the ODE (t small) the model, from the memory and
the conditions, predicts a dark crack (x1_hat dark); in the late detail steps it copies S's local look, and the
velocity field pulls the crack back to S's gray.

Test 1, intervention (refiner_target_x1 models: S is only a condition, so it can be swapped mid-trajectory).
Frame t is generated from the same start noise and the self memory with the hint's S
  SS          S_self all the way (sampling as it is)
  RR          the real frame t-1 all the way (upper bound)
  SR@t_sw     S_self for t < t_sw, the real t-1 after       -> what the late phase takes from S
  RS@t_sw     the real t-1 for t < t_sw, S_self after       -> what the early phase takes from S
Prediction of the hypothesis: SR@0.6 close to RR and RS@0.6 close to SS (the late S decides the dark cracks).
If instead SR@0.4 stays close to SS, S acts through the early layout and the late copy is not the mechanism.

Test 2, pull towards S, free of the selection bias of grouping pixels by the final frame: pixels are chosen from
quantities known at t_a (x1_hat(t_a) and S), so under a self-consistent flow E[final - x1_hat(t_a)] = 0 there.
  D_gray   x1_hat(t_a) has a local dark structure (>= --dark gray levels below its 7 px blur), S has none there
  D_dark   x1_hat(t_a) dark and S dark too (control)
  pull     mean(final - x1_hat(t_a)) in gray levels (> 0: brightened in the late steps)
  close    mean(final - x1_hat(t_a)) / mean(S - x1_hat(t_a)) on D_gray: share of the gap to S closed

Test 3, which S does the trajectory need (--conds "E" or "E>L@t_sw": S variant E for t < t_sw, L after):
  S / R       self rollout S / real frame t-1
  Am / Ar     S with the memory's missing structure drawn in: S + min(M - feat(S), 0) on every channel, where
              feat is the memory feature (refiner_memory.memory_feature); Am uses the self-written memory (the
              one the hint carries), Ar the memory written from the real frames up to t-1
  LrHs / LsHr low band (Gaussian --lp_sigma px) of the real t-1 + high band of S, and the reverse
The hint's memory channel is the self memory in every condition.

Scores of frame t against the real frame t: keep / false / recover (as memory_degraded_probe.py), fog share
(fog / (fog + rough), damage_texture classes, thresholds from the split's first 3 real frames per walk), d_mean.

  python3 scripts/eval/late_copy_probe.py --model rfm_9k=<run>@<ckpt> --out x.json
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
    "_umap", os.path.join(_ROOT, "scripts", "eval", "uncertainty_map_probe.py"))
_um = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_um)
_rp, _sp = _um._rp, _um._sp

from src.metrics import damage_texture as dt  # noqa: E402
from src.metrics import dino_mmd as dm  # noqa: E402
from src.model.cfm.refiner_memory import _gauss_1d, memory_feature  # noqa: E402

VARIANTS = ("S", "R", "Am", "Ar", "LrHs", "LsHr")
STATS = ("keep", "false", "recover", "fog_share", "d_mean")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", action="append", required=True, help="name=result_dir@checkpoint (repeatable)")
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--hops", default="1,4,8")
    p.add_argument("--conds", default="S,R,S>R@0.4,R>S@0.4,S>R@0.6,R>S@0.6",
                   help="S schedules: E (all the way) or E>L@t_sw; variants S R Am Ar LrHs LsHr")
    p.add_argument("--lp_sigma", type=float, default=4.0, help="Gaussian sigma (px) of the LrHs / LsHr split")
    p.add_argument("--t_a", default="0.4", help="flow times whose x1_hat anchors the pull measure")
    p.add_argument("--dark", type=float, default=10.0, help="gray levels below the local 7 px blur = dark")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--t_every", type=int, default=2)
    p.add_argument("--chunk", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


@torch.no_grad()
def _probe(spec, cli, device):
    name, cfm, ds, _dl, _lay, step = _sp._build(spec, cli.split, "mean", device)
    if not getattr(cfm, "refiner_target_x1", False):
        raise ValueError(f"{name}: S can only be swapped mid-trajectory in a refiner_target_x1 model")
    X, C, G, M = _rp._load_split(cfm, ds, device)
    ctx = _rp._Ctx(cfm, X, C, G, M, cli.n)
    walks = _rp._walks(ds)
    hops = [int(h) for h in cli.hops.split(",") if h.strip()]
    t_as = [float(t) for t in cli.t_a.split(",") if t.strip()]
    n = cli.n
    L = int(getattr(cfm, "refiner_window", 0)) or 8
    st = np.ones((5, 5), bool)
    gray = lambda x: dm.to_uint8_gray(cfm._decode_from_target_space(x)).astype(np.float32)
    local_dark = lambda g: (ndi.uniform_filter(g, (1, 7, 7)) - g) >= cli.dark
    intact = [int(k) for w in walks for k in w[:3]]
    thr = dt.calibrate(gray(X[torch.as_tensor(intact, device=device)]).astype(np.uint8))
    conds = [k.strip() for k in cli.conds.split(",") if k.strip()]
    sched = {}
    for k in conds:
        e, _, rest = k.partition(">")
        late, _, t_sw = rest.partition("@")
        sched[k] = (e, late or None, float(t_sw) if t_sw else None)
        if (late and not t_sw) or any(v not in VARIANTS for v in (e, late) if v):
            raise ValueError(f"bad condition {k!r}")
    used = {v for e, l, _t in sched.values() for v in (e, l) if v}
    sig, tau = float(cfm.refiner_memory_sigma), float(cfm.refiner_memory_tau)
    k1, r1 = _gauss_1d(cli.lp_sigma, device, torch.float32)
    C_ = X.shape[1]

    def low(x):
        x = x.float()
        x = F.conv2d(F.pad(x, (r1, r1, 0, 0), mode="reflect"), k1.view(1, 1, 1, -1).expand(C_, 1, 1, -1), groups=C_)
        return F.conv2d(F.pad(x, (0, 0, r1, r1), mode="reflect"), k1.view(1, 1, -1, 1).expand(C_, 1, -1, 1),
                        groups=C_)

    def aug(S, M):
        return (S.float() + (M.float() - memory_feature(S, sig, tau).float()).clamp(max=0.0)).to(S.dtype)
    rows = {(h, k): [] for h in hops for k in conds}
    pull = {(h, k, t): {"gray": [], "dark": [], "close_num": [], "close_den": []}
            for h in hops for k in conds for t in t_as}
    print(f"[late] {name} step {step} {cli.split}: {len(walks)} walks, hops {hops}, conds {conds}, "
          f"lp sigma {cli.lp_sigma}, anchor t {t_as}, conf {ctx.conf_inf}, ode steps {cfm.cfm_ode_steps}, solver {cfm.cfm_solver}", flush=True)
    t0 = time.time()
    for wi, walk in enumerate(walks):
        pos = list(range(max(hops) + 1, len(walk), cli.t_every))
        for c0 in range(0, len(pos), max(1, cli.chunk)):
            ps = pos[c0:c0 + max(1, cli.chunk)]
            P = len(ps)
            seeds = [cli.seed + 100003 * wi + p for p in ps]
            i, j = [walk[p] for p in ps], [walk[p - 1] for p in ps]
            idx = lambda r: torch.as_tensor([int(x) for x in r], device=device)
            R_u8 = dm.to_uint8_gray(cfm._decode_from_target_space(X[idx(i)]))
            cr = dt.crack_map(R_u8.astype(np.uint8))
            S_real = ctx.R(X, j)
            M_real = ctx.track([walk[:p] for p in ps]).value if "Ar" in used else None
            for h in hops:
                S, tr, _ = _um._rollout_rec(ctx, walk, ps, h, seeds, [])
                hop = min(h, L)
                var = {"S": S, "R": S_real}
                if "Am" in used:
                    var["Am"] = aug(S, tr.value)
                if "Ar" in used:
                    var["Ar"] = aug(S, M_real)
                if used & {"LrHs", "LsHr"}:
                    lr_, ls_ = low(S_real), low(S)
                    var["LrHs"] = (lr_ + S.float() - ls_).to(S.dtype)
                    var["LsHr"] = (ls_ + S_real.float() - lr_).to(S.dtype)
                hint = {v: ctx.hint(var[v], tr, ctx.conf_inf, hops=hop) for v in used}
                S_g = gray(S)
                cs = dt.crack_map(S_g.astype(np.uint8))
                S_dark = local_dark(S_g)
                for k in conds:
                    first, late, t_sw = sched[k]
                    with cfm.refiner_noise_streams(seeds, [n] * P, device):
                        shared = cfm._refiner_start_noise(None, P * n, device) if cfm.refiner_shared_lowfreq_noise \
                            else None
                        z0 = cfm._refiner_start_noise(shared, P * n, device)
                    cfm._x1_record_t, cfm._x1_trace = list(t_as), {}
                    cfm._lr_cond_switch = (t_sw, hint[late]) if t_sw is not None else None
                    try:
                        out = cfm.sample_with_ode(ctx.R(C, i), z0=z0, gru_feat=ctx.R(G, i), damage_mask=ctx.mask(i),
                                                  lr_cond=hint[first], gru_delta=ctx.dg(i, j))
                        trace = cfm._x1_trace
                    finally:
                        cfm._x1_record_t, cfm._x1_trace, cfm._lr_cond_switch = None, {}, None
                    o_g = gray(out)
                    o_u8 = o_g.astype(np.uint8)
                    co = dt.crack_map(o_u8)
                    rough, fog, _w, _E = dt.pixel_classes(o_u8, thr)
                    for b, p in enumerate(ps):
                        blk = slice(b * n, (b + 1) * n)
                        s = _rp._stats(o_u8[blk], R_u8[b])
                        rec = []
                        for r in range(b * n, (b + 1) * n):
                            lost = cr[b] & ~ndi.binary_dilation(cs[r], st)
                            rec.append((lost & ndi.binary_dilation(co[r], st)).sum() / max(int(lost.sum()), 1)
                                       if lost.any() else np.nan)
                        nf, nr = fog[blk].sum(), rough[blk].sum()
                        rows[(h, k)].append({"walk": wi, "pos": int(p), "keep": s["keep"], "false": s["false"],
                                             "recover": float(np.nanmean(rec)) if np.isfinite(rec).any() else np.nan,
                                             "fog_share": float(nf / max(nf + nr, 1)), "d_mean": s["d_mean"]})
                    for t in t_as:
                        a_g = gray(trace[t][1])
                        a_dark = local_dark(a_g)
                        dg, dd = a_dark & ~S_dark, a_dark & S_dark
                        diff = o_g - a_g
                        q = pull[(h, k, t)]
                        q["gray"].append(float(diff[dg].mean()) if dg.any() else np.nan)
                        q["dark"].append(float(diff[dd].mean()) if dd.any() else np.nan)
                        q["close_num"].append(float(diff[dg].sum()))
                        q["close_den"].append(float((S_g - a_g)[dg].sum()))
            print(f"[late] {name} walk {wi} t={ps[0]}..{ps[-1]} ({time.time() - t0:.0f} s)", flush=True)

    summary = {}
    for h in hops:
        for k in conds:
            rr = rows[(h, k)]
            e = {x: float(np.nanmean([r[x] for r in rr])) for x in STATS}
            for t in t_as:
                q = pull[(h, k, t)]
                e[f"pull_gray@{t}"] = float(np.nanmean(q["gray"]))
                e[f"pull_dark@{t}"] = float(np.nanmean(q["dark"]))
                e[f"close@{t}"] = float(np.sum(q["close_num"]) / np.sum(q["close_den"])) if np.sum(q["close_den"]) \
                    else float("nan")
            summary[f"h{h}_{k}"] = e
    cols = list(STATS) + [f"{m}@{t}" for t in t_as for m in ("pull_gray", "pull_dark", "close")]
    print(f"\n[late] {name} ({time.time() - t0:.0f} s)")
    print("[late]   " + f"{'cond':16s} " + " ".join(f"{c:>14s}" for c in cols))
    for key, e in summary.items():
        print("[late]   " + f"{key:16s} " + " ".join(f"{e[c]:14.3f}" for c in cols))
    del cfm
    torch.cuda.empty_cache()
    return name, {"model": spec, "step": step, "fog_thr": thr, "summary": summary,
                  "rows": {f"h{h}_{k}": v for (h, k), v in rows.items()}}


def main():
    cli = _parse_args()
    device = torch.device("cuda")
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    out = {}
    for spec in cli.model:
        name, r = _probe(spec, cli, device)
        out[name] = r
        with open(cli.out, "w") as fh:
            json.dump({"split": cli.split, "n": cli.n, "hops": cli.hops, "conds": cli.conds, "lp_sigma": cli.lp_sigma,
                   "t_a": cli.t_a,
                       "models": out}, fh, indent=2)
    print(f"[late] wrote {cli.out}")


if __name__ == "__main__":
    main()
