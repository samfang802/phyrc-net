#!/usr/bin/env python3
"""Does a per-pixel uncertainty read off the sampling trajectory say where S is wrong? (No training.)

Idea (docs: per-pixel self-estimated confidence): replace the scalar S confidence (t_s in training, a constant at
sampling) by a map U that the model computes the same way in training and at sampling. First candidate, read off
the ODE trajectory of the step that produced S:
  U_pix(t_ref)  = blur(|x1_hat(t_ref) - S|)                        gray levels, x1_hat = z + (1 - t) v at t_ref
  U_feat(t_ref) = blur(|feat(x1_hat(t_ref)) - feat(S)|)           feat = the rule memory's structure feature
  U+(t_ref)     = blur(relu(S - x1_hat(t_ref)))   S ended brighter than its early estimate: structure possibly LOST
  U-(t_ref)     = blur(relu(x1_hat(t_ref) - S))   S ended darker: structure added late (new or invented)
  (--signed: U+ / U- instead of U_feat. U+ alone tells the next step "S may have lost something here, ask the
  memory" without pointing at the model's own new cracks, which a real-frame target would teach it to delete.)
High U = the pixel was still open at t_ref and only settled late, so S's content there is a draw, not a copy.

S = the model's own h-step rollout from the real frame t-1-h (as memory_degraded_probe.py), memory = the self
memory along that rollout. U is recorded on the last rollout step. Labels against the real frame t-1 (crack maps,
2 px tolerance). Tasks, each an AUROC of the signal (0.5 = no information) inside its domain:
  lost   domain: no S crack within 2 px. positive: a real crack of t-1 (S lost it)
  false  domain: S crack pixels. positive: no real crack within 2 px (S invented it)
  arb    domain: the self memory has structure, S has no crack within 2 px (where S and memory disagree, the
         decision the next step has to make). positive: the real t-1 has a crack within 2 px (S lost it, the
         memory is right); negative: it does not (crack closed or memory wrong). Baseline: memory depth
  fog    domain: rough or fog pixels of S (damage_texture classes, thresholds from the split's first 3 real
         frames per walk). positive: fog. Baseline: texture energy E (fog is defined as low E)
Pre-registered reading (docs): AUROC < 0.6 on lost / arb: drop the trajectory U; >= 0.7: worth training with.
Signed run: U+ should reach >= 0.7 on lost / arb and stay near 0.5 on false.
Each AUROC is also reported texture-stratified (`_strat`: within quintiles of the texture energy E of S, pooled
with weights n_pos * n_neg), since |x1_hat - S| is larger in textured regions anyway.

  python3 scripts/eval/uncertainty_map_probe.py --model rfm_9k=<run>@<ckpt> --out x.json
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
from scipy.stats import rankdata

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

TASKS = ("lost", "false", "arb", "fog")
MAX_PIX = 3000  # per class, per position and task (random subsample)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", action="append", required=True, help="name=result_dir@checkpoint (repeatable)")
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--hops", default="1,4,8", help="rollout lengths h of S")
    p.add_argument("--t_ref", default="0.3,0.5,0.7", help="flow times at which x1_hat is read")
    p.add_argument("--signed", action="store_true", help="U+ / U- (signed gray difference) instead of U_feat")
    p.add_argument("--blur", type=float, default=1.5, help="Gaussian sigma (px) of the U maps")
    p.add_argument("--n", type=int, default=8, help="draws per target frame")
    p.add_argument("--t_every", type=int, default=2)
    p.add_argument("--chunk", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def _rollout_rec(ctx, walk, ps, h, seeds, t_refs):
    """memory_degraded_probe._rollout, recording x1_hat(t_ref) on the last step."""
    cfm, n = ctx.cfm, ctx.n
    starts = [p - 1 - h for p in ps]
    tr = ctx.track([walk[: s + 1] for s in starts])
    state = ctx.R(ctx.X, [walk[s] for s in starts])
    B = state.shape[0]
    trace = {}
    with cfm.refiner_noise_streams([s + 7 for s in seeds], [n] * len(ps), state.device):
        shared = cfm._refiner_start_noise(None, B, state.device) if cfm.refiner_shared_lowfreq_noise else None
        for q in range(1, h + 1):
            i, j = [walk[s + q] for s in starts], [walk[s + q - 1] for s in starts]
            z0 = cfm._refiner_start_noise(shared, B, state.device)
            last = q == h
            cfm._x1_record_t, cfm._x1_trace = (list(t_refs) if last else None), {}
            try:
                state = cfm.sample_with_ode(ctx.R(ctx.C, i), z0=z0, gru_feat=ctx.R(ctx.G, i), damage_mask=ctx.mask(i),
                                            lr_cond=ctx.hint(state, tr, ctx.conf_inf, hops=q - 1),
                                            gru_delta=ctx.dg(i, j))
            finally:
                if last:
                    trace = cfm._x1_trace
                cfm._x1_record_t, cfm._x1_trace = None, {}
            if tr is not None:
                tr.update(state)
    return state, tr, trace


@torch.no_grad()
def _probe(spec, cli, device):
    name, cfm, ds, _dl, _lay, step = _sp._build(spec, cli.split, "mean", device)
    X, C, G, M = _rp._load_split(cfm, ds, device)
    ctx = _rp._Ctx(cfm, X, C, G, M, cli.n)
    if not ctx.has_mem:
        raise ValueError(f"{name}: the model has no memory")
    walks = _rp._walks(ds)
    hops = [int(h) for h in cli.hops.split(",") if h.strip()]
    t_refs = [float(t) for t in cli.t_ref.split(",") if t.strip()]
    n = cli.n
    st = np.ones((5, 5), bool)
    feat = lambda x: _rp.memory_feature(x, ctx.sig, ctx.tau)[:, 0].float().cpu().numpy()
    gray = lambda x: dm.to_uint8_gray(cfm._decode_from_target_space(x)).astype(np.float32)
    blur = lambda a: ndi.gaussian_filter(a, (0, cli.blur, cli.blur))

    # fog thresholds from the first 3 real frames of every walk (low drift: the closest to intact here)
    intact = [int(k) for w in walks for k in w[:3]]
    thr = dt.calibrate(gray(X[torch.as_tensor(intact, device=device)]).astype(np.uint8))
    extra = [f"{u}_{t}" for u in (("Uplus", "Uminus") if cli.signed else ("Ufeat",)) for t in t_refs]
    signals = [f"Upix_{t}" for t in t_refs] + extra + ["memdepth", "texE"]
    # per (h, task): list of [n_pix, len(signals)] arrays for positives and negatives (same pixels for every signal)
    pool = {(h, k): ([], []) for h in hops for k in TASKS}
    rate = {(h, k): [] for h in hops for k in TASKS}
    t_used = {}
    rng = np.random.default_rng(cli.seed)
    print(f"[umap] {name} step {step} {cli.split}: {len(walks)} walks, hops {hops}, t_ref {t_refs}, "
          f"conf {ctx.conf_inf}, ode steps {cfm.cfm_ode_steps}, solver {cfm.cfm_solver}", flush=True)
    t0 = time.time()

    def take(mask_pos, mask_neg, stack, key):
        for m, lst in ((mask_pos, pool[key][0]), (mask_neg, pool[key][1])):
            idx = np.flatnonzero(m.ravel())
            if len(idx) > MAX_PIX:
                idx = rng.choice(idx, MAX_PIX, replace=False)
            lst.append(stack[:, idx].T.astype(np.float32))

    for wi, walk in enumerate(walks):
        pos = list(range(max(hops) + 1, len(walk), cli.t_every))
        for c0 in range(0, len(pos), max(1, cli.chunk)):
            ps = pos[c0:c0 + max(1, cli.chunk)]
            P = len(ps)
            seeds = [cli.seed + 100003 * wi + p for p in ps]
            j = [walk[p - 1] for p in ps]
            Rp_u8 = dm.to_uint8_gray(cfm._decode_from_target_space(X[torch.as_tensor([int(x) for x in j],
                                                                                      device=device)]))
            cr = dt.crack_map(Rp_u8.astype(np.uint8))
            crd = np.stack([ndi.binary_dilation(c, st) for c in cr])
            for h in hops:
                S, tr, trace = _rollout_rec(ctx, walk, ps, h, seeds, t_refs)
                S_g, S_f = gray(S), feat(S)
                cs = dt.crack_map(S_g.astype(np.uint8))
                csd = np.stack([ndi.binary_dilation(c, st) for c in cs])
                memv = tr.value[:, 0].float().cpu().numpy()
                rough, fog, _white, E = dt.pixel_classes(S_g.astype(np.uint8), thr)
                sig = {}
                for t in t_refs:
                    t_used[t] = trace[t][0]
                    x1 = trace[t][1]
                    d = gray(x1) - S_g
                    sig[f"Upix_{t}"] = blur(np.abs(d))
                    if cli.signed:
                        sig[f"Uplus_{t}"] = blur(np.maximum(-d, 0.0))
                        sig[f"Uminus_{t}"] = blur(np.maximum(d, 0.0))
                    else:
                        sig[f"Ufeat_{t}"] = blur(np.abs(feat(x1) - S_f))
                sig["memdepth"] = -memv
                sig["texE"] = -E  # higher = smoother, so a positive AUROC for fog
                for r in range(P * n):
                    b = r // n
                    dom = {
                        "lost": (~csd[r] & cr[b], ~csd[r] & ~crd[b]),
                        "false": (cs[r] & ~crd[b], cs[r] & crd[b]),
                        "arb": ((memv[r] < -0.03) & ~csd[r] & crd[b], (memv[r] < -0.03) & ~csd[r] & ~crd[b]),
                        "fog": (fog[r], rough[r]),
                    }
                    stack = np.stack([sig[s][r].ravel() for s in signals])
                    for k, (mp, mn) in dom.items():
                        tot = int(mp.sum() + mn.sum())
                        rate[(h, k)].append(mp.sum() / tot if tot else np.nan)
                        take(mp, mn, stack, (h, k))
            print(f"[umap] {name} walk {wi} t={ps[0]}..{ps[-1]} ({time.time() - t0:.0f} s)", flush=True)

    res = {}
    for h in hops:
        for k in TASKS:
            row = {"pos_rate": float(np.nanmean(rate[(h, k)]))}
            p_, n_ = pool[(h, k)]
            p_ = np.concatenate(p_) if p_ else np.zeros((0, len(signals)))
            n_ = np.concatenate(n_) if n_ else np.zeros((0, len(signals)))
            row["n"] = [int(len(p_)), int(len(n_))]
            ie = signals.index("texE")
            edges = np.quantile(np.concatenate([p_[:, ie], n_[:, ie]]), [0.2, 0.4, 0.6, 0.8]) if len(p_) and len(n_) \
                else []
            bp, bn = np.digitize(p_[:, ie], edges), np.digitize(n_[:, ie], edges)
            for si, s in enumerate(signals):
                row[s] = _auroc(p_[:, si], n_[:, si])
                num = den = 0.0
                for q in range(5):
                    a, b = p_[bp == q, si], n_[bn == q, si]
                    if len(a) and len(b):
                        w = float(len(a)) * len(b)
                        num, den = num + w * _auroc(a, b), den + w
                row[f"{s}_strat"] = num / den if den else float("nan")
            res[f"h{h}_{k}"] = row
    print(f"\n[umap] {name} ({time.time() - t0:.0f} s), t_ref actually used {t_used}")
    print("[umap]   " + f"{'task':10s} {'pos':>6s} " + " ".join(f"{s:>10s}" for s in signals))
    for key, row in res.items():
        print("[umap]   " + f"{key:10s} {row['pos_rate']:6.3f} " + " ".join(f"{row[s]:10.3f}" for s in signals))
    print("[umap]   texture-stratified (within E quintiles):")
    for key, row in res.items():
        print("[umap]   " + f"{key:10s} {'':6s} " + " ".join(f"{row[s + '_strat']:10.3f}" for s in signals))
    del cfm
    torch.cuda.empty_cache()
    return name, {"model": spec, "step": step, "t_used": {str(k): v for k, v in t_used.items()},
                  "fog_thr": thr, "auroc": res}


def main():
    cli = _parse_args()
    device = torch.device("cuda")
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    out = {}
    for spec in cli.model:
        name, r = _probe(spec, cli, device)
        out[name] = r
        with open(cli.out, "w") as fh:
            json.dump({"split": cli.split, "n": cli.n, "hops": cli.hops, "t_ref": cli.t_ref, "models": out}, fh,
                      indent=2)
    print(f"[umap] wrote {cli.out}")


if __name__ == "__main__":
    main()
