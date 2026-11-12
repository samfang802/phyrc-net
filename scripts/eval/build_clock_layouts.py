#!/usr/bin/env python3
"""Spall layouts whose area comes from a damage clock and whose shape comes from an OFP rollout or the gravity template.

phase1_c2.md / phase1_ofp.md: the layout generators get the shape roughly right (OFP: bottom first) but
cannot set the amount per column. Here the amount is a monotone regression R_spall = f(clock), fitted on the
4 training columns exactly as in phase0_clock_check.run_clock (LOCO-best model, refit on all 4), and the
shape is the OFP rollout's spalling order:

  per (column face sequence, k), cell priority = index of the first frame (time order) in which the OFP
  rollout has the cell spalled; cells it never spalls come after, ordered by distance to the last frame's
  spalled set. Ties: bottom rows first, then a fixed random order (seed + k).
  layout(frame, k) = the round(A * S^2) cells of lowest priority, A = f(clock at that frame).

The layout is monotone in time whenever A is (all clocks here are non-decreasing). The same OFP rollout
and tie order are used for every clock, so two clocks differ only in the area they ask for.

--shape template ("gravity base", docs/redesign/phase1_fiber_clock.md §7): one static priority for every
column and face instead of an OFP rollout. Template T = per-cell spalled frequency (iso32 > 0.5) over the
training-column frames with > 3 % spall, both faces, made left-right symmetric (hflip is a training
augmentation). Priority = higher T first; ties (cells T never reaches, at the top) bottom rows first, then a
fixed random order. Only the 4 training columns enter T. Deterministic: K = 1.

  python scripts/eval/build_clock_layouts.py --frames results/phase1_fiber_clock/frames_with_clocks.csv \
      --clock hsp_P --ofp results/phase1_ofp/layouts_test_c_iou.npz --out results/phase1_fiber_layout/layouts/hsp_P_test.npz
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.data import layout_io  # noqa: E402
from src.model import ofp as O  # noqa: E402

_spec = importlib.util.spec_from_file_location("p0clock", os.path.join(ROOT, "scripts", "eval", "phase0_clock_check.py"))
P0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P0)


def fit_clock(frames, clock):
    r = P0.run_clock(frames, [clock])
    return r["fit_all"], r["best_model"]


def priority_maps(lay, seed):
    """lay: (T, K, S, S) binary rollout of one sequence in time order. Returns (K, S, S) priority."""
    T, K, S, _ = lay.shape
    on = lay > 0.5
    first = np.where(on.any(0), on.argmax(0), T).astype(float)       # (K, S, S)
    rows = np.arange(S, dtype=float)[:, None] / (S - 1)              # 0 top .. 1 bottom
    pr = np.empty((K, S, S))
    for k in range(K):
        u = np.random.default_rng(seed + k).random((S, S))
        f = first[k].copy()
        never = f >= T
        if never.any():
            final = on[-1, k]
            dist = distance_transform_edt(~final) if final.any() else (1 - rows) * S * np.ones((1, S))
            f[never] = T + dist[never]
        pr[k] = f + 0.5 * (1 - rows) + 0.01 * u
    return pr


def gravity_template(min_area=0.03):
    """(S, S) spalled frequency over the training-column frames with > min_area spall, left-right symmetric."""
    seqs = O.sequences_of(O.load_sequences(), list(P0.TRAIN))
    g = [np.asarray(m, bool) for s in seqs.values() for m in s["frames"]["gt"]]
    T = np.mean([m for m in g if m.mean() > min_area], axis=0)
    return (T + T[:, ::-1]) / 2


def template_priority(T, K=1, seed=0):
    """(K, S, S) priority (lower = spalls first) from template T: higher T first; ties bottom first, then random."""
    S = T.shape[0]
    rows = np.arange(S, dtype=float)[:, None] / (S - 1)               # 0 top .. 1 bottom
    return np.stack([-T + 1e-4 * (1 - rows) + 1e-6 * np.random.default_rng(seed + k).random((S, S))
                     for k in range(K)])


def noisy_priority(T, K, seed, sigma, ell, key):
    """(K, S, S) priority for one sequence key: -(T + sigma * smooth unit-variance gaussian field), one field per
    (key, k). The field is fixed per sequence, so layouts nest (monotone) as the area grows. Calibrated on the
    training columns' real layouts (scripts in the session: sigma 0.05-0.07, ell >= 3 match their component count and
    top-edge roughness; sigma > 0.1 fragments like OFP)."""
    from scipy.ndimage import gaussian_filter
    S = T.shape[0]
    rows = np.arange(S, dtype=float)[:, None] / (S - 1)
    out = []
    for k in range(K):
        g = gaussian_filter(np.random.default_rng(seed + k + sum(map(ord, key))).standard_normal((S, S)), ell,
                            mode="reflect")
        out.append(-(T + sigma * g / g.std()) + 1e-4 * (1 - rows))
    return np.stack(out)


def apply_area(lay, fn, A, seq, ts, seed, prio=None):
    """lay (N, K, S, S) shape source, A (N,) target area, seq / ts (N,) sequence key and timestep.
    prio (K, S, S): one fixed priority for every sequence (template shape); lay then only gives the shape."""
    N, K, S, _ = lay.shape
    out = np.zeros_like(lay)
    fn = np.asarray(fn)
    for s in np.unique(seq):
        idx = np.where(seq == s)[0]
        idx = idx[np.lexsort((fn[idx], ts[idx]))]
        pr = (priority_maps(lay[idx], seed) if prio is None else prio(s) if callable(prio) else prio).reshape(K, -1)
        rank = np.argsort(np.argsort(pr, axis=1), axis=1)            # 0 = first to spall
        for i in idx:
            n_on = int(round(A[i] * S * S))
            out[i] = (rank < n_on).reshape(K, S, S).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="results/phase1_fiber_clock/frames_with_clocks.csv")
    ap.add_argument("--clock", required=True)
    ap.add_argument("--shape", choices=("ofp", "template"), default="ofp")
    ap.add_argument("--ofp", default="", help="OFP layout npz of one split (shape source; --shape ofp)")
    ap.add_argument("--split", default="", help="val / test (--shape template): frames of "
                                                 "Data/DamagePattern/id_list_val_R307.json[split]")
    ap.add_argument("--sigma", type=float, default=0.0, help="--shape template: >0 adds correlated noise (noisy gravity base)")
    ap.add_argument("--ell", type=float, default=3.0, help="noise correlation length (cells)")
    ap.add_argument("--k", type=int, default=1, help="--shape template with --sigma: noise draws per frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    frames = pd.read_csv(a.frames)
    model, mname = fit_clock(frames, a.clock)
    prio = None
    if a.shape == "ofp":
        ofp = layout_io.load_layout_npz(a.ofp)
        split, fn, lay = ofp["split"], ofp["filename"], ofp["layout"]
    else:
        split = a.split
        fn = list(json.load(open(os.path.join(ROOT, "Data/DamagePattern/id_list_val_R307.json")))[split])
        T = gravity_template()
        if a.sigma > 0:
            prio = lambda key: noisy_priority(T, a.k, a.seed, a.sigma, a.ell, key)
        else:
            prio = template_priority(T, 1, a.seed)
        lay = np.zeros((len(fn), a.k if a.sigma > 0 else 1) + T.shape, np.float32)
    N, K, S, _ = lay.shape
    meta = frames.set_index("filename").loc[fn]
    A = np.clip(model.predict(meta[[a.clock]].to_numpy(float)), 0.0, 1.0)
    seq = (meta["exp"] + "_" + meta["face"]).to_numpy()
    ts = meta["timestep"].to_numpy()

    out = apply_area(lay, fn, A, seq, ts, a.seed, prio)
    method = f"{a.shape}_shape+{a.clock}_area"
    layout_io.save_layout_npz(a.out, fn, out, method, split)

    true = meta["R_spall"].to_numpy()
    di = meta["DI"].to_numpy()
    hi = np.argsort(di)[-int(np.ceil(N / 3)):]
    lv = meta["level"].to_numpy()
    area = out.mean(axis=(1, 2, 3))
    rep = {"clock": a.clock, "model": mname, "split": split, "n": int(N),
           "target_bias_all": float((A - true).mean()), "target_bias_hiDI": float((A - true)[hi].mean()),
           "layout_bias_hiDI": float((area - true)[hi].mean()),
           **({"ofp_area_bias_hiDI": float((lay.mean(axis=(1, 2, 3)) - true)[hi].mean())} if a.shape == "ofp" else {}),
           **{f"target_bias_{int(L)}pct": float((A - true)[lv == L].mean()) for L in (4.0, 6.0, 8.0) if (lv == L).any()}}
    with open(os.path.splitext(a.out)[0] + "_report.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
