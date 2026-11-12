#!/usr/bin/env python3
"""Irreversibility curve of generated sequences vs the real ones (no training, CPU only).

Thin dark structure (cracks, grid lines, spall edges) = pixels where a band-pass of the image is below
-thr. For frame t of a face and lag L, against frame t-L of the same face:
  born  dark at t,   no dark at t-L within a (2·tol+1)² window
  died  dark at t-L, no dark at t   within the same window
born/died > 1 means structure accumulates; a reversible process (exposure, crack opening, an
independent sampler) gives ~1 at every lag. Real data (raw JPG, 256 px) rises from 1.08 at lag 1 to
3.09 at lag 24, and at lag 2 (same load sign) is ~1.3 on the first two cycles of a drift level vs ~1.0
on cycles 5-6 (Kaiser effect).

  keep  1 - died / dark(t-L)   share of the earlier structure still present L frames later

Step births (docs/redesign/phase1_ilf.md §12.1): frames alternate load sign, so lag-1 "born" is mostly a crack
reopening. A step birth at t is dark at t, present (within tol) at t+2 and t+4, absent at t-2, t-4 and t-6 (same-sign
frames only; frames 6 .. T-5):
  step_birth.per_frame     mean step births per frame (pixels)
  step_birth.c1_ratio      births per frame on the first cycle of a drift level (suffix 1-2) / on the other frames
  step_birth.s2_ratio      same for the first - peak only (suffix 2)
Real births concentrate on new levels (ratios well above 1); a sampler that grows new structure evenly over the
cycles has ratios near 1.

Per model the npz `fake` (N, K, H, W) holds K trajectories when the sampler walks the sequence
(refiner_sequence); for single-frame samplers sample k of different frames are independent draws,
which is the "no memory" reference. `real` of the first npz is the reference curve.
cross_trajectory: the earlier frame comes from trajectory k+1 instead of k, i.e. what the conditions
alone (g, layout, shared noise) explain; keep above it is persistence the sampler carries itself.

  python scripts/eval/irreversibility_curve.py --split test \
      --model v12gt=results/eval_suite/v12_compare/test/samples/v12gt_010000_grav.npz \
      --model refiner=results/eval_suite/v12gtrefiner/test/samples/v12gtrefiner_006000_grav.npz \
      --out results/eval_suite/irreversibility/test.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter

LAGS = (1, 2, 6, 12, 24)


def dark_masks(imgs, thr, s_fine, s_coarse):
    """imgs uint8 (T, H, W) -> bool (T, H, W)."""
    a = imgs.astype(np.float32) / 255.0
    hp = np.stack([gaussian_filter(x, s_fine) - gaussian_filter(x, s_coarse) for x in a])
    return hp < -thr


def pair_counts(m, lags, tol, m_then=None):
    """m bool (T, H, W) in timestep order -> {lag: (born, died, dark_t, dark_prev)} per pair.
    m_then: the earlier frames come from this array instead (another trajectory: condition-only control)."""
    m_then = m if m_then is None else m_then
    size = 2 * tol + 1
    dil = np.stack([maximum_filter(x, size=size) for x in m])
    dil_then = dil if m_then is m else np.stack([maximum_filter(x, size=size) for x in m_then])
    out = {}
    for lag in lags:
        if lag >= len(m):
            continue
        born = (m[lag:] & ~dil_then[:-lag]).sum(axis=(1, 2))
        died = (m_then[:-lag] & ~dil[lag:]).sum(axis=(1, 2))
        out[lag] = (born, died, m[lag:].sum(axis=(1, 2)), m_then[:-lag].sum(axis=(1, 2)))
    return out


def step_births(m, tol):
    """m bool (T, H, W) -> (T,) step-birth counts, (T,) valid."""
    size = 2 * tol + 1
    pres = np.stack([maximum_filter(x, size=size) for x in m])
    T = len(m)
    n = np.zeros(T)
    valid = np.zeros(T, dtype=bool)
    for t in range(6, T - 4):
        n[t] = (m[t] & pres[t + 2] & pres[t + 4] & ~(pres[t - 2] | pres[t - 4] | pres[t - 6])).sum()
        valid[t] = True
    return n, valid


def curve(seqs, lags, tol):
    """seqs: list of (masks (T,H,W), cycles (T,)[, earlier-frame masks (T,H,W)]) -> dict of summaries."""
    acc = {lag: np.zeros(4) for lag in lags}
    cyc = {}
    sb = {"all": np.zeros(2), "c1": np.zeros(2), "c1_not": np.zeros(2), "s2": np.zeros(2), "s2_not": np.zeros(2)}
    for m, cycles, *then in seqs:
        if not then:
            n, v = step_births(m, tol)
            c = np.asarray(cycles)
            for key, sel in (("all", v), ("c1", v & ((c == 1) | (c == 2))), ("c1_not", v & ~((c == 1) | (c == 2))),
                             ("s2", v & (c == 2)), ("s2_not", v & (c != 2))):
                sb[key] += [n[sel].sum(), sel.sum()]
        pc = pair_counts(m, lags, tol, then[0] if then else None)
        for lag, (b, d, n, n0) in pc.items():
            acc[lag] += [b.sum(), d.sum(), n.sum(), n0.sum()]
        if 2 in pc:
            b, d, _, _ = pc[2]
            for c, bb, dd in zip(cycles[2:], b, d):
                s = cyc.setdefault(int(c), np.zeros(2))
                s += [bb, dd]
    res = {"dark_frac": float(np.mean([s[0].mean() for s in seqs]))}
    if sb["all"][1] > 0:
        r = {k: v[0] / max(v[1], 1) for k, v in sb.items()}
        res["step_birth"] = {"per_frame": float(r["all"]), "c1_ratio": float(r["c1"] / max(r["c1_not"], 1e-9)),
                             "s2_ratio": float(r["s2"] / max(r["s2_not"], 1e-9))}
    for lag in lags:
        b, d, n, n0 = acc[lag]
        if n0 == 0:
            continue
        res[f"lag{lag}"] = {"born_over_died": float(b / max(d, 1)), "born_frac": float(b / max(n, 1)),
                            "keep": float(1 - d / n0)}
    new = sum(cyc[c] for c in (1, 2) if c in cyc)
    rep = sum(cyc[c] for c in (5, 6) if c in cyc)
    if isinstance(new, np.ndarray) and isinstance(rep, np.ndarray):
        res["kaiser_lag2"] = {"new_cycles_1_2": float(new[0] / max(new[1], 1)),
                              "repeat_cycles_5_6": float(rep[0] / max(rep[1], 1))}
    return res


def sequences(filenames, label):
    """Row indices of each (exp, direction) in timestep order, and the cycle of each row."""
    lab = label.set_index("filename").loc[list(filenames)].reset_index()
    lab["row"] = np.arange(len(lab))
    lab["cycle"] = lab.filename.str.extract(r"_(\d+)\.jpg$")[0].astype(int)
    out = []
    for _, g in lab.groupby(["exp", "direction"], sort=True):
        g = g.sort_values("timestep", kind="stable")
        out.append((g.row.to_numpy(), g.cycle.to_numpy()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True, metavar="NAME=NPZ")
    p.add_argument("--split", default="test")
    p.add_argument("--label", default="Data/DamagePattern/label.csv")
    p.add_argument("--thr", type=float, nargs="+", default=[0.04, 0.07])
    p.add_argument("--sigma", type=float, nargs=2, default=[0.5, 2.0], help="band-pass gaussian sigmas (px)")
    p.add_argument("--tol", type=int, default=1, help="position tolerance (px) for born / died")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    label = pd.read_csv(a.label)
    report = {"split": a.split, "thr": a.thr, "sigma": a.sigma, "tol": a.tol, "lags": list(LAGS), "models": {}}
    real_done = False
    for spec in a.model:
        name, path = spec.split("=", 1)
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        seqs = sequences(z["filename"], label)
        fake = z["fake"]
        for thr in a.thr:
            key = f"thr{thr:g}"
            if not real_done:
                rs = [(dark_masks(z["real"][rows], thr, *a.sigma), cyc) for rows, cyc in seqs]
                report["models"].setdefault("real", {})[key] = curve(rs, LAGS, a.tol)
            K = fake.shape[1]
            masks = [[dark_masks(fake[rows, k], thr, *a.sigma) for rows, _ in seqs] for k in range(K)]
            per_k = [curve([(masks[k][j], cyc) for j, (_, cyc) in enumerate(seqs)], LAGS, a.tol) for k in range(K)]
            report["models"].setdefault(name, {})[key] = _mean_std(per_k)
            if K > 1:
                cross = [curve([(masks[k][j], cyc, masks[(k + 1) % K][j]) for j, (_, cyc) in enumerate(seqs)],
                               LAGS, a.tol) for k in range(K)]
                report["models"][name][key]["cross_trajectory"] = _mean_std(cross)
        report["models"][name]["sampler"] = meta.get("sampler", "")
        report["models"][name]["npz"] = os.path.abspath(path)
        real_done = True
        print(f"[irrev] {name}: {meta.get('sampler', '')} {fake.shape}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    _print_table(report)


def _mean_std(per_k):
    """Mean (and std over the K trajectories) of every leaf."""
    def walk(items):
        if isinstance(items[0], dict):
            return {k: walk([d[k] for d in items]) for k in items[0] if all(k in d for d in items)}
        v = np.asarray(items, dtype=np.float64)
        return {"mean": float(v.mean()), "std": float(v.std())}
    return walk(per_k)


def _val(x):
    return x["mean"] if isinstance(x, dict) else x


def _print_table(report):
    for thr in report["thr"]:
        key = f"thr{thr:g}"
        lags = " ".join(f"L{l:<4d}" for l in report["lags"])
        print(f"\n== {report['split']}  thr {thr:g}  born/died {lags}| keep@1 keep@6 | Kaiser lag2 new/repeat | dark"
              f" | step births/frame c1 s2")
        for name, r in report["models"].items():
            rows = [(name, r[key])]
            if "cross_trajectory" in r[key]:
                rows.append(("  (cross-trajectory)", r[key]["cross_trajectory"]))
            for label, c in rows:
                bd = " ".join(f"{_val(c[f'lag{l}']['born_over_died']):5.2f}" if f"lag{l}" in c else "   - "
                              for l in report["lags"])
                keep = " ".join(f"{_val(c[f'lag{l}']['keep']):.2f}" for l in (1, 6) if f"lag{l}" in c)
                ks = c.get("kaiser_lag2", {})
                kz = f"{_val(ks['new_cycles_1_2']):.2f}/{_val(ks['repeat_cycles_5_6']):.2f}" if ks else "-"
                st = c.get("step_birth")
                sbt = (f" | {_val(st['per_frame']):6.1f} {_val(st['c1_ratio']):.2f} {_val(st['s2_ratio']):.2f}"
                       if st else "")
                print(f"  {label:28s} {bd} | {keep} | {kz} | {_val(c['dark_frac']):.3f}{sbt}")


if __name__ == "__main__":
    main()
