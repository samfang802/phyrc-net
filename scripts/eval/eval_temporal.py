#!/usr/bin/env python3
"""Temporal consistency of generated sequences (src/metrics/temporal.py) against the real loading history.

For every face of the column, sample k of each frame in timestep order is one trajectory. Reported per
model and for the GT (K = 1), pooled over all consecutive pairs of all trajectories:
  flicker, tone_jump, struct_flicker, persistence, crack_keep, regress   (see src/metrics/temporal.py)
  flicker_ratio = model flicker / GT flicker on the same pairs (1 = changes as much as the photos)
  brightness_slope (gray levels per frame) and its difference from the GT's
SE: leave-one-block-out jackknife over the pairs, block = face × drift level of the first frame
(dm.frame_block). Pixel thresholds are calibrated on intact training frames, as in eval_damage_texture.py.

The plot shows, per face, the damage area and the mean brightness along the sequence (GT and the
first trajectory of every model).

Example:
  python scripts/eval/eval_temporal.py --samples gn=gn.npz refiner=refiner.npz --no_clahe --out_dir out/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics import damage_texture as dt
from src.metrics import dino_mmd as dm
from src.metrics import temporal as tp

KEYS = ("flicker", "tone_jump", "struct_flicker", "persistence", "crack_keep", "regress")


def _parse_args():
    p = argparse.ArgumentParser(description="Temporal consistency metrics")
    p.add_argument("--samples", nargs="+", required=True, metavar="NAME=NPZ")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--data_path", default="Data/DamagePattern/id_list_val_R307.json")
    p.add_argument("--image_dir", default="Data/DamagePattern_inpaint_selfcondnoDI/images")
    p.add_argument("--no_clahe", action="store_true")
    p.add_argument("--intact_max_code", type=int, default=250)
    return p.parse_args()


def _pool(seqs_by_k, thr, trajs, blocks_of_row):
    """seqs_by_k: (N, K, H, W). Pairs of all trajectories and all k pooled; block of a pair = its first frame."""
    vals = {k: [] for k in KEYS}
    blocks, slopes, curves = [], [], {}
    for key, rows in trajs.items():
        for k in range(seqs_by_k.shape[1]):
            st, slope, mean_curve, area_curve = tp.pair_stats(seqs_by_k[rows, k], thr)
            for name in KEYS:
                vals[name].append(st[name])
            blocks.append(blocks_of_row[rows[:-1]])
            slopes.append(slope)
            if k == 0:
                curves[key] = {"mean": mean_curve.tolist(), "damage_area": area_curve.tolist()}
    return {n: np.concatenate(v) for n, v in vals.items()}, np.concatenate(blocks), np.asarray(slopes), curves


def _jk(x, blocks):
    ok = np.isfinite(x)
    r, loo = dm.mean_jackknife(x[ok], blocks[ok])
    return {"value": round(r["value"], 5), "se": round(r["se"], 5), "n_pairs": int(ok.sum())}


def _plot(curves, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    faces = list(curves["gt"])
    fig, axes = plt.subplots(2, len(faces), figsize=(5.5 * len(faces), 6.5), squeeze=False)
    for j, face in enumerate(faces):
        for name, c in curves.items():
            if face not in c:
                continue
            style = dict(color="black", lw=2.2) if name == "gt" else dict(lw=1.2, alpha=0.85)
            axes[0, j].plot(c[face]["damage_area"], label=name, **style)
            axes[1, j].plot(c[face]["mean"], label=name, **style)
        axes[0, j].set_title(f"{face}: damage area (first trajectory)")
        axes[1, j].set_title(f"{face}: mean brightness")
        axes[1, j].set_xlabel("frame index (timestep order)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    cli = _parse_args()
    os.makedirs(cli.out_dir, exist_ok=True)
    with open(cli.data_path) as f:
        ids = json.load(f)
    label = pd.read_csv("Data/DamagePattern/label.csv")
    clahe = not cli.no_clahe

    fakes, filenames = {}, None
    for spec in cli.samples:
        name, path = spec.split("=", 1)
        z = np.load(path, allow_pickle=False)
        fn = [str(n) for n in z["filename"]]
        if filenames is None:
            filenames = fn
        elif fn != filenames:
            raise ValueError(f"{name}: npz rows differ from the first npz")
        fakes[name] = z["fake"]
    lab = label.set_index("filename").loc[filenames].reset_index()
    trajs = tp.trajectories(lab)
    blocks = np.asarray([dm.frame_block(n) for n in filenames])

    train = list(ids["train"])
    codes = np.array([int(n.split("_")[2]) for n in train])
    thr = dt.calibrate(dm.load_real_uint8([n for n, c in zip(train, codes) if c <= cli.intact_max_code], cli.image_dir, use_clahe=clahe))
    gt = dm.load_real_uint8(filenames, cli.image_dir, use_clahe=clahe)[:, None]

    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "args": vars(cli), "thresholds": thr,
              "trajectories": {k: int(len(v)) for k, v in trajs.items()}, "models": {}}
    gt_vals, pair_blocks, gt_slopes, gt_curves = _pool(gt, thr, trajs, blocks)
    report["gt"] = {**{n: _jk(gt_vals[n], pair_blocks) for n in KEYS}, "brightness_slope": round(float(gt_slopes.mean()), 4)}
    curves = {"gt": gt_curves}
    for name, fake in fakes.items():
        v, b, slopes, c = _pool(fake, thr, trajs, blocks)
        K = fake.shape[1]
        # Pairs are pooled k-major within each trajectory; tile the GT flicker the same way.
        gt_rep = np.concatenate([np.tile(tp.pair_stats(gt[rows, 0], thr)[0]["flicker"], K) for rows in trajs.values()])
        res = {n: _jk(v[n], b) for n in KEYS}
        res["flicker_ratio"] = _jk(v["flicker"] / np.maximum(gt_rep, 1e-6), b)
        res["brightness_slope"] = round(float(np.nanmean(slopes)), 4)
        res["brightness_slope_minus_gt"] = round(float(np.nanmean(slopes) - gt_slopes.mean()), 4)
        report["models"][name] = res
        curves[name] = c

    path = os.path.join(cli.out_dir, "temporal.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    _plot(curves, os.path.join(cli.out_dir, "temporal.png"))

    fmt = lambda d: f"{d['value']:.3f}±{d['se']:.3f}"
    print(f"[temporal] trajectories {report['trajectories']}")
    print(f"{'':16s}" + "".join(f"{k:>18s}" for k in (*KEYS, "flicker_ratio", "bright_slope")))
    g = report["gt"]
    print(f"{'gt':16s}" + "".join(f"{fmt(g[k]):>18s}" for k in KEYS) + f"{'1':>18s}{g['brightness_slope']:>18.3f}")
    for name, r in report["models"].items():
        print(f"{name:16s}" + "".join(f"{fmt(r[k]):>18s}" for k in (*KEYS, "flicker_ratio")) + f"{r['brightness_slope']:>18.3f}")
    print(f"[temporal] wrote {path} and temporal.png")


if __name__ == "__main__":
    main()
