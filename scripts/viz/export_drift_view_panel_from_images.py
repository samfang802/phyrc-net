#!/usr/bin/env python3
"""Drift × DI view panels (SW / NE) from saved images, laid out like export_di_tertile_overview_grids.py.

Same frame selection as `--view_panels` there: test frames of one view (SW = Pull, NE = Push), one
row per drift code 250 / 2000 / 4000 / 6000 / 8000 (labelled 0 / 2 / 4 / 6 / 8 %), frames sorted
by DI and thinned to 6 evenly spaced. Images come from an npz (generate_eval_samples.py layout,
sample index --k) or from a folder of files named after the GT frame.

Examples:
  python scripts/viz/export_drift_view_panel_from_images.py --image_dir cGANckpt/CTR1 \
      --suffix _test.jpg --tag cgan --out_dir cGANckpt/inference_samples
  python scripts/viz/export_drift_view_panel_from_images.py --npz cgan_inpainted_k1.npz \
      --tag cgan_inpainted --out_dir cGANckpt/inference_samples
  python scripts/viz/export_drift_view_panel_from_images.py --gt --tag gt --out_dir ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics.dino_mmd import load_real_uint8

ROWS = ((250, "0"), (2000, "2"), (4000, "4"), (6000, "6"), (8000, "8"))
VIEWS = (("SW", "Pull"), ("NE", "Push"))


def _parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz")
    src.add_argument("--image_dir")
    src.add_argument("--gt", action="store_true", help="Real frames through the generator preprocessing")
    p.add_argument("--k", type=int, default=0, help="Sample index inside the npz")
    p.add_argument("--suffix", default=".jpg", help="--image_dir file name = <GT stem><suffix>")
    p.add_argument("--gt_image_dir", default="Data/DamagePattern/images")
    p.add_argument("--gt_no_clahe", action="store_true", help="--gt_image_dir is already CLAHE'd (inpainted set)")
    p.add_argument("--split", default="test")
    p.add_argument("--data_path", default="Data/DamagePattern/id_list_val_R307.json")
    p.add_argument("--tag", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def _save_panel(rows, out_path, title, max_cols=6):
    """Copy of export_di_tertile_overview_grids._save_drift_time_panel (same figure geometry)."""
    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, max_cols, figsize=(max_cols * 1.15 + 0.5, n_rows * 1.3 + 0.55),
        gridspec_kw=dict(wspace=0.03, hspace=0.2),
    )
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for r, (drift_label, cells) in enumerate(rows):
        axes[r, 0].text(-0.06, 0.5, drift_label, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=9)
        for c, (img, cap) in enumerate(cells):
            axes[r, c].imshow(img, cmap="gray", vmin=0, vmax=255)
            axes[r, c].set_title(cap, fontsize=6.5, pad=1.5)
    fig.suptitle(title, fontsize=12, y=0.99)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.04, top=0.9)
    fig.text(0.005, 0.5, "Drift (%)  ↓", rotation=90, va="center", fontsize=10)
    fig.text(0.55, 0.008, "DI  →", ha="center", fontsize=10)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[view] saved {out_path}", flush=True)


def main():
    cli = _parse_args()
    with open(cli.data_path) as f:
        names = set(json.load(f)[cli.split])
    label = pd.read_csv("Data/DamagePattern/label.csv")
    label = label[label["filename"].isin(names)].reset_index(drop=True)

    if cli.npz:
        z = np.load(cli.npz, allow_pickle=False)
        lookup = {str(n): z["fake"][i, cli.k] for i, n in enumerate(z["filename"])}
        get = lambda n: lookup[n]
    elif cli.gt:
        get = lambda n: load_real_uint8([n], cli.gt_image_dir, use_clahe=not cli.gt_no_clahe)[0]
    else:
        def get(n):
            path = os.path.join(cli.image_dir, os.path.splitext(n)[0] + cli.suffix)
            return np.array(Image.open(path).convert("L").resize((128, 128), Image.BICUBIC))

    picked = {}
    for view, dname in VIEWS:
        sel = label[label["direction"] == dname]
        rows = []
        for code, row_lab in ROWS:
            fr = sel[sel["filename"].str.split("_").str[2].astype(int) == code].sort_values("DI", kind="stable")
            if len(fr) == 0:
                continue
            if len(fr) > 6:
                fr = fr.iloc[np.linspace(0, len(fr) - 1, 6).round().astype(int)]
            rows.append((f"{row_lab}%", [(get(r.filename), f"DI {float(r.DI):.2f}") for r in fr.itertuples()]))
            picked.setdefault(view, []).extend(fr["filename"].tolist())
        _save_panel(rows, os.path.join(cli.out_dir, f"drift_di_{cli.tag}_{view}.png"), f"{view} · Drift × DI ({cli.tag})")
    with open(os.path.join(cli.out_dir, f"drift_di_{cli.tag}_frames.json"), "w") as f:
        json.dump(picked, f, indent=2)


if __name__ == "__main__":
    main()
