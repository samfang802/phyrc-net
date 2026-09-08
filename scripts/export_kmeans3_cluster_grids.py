#!/usr/bin/env python3
"""Preview KMeans k=3 clusters K=1 (darkest) and K=2 (mid) after open/close.

Each tertile overview: GT | K=1 | K=2 as three 6×6 blocks, 128² tiles, no labels.
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
from PIL import Image
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.export_spall_mask_tertile_grids import (
    _load_gt_gray,
    _parse_args,
    _paste_grid,
    _select_uniform,
    _spall_to_image,
)
from scripts.extract_damage_features import DamageFeatureExtractor


def _save_triple(gt, k1, k2, out_path, grid, tile, gap, pair_gap):
    n = grid
    block_w = n * tile + (n - 1) * gap
    block_h = n * tile + (n - 1) * gap
    canvas = Image.new("L", (block_w * 3 + pair_gap * 2, block_h), 255)
    _paste_grid(canvas, gt, 0, 0, n, tile, gap)
    _paste_grid(canvas, k1, block_w + pair_gap, 0, n, tile, gap)
    _paste_grid(canvas, k2, 2 * (block_w + pair_gap), 0, n, tile, gap)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path, optimize=True)
    print(f"[k3] {out_path}  {canvas.size[0]}x{canvas.size[1]}", flush=True)


def main():
    cli = _parse_args()
    cli.out_dir = getattr(cli, "out_dir", None) or "./results/DamageQuantification/spall_mask_kmeans"
    root = os.path.abspath(cli.root)
    images_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "mask")
    size = int(cli.image_size)
    grid = int(cli.grid_size)
    n_per = grid * grid

    split_json = json.load(open(os.path.join(root, cli.data_path)))
    if cli.split == "all":
        names = list(dict.fromkeys(sum((split_json[k] for k in split_json), [])))
    else:
        names = list(split_json[cli.split])
    label = pd.read_csv(os.path.join(root, "label.csv"))
    label = label[label["filename"].isin(names)].reset_index(drop=True)
    rows = label.to_dict("records")
    di = label["DI"].to_numpy(dtype=float)
    edges = list(__import__("numpy").quantile(di, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]))
    edges[-1] = float(di.max()) + 1e-9
    print(f"[k3] n={len(rows)} kmeans_k=3 open={cli.open_ksize} DI edges={edges}", flush=True)

    extractor = DamageFeatureExtractor(
        image_size=(size, size),
        open_ksize=int(cli.open_ksize),
        close_ksize=int(cli.close_ksize),
        kmeans_k=3,
        fill_below_spall_top=True,
        use_object_mask=True,
        mask_dir=mask_dir,
    )

    out_dir = os.path.abspath(cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    used = set()
    for g_idx, g_name in enumerate(["low", "mid", "high"]):
        di_lo = float(edges[g_idx])
        di_hi = float(edges[g_idx + 1])
        hi_sel = di_hi - 1e-12 if g_idx < 2 else di_hi
        avail = [r for r in rows if r["filename"] not in used]
        picked = _select_uniform(avail, di_lo, hi_sel, n_per)
        for r in picked:
            used.add(r["filename"])

        gt_tiles, k1_tiles, k2_tiles = [], [], []
        for r in tqdm(picked, desc=g_name):
            path = os.path.join(images_dir, str(r["filename"]))
            clusters = extractor.cluster_masks_from_path(path)
            gt_tiles.append(_load_gt_gray(path, size))
            k1_tiles.append(_spall_to_image(clusters[1], size))
            k2_tiles.append(_spall_to_image(clusters[2], size))

        _save_triple(
            gt_tiles,
            k1_tiles,
            k2_tiles,
            os.path.join(out_dir, f"kmeans3_gt_k1_k2_{g_name}_6x6.png"),
            grid,
            size,
            int(cli.gap),
            int(cli.pair_gap),
        )


if __name__ == "__main__":
    main()
