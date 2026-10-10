#!/usr/bin/env python3
"""KMeans-inpaint masks vs GT: 3 compact 6×6 pair overviews (Low/Mid/High DI).

Each overview is two 6×6 grids side-by-side (GT | mask), 72 tiles of 128², no labels.
Mask = K=2 dark cluster → open/close 5×5 → all CCs passing top/width gates
(no area gate, no height gate, no largest-only, no fill-below).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.extract_damage_features import DamageFeatureExtractor


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=str,
        default="./Data/DamagePattern_inpaint_selfcondnoDI",
    )
    p.add_argument("--data_path", type=str, default="id_list_val_R307.json")
    p.add_argument(
        "--split",
        type=str,
        default="all",
        help="train|val|test|all (all = union of json splits)",
    )
    p.add_argument("--grid_size", type=int, default=6)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--open_ksize", type=int, default=5)
    p.add_argument("--close_ksize", type=int, default=5)
    p.add_argument("--gap", type=int, default=0, help="Pixels between tiles")
    p.add_argument(
        "--pair_gap",
        type=int,
        default=1,
        help="Pixels between the GT 6×6 and the spall 6×6",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="./results/DamageQuantification/spall_mask_kmeans",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _select_uniform(rows, di_lo, di_hi, n):
    in_bin = [r for r in rows if di_lo <= float(r["DI"]) <= di_hi]
    search = in_bin if len(in_bin) >= n else list(rows)
    targets = np.linspace(di_lo, di_hi, n)
    selected = []
    remaining = list(search)
    for t in targets:
        cand = remaining if remaining else search
        in_rem = [r for r in cand if di_lo <= float(r["DI"]) <= di_hi]
        use = in_rem if in_rem else cand
        pick = min(use, key=lambda r: (abs(float(r["DI"]) - float(t)), str(r["filename"])))
        selected.append(pick)
        remaining = [r for r in remaining if r["filename"] != pick["filename"]]
    return selected


def _load_gt_gray(path, size):
    img = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return img


def _spall_to_image(mask_u8, size):
    if mask_u8.shape[0] != size or mask_u8.shape[1] != size:
        mask_u8 = np.array(
            Image.fromarray(mask_u8).resize((size, size), Image.Resampling.NEAREST)
        )
    return Image.fromarray(mask_u8, mode="L")


def _paste_grid(canvas, tiles, origin_x, origin_y, grid, tile, gap):
    for i, im in enumerate(tiles):
        r, c = divmod(i, grid)
        x = origin_x + c * (tile + gap)
        y = origin_y + r * (tile + gap)
        canvas.paste(im, (x, y))


def _save_pair_overview(gt_tiles, spall_tiles, out_path, grid, tile, gap, pair_gap):
    n = grid
    block_w = n * tile + (n - 1) * gap
    block_h = n * tile + (n - 1) * gap
    canvas = Image.new("L", (block_w * 2 + pair_gap, block_h), 255)
    _paste_grid(canvas, gt_tiles, 0, 0, n, tile, gap)
    _paste_grid(canvas, spall_tiles, block_w + pair_gap, 0, n, tile, gap)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path, optimize=True)
    print(f"[spall] {out_path}  {canvas.size[0]}x{canvas.size[1]}", flush=True)


def main():
    cli = _parse_args()
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
    if len(rows) < n_per:
        raise RuntimeError(f"Not enough images: {len(rows)} < {n_per}")

    extractor = DamageFeatureExtractor(
        image_size=(size, size),
        open_ksize=int(cli.open_ksize),
        close_ksize=int(cli.close_ksize),
        min_spall_area=0.0,
        kmeans_k=2,
        fill_below_spall_top=False,
        keep_largest_component=False,
        use_object_mask=True,
        mask_dir=mask_dir,
    )

    di = label["DI"].to_numpy(dtype=np.float64)
    edges = np.quantile(di, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]).tolist()
    edges[-1] = float(di.max()) + 1e-9
    group_names = ["low", "mid", "high"]
    print(
        f"[spall] n={len(rows)} split={cli.split} k=2 open={cli.open_ksize} "
        f"min_area=0 keep_largest=False fill_below=False DI edges={edges}",
        flush=True,
    )

    out_dir = os.path.abspath(cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    used = set()
    meta = {"root": root, "split": cli.split, "edges": edges, "groups": []}

    for g_idx, g_name in enumerate(group_names):
        di_lo = float(edges[g_idx])
        di_hi = float(edges[g_idx + 1])
        hi_sel = di_hi - 1e-12 if g_idx < 2 else di_hi
        avail = [r for r in rows if r["filename"] not in used]
        picked = _select_uniform(avail, di_lo, hi_sel, n_per)
        for r in picked:
            used.add(r["filename"])

        gt_tiles, spall_tiles = [], []
        items = []
        for r in tqdm(picked, desc=g_name, leave=True):
            path = os.path.join(images_dir, str(r["filename"]))
            gt = _load_gt_gray(path, size)
            result = extractor.evaluate(path)
            spall = _spall_to_image(result["spall_mask"], size)
            gt_tiles.append(gt)
            spall_tiles.append(spall)
            items.append(
                {
                    "filename": str(r["filename"]),
                    "exp": str(r["exp"]),
                    "DI": float(r["DI"]),
                    "R_spall": float(result["R_spall"]),
                }
            )

        out_path = os.path.join(out_dir, f"k2_morph5_cc_all_gt_vs_mask_{g_name}_6x6.png")
        _save_pair_overview(
            gt_tiles,
            spall_tiles,
            out_path,
            grid,
            size,
            int(cli.gap),
            int(cli.pair_gap),
        )
        meta["groups"].append(
            {
                "name": g_name,
                "di_lo": di_lo,
                "di_hi": float(min(di_hi, float(di.max()))),
                "figure": out_path,
                "items": items,
            }
        )

    meta_path = os.path.join(out_dir, "k2_morph5_cc_all_gt_vs_mask_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[spall] meta → {meta_path}", flush=True)


if __name__ == "__main__":
    main()
