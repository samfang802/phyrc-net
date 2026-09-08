#!/usr/bin/env python3
"""Union all AnyLabel ``spall`` polygons onto one mask; write R_spall CSV.

Overlapping polygons paint the same canvas, so shared pixels count once.
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ann_dir",
        type=str,
        default="./Data/DamagePattern/anylabel",
    )
    p.add_argument(
        "--out_csv",
        type=str,
        default="./Data/DamagePattern/anylabel_r_spall.csv",
    )
    p.add_argument("--label", type=str, default="spall")
    return p.parse_args()


def _points_xy(shape: dict) -> np.ndarray:
    pts = np.asarray(shape.get("points") or [], dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.int32)
    return np.round(pts[:, :2]).astype(np.int32)


def r_spall_from_json(path: str, label: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    h = int(data.get("imageHeight") or 0)
    w = int(data.get("imageWidth") or 0)
    if h <= 0 or w <= 0:
        raise ValueError(f"Missing image size in {path}")
    image_path = str(data.get("imagePath") or os.path.basename(path))
    stem, ext = os.path.splitext(image_path)
    filename = image_path if ext else f"{stem}.jpg"

    mask = np.zeros((h, w), dtype=np.uint8)
    n_spall = 0
    for shape in data.get("shapes") or []:
        if str(shape.get("label") or "") != label:
            continue
        if str(shape.get("shape_type") or "polygon") != "polygon":
            continue
        pts = _points_xy(shape)
        if pts.shape[0] < 3:
            continue
        n_spall += 1
        cv2.fillPoly(mask, [pts], 255)

    n_pixels = int(h * w)
    n_spall_pixels = int(np.count_nonzero(mask))
    r_spall = float(n_spall_pixels) / float(n_pixels)
    return {
        "filename": filename,
        "R_spall": r_spall,
        "n_spall": n_spall,
        "n_spall_pixels": n_spall_pixels,
        "imageWidth": w,
        "imageHeight": h,
    }


def main():
    cli = _parse_args()
    ann_dir = os.path.abspath(cli.ann_dir)
    files = sorted(fn for fn in os.listdir(ann_dir) if fn.endswith(".json"))
    if not files:
        raise RuntimeError(f"No json in {ann_dir}")

    rows = []
    for fn in files:
        rows.append(r_spall_from_json(os.path.join(ann_dir, fn), cli.label))
    df = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)

    out_csv = os.path.abspath(cli.out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    n_pos = int((df["R_spall"] > 0).sum())
    print(
        f"[anylabel] n={len(df)} label={cli.label} with_spall={n_pos} "
        f"R_spall mean={df['R_spall'].mean():.6f} max={df['R_spall'].max():.6f}",
        flush=True,
    )
    print(f"[anylabel] → {out_csv}", flush=True)


if __name__ == "__main__":
    main()
