#!/usr/bin/env python3
"""AnyLabel ``spall`` polygons → per-frame 128×128 truncated SDF cache (phase-0 layout-oracle).

Same rasterization as export_anylabel_r_spall.py (union of polygons at the original resolution).
Then the generator's geometry: a plain resize to image_size × image_size (v2.Resize, no crop, no
aspect keep; src/utils.py). The binary mask at 128 is coverage ≥ 0.5 (INTER_AREA).

SDF sign: positive inside spall, negative outside, in 128-px units, truncated to ±trunc_px and
divided by trunc_px → [-1, 1]. A frame without spall polygons is all −1.
The dataset (--layout_cond) reads this file; 0 everywhere is the "no layout" (dropped) input.
Horizontal flips are applied in the dataset, not here.

npz keys: filename (N,), sdf float16 (N,S,S), mask uint8 (N,S,S), r_spall float32 (N,) (full-res
fraction, = anylabel_r_spall.csv), r_spall128 float32 (N,) (fraction of the 128 mask), meta (json).
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ann_dir", default="./Data/DamagePattern/anylabel")
    p.add_argument("--out", default="./Data/DamagePattern/anylabel_layout_sdf128.npz")
    p.add_argument("--label", default="spall")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--trunc_px", type=float, default=8.0)
    return p.parse_args()


def spall_mask_full(data: dict, label: str = "spall") -> np.ndarray:
    h, w = int(data["imageHeight"]), int(data["imageWidth"])
    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in data.get("shapes") or []:
        if str(shape.get("label") or "") != label or str(shape.get("shape_type") or "polygon") != "polygon":
            continue
        pts = np.asarray(shape.get("points") or [], dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [np.round(pts[:, :2]).astype(np.int32)], 255)
    return mask


def truncated_sdf(mask: np.ndarray, trunc_px: float) -> np.ndarray:
    """mask uint8 {0,1} → SDF in [-1, 1], positive inside."""
    if mask.max() == 0:
        return -np.ones(mask.shape, dtype=np.float32)
    if mask.min() == 1:
        return np.ones(mask.shape, dtype=np.float32)
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)  # distance to nearest 0
    outside = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 5)
    # Pixel centers: a boundary pixel inside is 1 from the nearest outside pixel; shift by 0.5.
    sdf = np.where(mask > 0, inside - 0.5, -(outside - 0.5))
    return (np.clip(sdf, -trunc_px, trunc_px) / trunc_px).astype(np.float32)


def main():
    cli = _parse_args()
    files = sorted(f for f in os.listdir(cli.ann_dir) if f.endswith(".json"))
    s = int(cli.image_size)
    names, sdfs, masks, r_full, r_128 = [], [], [], [], []
    for fn in files:
        with open(os.path.join(cli.ann_dir, fn), encoding="utf-8") as f:
            data = json.load(f)
        image_path = str(data.get("imagePath") or fn)
        stem, ext = os.path.splitext(os.path.basename(image_path))
        names.append(f"{stem}{ext or '.jpg'}")
        full = spall_mask_full(data, cli.label)
        cov = cv2.resize((full > 0).astype(np.float32), (s, s), interpolation=cv2.INTER_AREA)
        m = (cov >= 0.5).astype(np.uint8)
        masks.append(m)
        sdfs.append(truncated_sdf(m, float(cli.trunc_px)).astype(np.float16))
        r_full.append(float((full > 0).mean()))
        r_128.append(float(m.mean()))
    meta = {
        "ann_dir": os.path.abspath(cli.ann_dir),
        "label": cli.label,
        "image_size": s,
        "trunc_px": float(cli.trunc_px),
        "sign": "positive inside spall; no-spall frame = -1; dropped layout = 0",
        "resize": "INTER_AREA coverage >= 0.5 after union fill at full resolution",
    }
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    np.savez_compressed(
        cli.out,
        filename=np.asarray(names),
        sdf=np.stack(sdfs),
        mask=np.stack(masks),
        r_spall=np.asarray(r_full, dtype=np.float32),
        r_spall128=np.asarray(r_128, dtype=np.float32),
        meta=np.asarray(json.dumps(meta)),
    )
    r_full, r_128 = np.asarray(r_full), np.asarray(r_128)
    print(
        f"[layout_sdf] n={len(names)} with_spall={(r_full > 0).sum()} "
        f"|r128 - r_full| mean={np.abs(r_128 - r_full).mean():.5f} max={np.abs(r_128 - r_full).max():.5f} → {cli.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
