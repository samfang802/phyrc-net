#!/usr/bin/env python3
"""Extract 128x128 GT images closest to a target DI for train/val inference anchors."""

import argparse
import json
import os

import pandas as pd
from PIL import Image


def _parse_args():
    parser = argparse.ArgumentParser(description="Extract GT images at target DI (no inference)")
    parser.add_argument("--target_di", type=float, default=0.75)
    parser.add_argument("--train_anchor_exp", type=str, default="R307")
    parser.add_argument("--val_anchor_exp", type=str, default="R615")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument(
        "--data_path",
        type=str,
        default="Data/DamagePattern/id_list_no_leak.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/inference_gt_extract",
    )
    return parser.parse_args()


def _pick_closest_row(label_df, split_ids, exp, target_di):
    sub = label_df[label_df["filename"].isin(split_ids) & (label_df["exp"] == exp)].copy()
    if sub.empty:
        raise RuntimeError(f"No samples found for split exp={exp}")
    sub["di_diff"] = (sub["DI"] - target_di).abs()
    return sub.sort_values(["di_diff", "timestep"]).iloc[0]


def _save_gt(image_path, out_path, image_size):
    image = Image.open(image_path)
    image = image.resize((image_size, image_size), Image.BILINEAR)
    image.save(out_path)


def main():
    cli = _parse_args()
    root = "Data/DamagePattern"
    split_ids = json.load(open(cli.data_path))
    label_df = pd.read_csv(os.path.join(root, "label.csv"))
    os.makedirs(cli.output_dir, exist_ok=True)

    for split, exp in (("train", cli.train_anchor_exp), ("val", cli.val_anchor_exp)):
        row = _pick_closest_row(label_df, split_ids[split], exp, cli.target_di)
        src = os.path.join(root, "images", row["filename"])
        out_png = os.path.join(
            cli.output_dir,
            f"gt_{split}_{exp}_di{cli.target_di:.2f}_t{int(row['timestep'])}.png",
        )
        _save_gt(src, out_png, cli.image_size)
        print(
            f"[extract_gt] {split}: exp={row['exp']} t={int(row['timestep'])} "
            f"GT_DI={float(row['DI']):.4f} src={row['filename']} -> {out_png}",
            flush=True,
        )

    print(f"[extract_gt] Done. Output dir: {cli.output_dir}", flush=True)


if __name__ == "__main__":
    main()
