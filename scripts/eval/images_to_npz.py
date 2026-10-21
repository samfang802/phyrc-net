#!/usr/bin/env python3
"""Saved images of any model → the sample npz the evaluation scripts read (src/metrics/samples_io.py).

Use this for models that are not CFM runs of this repo (the cGAN baseline, another codebase, ...).
Each image must be named after its GT frame:

  <image_dir>/<GT stem><suffix>                 one sample per condition (K = 1)
  <image_dir>/<GT stem><suffix with {k}>        K samples, e.g. --suffix "_k{k}.png" --k 8

Images are converted to gray (PIL "L") and resized to 128 with bicubic, the same as the cGAN
baseline in docs/eval/results.md. Pass --clahe only if the model's output is not already in the
CLAHE'd domain of the training images (the CFM runs and the cGAN output already are).
Rows follow the split's label.csv order; a missing image is an error.

Example:
  python scripts/eval/images_to_npz.py --image_dir cGANckpt/CTR1 --suffix _test.jpg --out cgan_test.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics.dino_mmd import load_real_uint8
from src.metrics.samples_io import save_samples_npz, split_rows


def _parse_args():
    p = argparse.ArgumentParser(description="Image folder → evaluation npz")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--suffix", default=".png", help='File name after the GT stem; may contain "{k}"')
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--data_path", default="Data/DamagePattern/id_list_val_R307.json")
    p.add_argument("--gt_dir", default="Data/DamagePattern/images", help="Original GT, stored in `real` for reference")
    p.add_argument("--clahe", action="store_true", help="Apply the training CLAHE to the generated images")
    p.add_argument("--size", type=int, default=128)
    return p.parse_args()


def _clahe(u8):
    import cv2

    op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return op.apply(u8)


def main():
    cli = _parse_args()
    if cli.k > 1 and "{k}" not in cli.suffix:
        raise SystemExit('--k > 1 needs "{k}" in --suffix')
    rows = split_rows(cli.split, cli.data_path)
    names = rows["filename"].tolist()
    fake = np.zeros((len(names), cli.k, cli.size, cli.size), np.uint8)
    missing = []
    for i, n in enumerate(names):
        stem = os.path.splitext(n)[0]
        for k in range(cli.k):
            path = os.path.join(cli.image_dir, stem + cli.suffix.format(k=k))
            if not os.path.isfile(path):
                missing.append(path)
                continue
            u8 = np.array(Image.open(path).convert("L").resize((cli.size, cli.size), Image.BICUBIC))
            fake[i, k] = _clahe(u8) if cli.clahe else u8
    if missing:
        raise SystemExit(f"{len(missing)} images missing, e.g. {missing[:3]}")
    real = load_real_uint8(names, cli.gt_dir, image_size=cli.size)
    meta = {"src": os.path.abspath(cli.image_dir), "suffix": cli.suffix, "k": cli.k, "split": cli.split,
            "preproc": "gray (PIL L) + bicubic" + (" + CLAHE" if cli.clahe else "")}
    save_samples_npz(cli.out, fake, real, np.zeros((len(names), 1)), rows["DI"].to_numpy(), names, meta)
    print(f"[images_to_npz] wrote {cli.out}  fake={fake.shape}")


if __name__ == "__main__":
    main()
