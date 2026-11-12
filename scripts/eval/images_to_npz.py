#!/usr/bin/env python3
"""Saved images of any model → the sample npz the evaluation scripts read (src/metrics/samples_io.py).

Use this for models that are not CFM runs of this repo (the cGAN baseline, another codebase, ...).
Each image must be named after its GT frame:

  <image_dir>/<GT stem><suffix>                 one sample per condition (K = 1)
  <image_dir>/<GT stem><suffix with {k}>        K samples, e.g. --suffix "_k{k}.png" --k 8

Images are converted to gray (PIL "L") and resized to 128 with bicubic, the same as the cGAN
baseline in docs/eval/results.md. Pass --clahe when the model draws raw photos instead of the
CLAHE'd domain of the reference set: the cGAN was trained on Data/DamagePattern/images and needs
it, the CFM runs of this repo do not. It is the generator's own CLAHE (dino_mmd.clahe_uint8).
Rows follow the split's label.csv order; a missing image is an error.

--from_npz instead takes an existing sample npz and only applies --clahe to its `fake`.

Examples:
  python scripts/eval/images_to_npz.py --image_dir cGANckpt/CTR1 --suffix _test.jpg --clahe --out cgan_test.npz
  python scripts/eval/images_to_npz.py --from_npz cGAN_eq.npz --clahe --out cGAN_eq_clahe.npz
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

from src.metrics.dino_mmd import clahe_uint8, load_real_uint8
from src.metrics.samples_io import save_samples_npz, split_rows


def _parse_args():
    p = argparse.ArgumentParser(description="Image folder → evaluation npz")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image_dir")
    src.add_argument("--from_npz", help="An existing sample npz (only --clahe is applied)")
    p.add_argument("--suffix", default=".png", help='File name after the GT stem; may contain "{k}"')
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--data_path", default="Data/DamagePattern/id_list_val_R307.json")
    p.add_argument("--gt_dir", default="Data/DamagePattern/images", help="Original GT, stored in `real` for reference")
    p.add_argument("--clahe", action="store_true", help="Apply the training CLAHE to the generated images")
    p.add_argument("--size", type=int, default=128)
    return p.parse_args()


def _from_npz(cli):
    import json

    z = np.load(cli.from_npz, allow_pickle=False)
    fake = z["fake"]
    if cli.clahe:
        fake = clahe_uint8(fake.reshape(-1, *fake.shape[2:]), cli.size).reshape(fake.shape)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    meta = {**meta, "src_npz": os.path.abspath(cli.from_npz), "preproc_added": "CLAHE (generator)" if cli.clahe else "none"}
    save_samples_npz(cli.out, fake, z["real"], z["cond"], z["DI"], z["filename"], meta)
    print(f"[images_to_npz] wrote {cli.out}  fake={fake.shape}")


def main():
    cli = _parse_args()
    if cli.from_npz:
        return _from_npz(cli)
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
            fake[i, k] = np.array(Image.open(path).convert("L").resize((cli.size, cli.size), Image.BICUBIC))
    if missing:
        raise SystemExit(f"{len(missing)} images missing, e.g. {missing[:3]}")
    if cli.clahe:
        fake = clahe_uint8(fake.reshape(-1, cli.size, cli.size), cli.size).reshape(fake.shape)
    real = load_real_uint8(names, cli.gt_dir, image_size=cli.size)
    meta = {"src": os.path.abspath(cli.image_dir), "suffix": cli.suffix, "k": cli.k, "split": cli.split,
            "preproc": "gray (PIL L) + bicubic" + (" + CLAHE (generator)" if cli.clahe else "")}
    save_samples_npz(cli.out, fake, real, np.zeros((len(names), 1)), rows["DI"].to_numpy(), names, meta)
    print(f"[images_to_npz] wrote {cli.out}  fake={fake.shape}")


if __name__ == "__main__":
    main()
