#!/usr/bin/env python3
"""Inception FID (K = 8) for eval-suite npz files, in the same image domain as the DINOv3 KID.

Reference = --image_dir images for the npz rows (the inpainted reference set, no CLAHE), exactly
as eval_dinov3_kid.py reads it; fakes get the same --jpeg round trip. Grayscale 128 px images are
repeated to 3 channels; torchmetrics resizes to 299 for Inception-v3 (pool3, 2048-d).

Per model:
  fid_pooled   all N·K fakes vs the N reals
  fid_per_k    mean ± std over the K draws (N fakes vs N reals each; comparable to a K = 1 model)
Reference row (--other_npz): the real images of another column vs this reference, a "real but
different column" scale. With only ~124 reals FID is strongly biased upward; compare models
only within one run.

  python scripts/eval/eval_fid.py --split test --samples V10=a.npz V11=b.npz --jpeg V10=75 V11=75 \
      --other_npz results/eval_suite/v106/val/samples/refiner.npz --out results/eval_suite/fid/test.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics.dino_mmd import load_real_uint8  # noqa: E402

_spec = importlib.util.spec_from_file_location("_kid", os.path.join(_ROOT, "scripts", "eval", "eval_dinov3_kid.py"))
_kid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kid)


def _to_t(u8):
    x = torch.from_numpy(u8.reshape(-1, *u8.shape[-2:])).float().div(255.0)
    return x[:, None].repeat(1, 3, 1, 1)


@torch.no_grad()
def _fid(fid, real_t, fake_t, device, bs=64):
    fid.reset()  # keeps real features (reset_real_features=False) after the first call
    if not getattr(fid, "_real_done", False):
        for i in range(0, len(real_t), bs):
            fid.update(real_t[i:i + bs].to(device), real=True)
        fid._real_done = True
    for i in range(0, len(fake_t), bs):
        fid.update(fake_t[i:i + bs].to(device), real=False)
    return float(fid.compute())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--samples", nargs="+", required=True, metavar="NAME=NPZ")
    p.add_argument("--jpeg", nargs="*", default=[], metavar="NAME=QUALITY")
    p.add_argument("--image_dir", default="Data/DamagePattern_inpaint_selfcondnoDI/images")
    p.add_argument("--other_npz", default="", help="npz whose `filename` rows are another real column")
    p.add_argument("--out", required=True)
    cli = p.parse_args()

    from torchmetrics.image.fid import FrechetInceptionDistance

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = {n: _kid._load_npz(path) for n, path in (s.split("=", 1) for s in cli.samples)}
    jpeg_q = {n: int(q) for n, q in (j.split("=", 1) for j in cli.jpeg)}
    names = list(runs)
    ref_files = list(runs[names[0]]["filename"])
    for n in names:
        assert list(runs[n]["filename"]) == ref_files, f"{n}: rows differ from {names[0]}"
    real_u8 = load_real_uint8(ref_files, cli.image_dir, 128, use_clahe=False)
    real_t = _to_t(real_u8)
    print(f"[fid] {cli.split}: reference n={len(real_u8)} from {cli.image_dir}", flush=True)

    fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True).to(device)
    report = {"split": cli.split, "n_real": len(real_u8), "image_dir": cli.image_dir, "jpeg": jpeg_q,
              "created": datetime.now(timezone.utc).isoformat(), "models": {}, "reference_rows": {}}
    for n in names:
        fake = runs[n]["fake"]
        if n in jpeg_q:
            fake = _kid._jpeg_u8(fake, jpeg_q[n])
        K = fake.shape[1]
        pooled = _fid(fid, real_t, _to_t(fake), device)
        per_k = [_fid(fid, real_t, _to_t(fake[:, k]), device) for k in range(K)]
        report["models"][n] = {"K": K, "fid_pooled": pooled,
                               "fid_per_k_mean": float(np.mean(per_k)), "fid_per_k_std": float(np.std(per_k)),
                               "fid_per_k": per_k}
        print(f"[fid] {n}: K={K} pooled {pooled:.2f}  per-k {np.mean(per_k):.2f} ± {np.std(per_k):.2f}", flush=True)

    if cli.other_npz:
        other_files = list(np.load(cli.other_npz)["filename"])
        other_u8 = load_real_uint8(other_files, cli.image_dir, 128, use_clahe=False)
        v = _fid(fid, real_t, _to_t(other_u8), device)
        report["reference_rows"]["other_real"] = {"n": len(other_u8), "fid": v, "npz": cli.other_npz}
        print(f"[fid] other real column (n={len(other_u8)}): {v:.2f}", flush=True)

    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    with open(cli.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[fid] wrote {cli.out}")


if __name__ == "__main__":
    main()
