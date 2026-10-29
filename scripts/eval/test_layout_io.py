#!/usr/bin/env python3
"""Unit checks for src/data/layout_io.py.  python scripts/eval/test_layout_io.py  (also pytest-able)."""
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from src.data import layout_io as L  # noqa: E402


def _iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


def test_identity_and_edges():
    names, mask, sdf = L.anylabel_cache()
    # a 128 "coverage" that is already binary reproduces the cached SDF (fp16 cache)
    idx = np.linspace(0, len(mask) - 1, 40).astype(int)
    got = L.coverage_to_sdf128(mask[idx].astype(np.float32))
    assert np.abs(got - sdf[idx]).max() < 2e-3, np.abs(got - sdf[idx]).max()
    assert (L.coverage_to_sdf128(np.zeros((32, 32))) == -1).all()
    assert (L.coverage_to_sdf128(np.ones((32, 32))) == 1).all()
    assert L.coverage_to_sdf128(np.zeros((2, 3, 32, 32))).shape == (2, 3, 128, 128)


def test_downsample_roundtrip(verbose=True):
    """anylabel full-res mask → 32×32 area coverage (as onset_field cov32) → SDF, vs the 128 SDF."""
    import json
    import glob
    ex = L._exporter()
    names, mask, sdf = L.anylabel_cache()
    files = sorted(glob.glob(os.path.join(_ROOT, "Data/DamagePattern/anylabel/*.json")))
    rng = np.random.RandomState(0)
    ious, errs, da = [], [], []
    for fn in rng.permutation(files)[:120]:
        data = json.load(open(fn, encoding="utf-8"))
        full = ex.spall_mask_full(data) > 0
        if not full.any():
            continue
        stem = os.path.splitext(os.path.basename(data.get("imagePath") or fn))[0] + ".jpg"
        if stem not in names:
            continue
        i = names[stem]
        cov32 = cv2.resize(full.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA)
        m = L.coverage_to_mask128(cov32)
        s = L.coverage_to_sdf128(cov32)
        ious.append(_iou(m, mask[i]))
        errs.append(np.abs(s - sdf[i].astype(np.float32)).mean())
        da.append(m.mean() - mask[i].mean())
    ious, errs, da = map(np.array, (ious, errs, da))
    if verbose:
        print(f"[roundtrip] n={len(ious)} IoU mean {ious.mean():.3f} (p10 {np.quantile(ious, .1):.3f}) "
              f"SDF mean|err| {errs.mean():.3f} area bias {da.mean():+.4f} (|max| {np.abs(da).max():.3f})")
    assert ious.mean() > 0.80 and errs.mean() < 0.15 and abs(da.mean()) < 0.01


if __name__ == "__main__":
    test_identity_and_edges()
    test_downsample_roundtrip()
    print("ok")
