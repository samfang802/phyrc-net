#!/usr/bin/env python3
"""Layout npz from the real labels (controls for the harness; src/data/layout_io.py format).

  --source iso32 | iso16 | cov32   real coverage from onset_field.npz (iso = monotone-ized)
  --source mask128                 AnyLabel binary mask at 128 (the layout_scratch oracle; S=128)
Same layout repeated for K samples.   python scripts/eval/make_real_layouts.py --split test --source iso16 --k 4 --out x.npz
"""
import argparse
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
from src.data import layout_io  # noqa: E402
from src.metrics.samples_io import split_rows  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--split", required=True, choices=("test", "val"))
p.add_argument("--source", required=True)
p.add_argument("--k", type=int, default=4)
p.add_argument("--out", required=True)
cli = p.parse_args()
files = split_rows(cli.split)["filename"].tolist()
if cli.source == "mask128":
    idx, mask, _ = layout_io.anylabel_cache()
    cov = np.stack([mask[idx[f]] for f in files]).astype(np.float32)
else:
    cov = layout_io.onset_layout(files, cli.source)
lay = np.repeat(cov[:, None], cli.k, axis=1)
layout_io.save_layout_npz(cli.out, files, lay, f"real_{cli.source}", cli.split)
print(f"[real_layouts] {cli.source} {cli.split} {lay.shape} mean {lay.mean():.4f} -> {cli.out}")
