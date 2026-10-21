"""The sample file every evaluation script reads (one row per condition of a split).

  fake      uint8 (N, K, H, W)   generated images, gray, 128×128
  real      uint8 (N, H, W)      the condition's GT through the same quantization (informational:
                                 the evaluation scripts read the reference from --image_dir)
  cond      float32 (N, D)       condition vector, if any (zeros are fine)
  DI        float32 (N,)
  filename  str (N,)             GT frame names, in the split's label.csv order
  meta      JSON string          provenance (checkpoint, sampler, preprocessing ...)

Rows must follow Data/DamagePattern/label.csv filtered to the split (`split_rows`), which is the
order PatternLoopDataset uses; all npz of one evaluation must share it.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def save_samples_npz(path, fake, real, cond, DI, filename, meta=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        fake=np.asarray(fake, dtype=np.uint8),
        real=np.asarray(real, dtype=np.uint8),
        cond=np.asarray(cond, dtype=np.float32),
        DI=np.asarray(DI, dtype=np.float32),
        filename=np.asarray(filename),
        meta=np.asarray(json.dumps(meta or {})),
    )


def split_rows(split, data_path="Data/DamagePattern/id_list_val_R307.json", label_csv="Data/DamagePattern/label.csv"):
    """label.csv rows of one split, in file order (= the npz row order)."""
    with open(data_path) as f:
        names = set(json.load(f)[split])
    label = pd.read_csv(label_csv)
    return label[label["filename"].isin(names)].reset_index(drop=True)
