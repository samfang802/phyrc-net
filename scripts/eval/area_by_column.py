#!/usr/bin/env python3
"""Spalled-area check per column and drift level: generated vs GT, from one samples npz.

damage_area of src/metrics/damage_texture.py (thresholds calibrated on intact train frames, as in
eval_damage_texture.py), mean over the K samples of each condition, then mean over the frames of a
(column, drift level). Used to tell whether a model under-draws on its own training columns
(generator bias) or only on unseen ones (extrapolation).

  python scripts/eval/area_by_column.py --samples x.npz --out area.csv
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.metrics import damage_texture as dt
from src.metrics import dino_mmd as dm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data_path", default="Data/DamagePattern/id_list_val_R307.json")
    p.add_argument("--image_dir", default="Data/DamagePattern_inpaint_selfcondnoDI/images")
    p.add_argument("--intact_max_code", type=int, default=250)
    cli = p.parse_args()
    z = np.load(cli.samples)
    fn = [str(n) for n in z["filename"]]
    fake = z["fake"]
    G, K = fake.shape[:2]
    train = json.load(open(cli.data_path))["train"]
    codes = np.array([int(n.split("_")[2]) for n in train])
    thr = dt.calibrate(dm.load_real_uint8(train, cli.image_dir, use_clahe=False)[codes <= cli.intact_max_code])
    gt = dt.texture_stats(dm.load_real_uint8(fn, cli.image_dir, use_clahe=False), thr)["damage_area"]
    md = dt.texture_stats(fake.reshape(-1, *fake.shape[2:]), thr)["damage_area"].reshape(G, K).mean(1)
    lab = pd.read_csv("Data/DamagePattern/label.csv").set_index("filename")
    F = pd.DataFrame({"filename": fn, "gt": gt, "model": md})
    F["column"] = F.filename.str.split("_").str[0]
    F["level"] = F.filename.str.split("_").str[2].astype(int)
    F["DI"] = lab.loc[fn, "DI"].to_numpy()
    by = F.groupby(["column", "level"])[["gt", "model"]].mean()
    by["ratio"] = by["model"] / by["gt"].where(by["gt"] > 0.01)
    by.round(3).to_csv(cli.out)
    print(by.round(3).to_string())
    hi = F[F.DI >= F.DI.quantile(2 / 3)]
    col = hi.groupby("column")[["gt", "model"]].mean()
    col["ratio"] = col["model"] / col["gt"]
    print("\nhigh-DI third (of this split) per column\n" + col.round(3).to_string())
    print(f"\nall high-DI: gt {hi['gt'].mean():.3f}  model {hi['model'].mean():.3f}  ratio {hi['model'].mean() / hi['gt'].mean():.2f}")


if __name__ == "__main__":
    main()
