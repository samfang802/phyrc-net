#!/usr/bin/env python3
"""Condition cache of the image-free columns for OPA-Flow group C (docs/redesign/phase1_opaflow.md).

For every specimen with a simulated loop except the six imaged ones, at every loop row t that is a
multiple of 30 (a load peak, the grid the image frames sit on):
  g_t        GRU hidden state of the gn extractor (phase1_extractor), the generator's condition
  max_drift  largest |drift| (%) up to t, from the simulated loop
  P          damage progress of the damage-state encoder, (-strength + energy) / 2 in its standardized
             units: the same quantity the critic's magnitude head was trained on (phase0_critic.py)
and the standardized 17-column specimen info vector.

npz keys: exps (E,), info (E,17) f32, row_exp (R,) int, row_ts (R,) int, row_maxdrift (R,) f32,
row_P (R,) f32, g (R,1024) f16. Read once into memory by the trainer (no per-item np.load).

  python scripts/export/export_opa_free_cache.py --out Data/DamagePattern/opa_free_cache.npz
"""
import argparse
import glob
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.getcwd())
from src.data.damage_condition import build_frame_conditions  # noqa: E402
from src.data.hysteresis import HystereticDataset  # noqa: E402

IMAGED = {"R315", "R330", "R615", "R1015", "R307", "CTR1"}
ENCODER_RUN = "results/DamageStateEncoder/A_info/2026-10-23_22-34-47"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="Data/DamagePattern/opa_free_cache.npz")
    p.add_argument("--encoder_run", default=ENCODER_RUN)
    p.add_argument("--extractor", default="phase1_extractor")
    p.add_argument("--step", type=int, default=30)
    cli = p.parse_args()
    torch.set_num_threads(2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    a = argparse.Namespace(**json.load(open(os.path.join(cli.extractor, "args.json"))))
    ext = getattr(importlib.import_module("src.model.HystereticPrediction"), a.architecture)(**vars(a))
    ck = glob.glob(os.path.join(cli.extractor, "**/best.ckpt"), recursive=True)[0]
    ext.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"])
    ext.eval().to(dev)

    info_all = pd.read_csv("Data/Hysteresis/info_w_mander_big.csv", dtype={"id": str})
    ids = [i for i in info_all.id if i not in IMAGED
           and os.path.exists(f"Data/Hysteresis/Simulate/{i}.csv") and os.path.exists(f"Data/Hysteresis/Experiment/{i}.csv")]
    print(f"[free-cache] {len(ids)} image-free specimens", flush=True)
    ds = HystereticDataset(root="Data/Hysteresis", specified=ids, **vars(a))
    cond = build_frame_conditions(cli.encoder_run, ids, root="Data/Hysteresis")

    info, row_exp, row_ts, row_md, row_P, G = [], [], [], [], [], []
    with torch.no_grad():
        for k, e in enumerate(ds.exp_list):
            inp, _tf, _L, _mx, _ = ds[k]
            out, _ = ext.L(inp[None].to(dev))
            g = out[0].cpu().numpy()
            sim = pd.read_csv(f"Data/Hysteresis/Simulate/{e}.csv")["Drift(%)"].to_numpy()
            c = cond[e].numpy()
            n = min(len(g), len(sim), len(c))
            ts = np.arange(cli.step, n, cli.step)
            md = np.maximum.accumulate(np.abs(sim))[ts]
            P = 0.5 * (-c[ts, -4] + c[ts, -2])  # strength_ratio, stiffness, cum_energy, log1p_max_drift
            keep = md > 0.2
            ts, md, P = ts[keep], md[keep], P[keep]
            info.append(ds.info[ds.info["id"] == e].iloc[:, 1:].values.squeeze().astype(np.float32))
            row_exp += [len(info) - 1] * len(ts)
            row_ts.append(ts)
            row_md.append(md)
            row_P.append(P)
            G.append(g[ts].astype(np.float16))
            if k % 20 == 0:
                print(f"[free-cache] {k + 1}/{len(ids)} {e} rows {len(ts)}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    np.savez_compressed(
        cli.out, exps=np.array(list(ds.exp_list)), info=np.stack(info), row_exp=np.array(row_exp, np.int64),
        row_ts=np.concatenate(row_ts).astype(np.int64), row_maxdrift=np.concatenate(row_md).astype(np.float32),
        row_P=np.concatenate(row_P).astype(np.float32), g=np.concatenate(G),
    )
    print(f"[free-cache] wrote {cli.out}: {len(row_exp)} rows", flush=True)


if __name__ == "__main__":
    main()
