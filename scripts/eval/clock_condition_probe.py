#!/usr/bin/env python3
"""Does the KF damage clock tau add anything to the v11 condition? Probe check before training a generator.

Representations per labelled frame of the six imaged specimens (docs/redesign/kf_clock.md):
  v11 scalars     the 4 damage scalars of the v11 condition (src/data/damage_condition.py)
  v11 [z | s]     the full 260-d v11 condition
  ... + tau_sim   with the KF clock from the simulated force (what the generator inputs are built from)
  ... + tau_exp   with the KF clock from the experimental force (information no condition uses yet)
  tau_* alone     the 1-D clocks (reference rows of kf_clock.md)
  GRU g           reference
tau is read from results/kf_clock/frames.csv (KF fitted on the non-imaged specimens only, no image labels).

Probes, both leave-one-imaged-specimen-out, trained on R315 / R330 / R615 / R1015 (R307 / CTR1 from all four):
  ridge   RidgeCV, as scripts/eval/kf_damage_clock.py
  mlp     2 x 64 SiLU MLP, AdamW + weight decay, full batch, mean of 5 seeds (GPU). The generator is
          nonlinear, so this asks whether a nonlinear reader of the v11 scalars already recovers tau.
Error = mean prediction - mean label per (specimen, drift level); LOSO RMSE over the train specimens' levels.

  python scripts/eval/clock_condition_probe.py [--out results/clock_condition_probe]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.getcwd())
from src.data.damage_condition import build_frame_conditions
from src.data.damage_state import IMAGE_SPECIMENS
from src.metrics import spall_labels

TR = ["R315", "R330", "R615", "R1015"]
CELLS = (("CTR1", 6000), ("CTR1", 8000), ("R307", 6000), ("R307", 8000))


def ridge(X, F, trm, device):
    out = np.full(len(F), np.nan)
    for e in IMAGE_SPECIMENS:
        tr = trm & (F.exp.values != e)
        te = F.exp.values == e
        out[te] = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(X[tr], F.R_spall.values[tr]).predict(X[te])
    return out


def mlp(X, F, trm, device, seeds=5, steps=3000):
    out = np.zeros(len(F))
    Xt = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(F.R_spall.values).float().to(device)
    for e in IMAGE_SPECIMENS:
        tr = torch.from_numpy(trm & (F.exp.values != e)).to(device)
        te = F.exp.values == e
        preds = []
        for s in range(seeds):
            torch.manual_seed(s)
            net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.SiLU(),
                                      torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1)).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
            for _ in range(steps):
                opt.zero_grad()
                loss = ((net(Xt[tr])[:, 0] - y[tr]) ** 2).mean()
                loss.backward()
                opt.step()
            with torch.no_grad():
                preds.append(net(Xt[torch.from_numpy(te).to(device)])[:, 0].cpu().numpy())
        out[te] = np.mean(preds, axis=0)
    return out


def score(F, hat, name):
    g = F.assign(hat=hat).groupby(["exp", "level"])[["R_spall", "hat"]].mean()
    err = g.hat - g.R_spall
    loso = g[g.index.get_level_values(0).isin(TR)]
    r = {"rep": name}
    for e, lv in CELLS:
        r[f"{e} {lv // 1000}%"] = float(err.get((e, lv), np.nan))
    r["LOSO RMSE"] = float(np.sqrt(((loso.hat - loso.R_spall) ** 2).mean()))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="results/DamageStateEncoder/A_info/2026-10-23_22-34-47")
    ap.add_argument("--clock", default="results/kf_clock/frames.csv")
    ap.add_argument("--out", default="results/clock_condition_probe")
    cli = ap.parse_args()
    os.makedirs(cli.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)

    lab = pd.read_csv("Data/DamagePattern/label.csv")
    F = lab[lab.exp.isin(IMAGE_SPECIMENS)].reset_index(drop=True)
    F["level"] = F.filename.str.split("_").str[2].astype(int)
    F["R_spall"] = F.filename.map(spall_labels.load())
    F = F[F.R_spall.notna()].reset_index(drop=True)
    clock = pd.read_csv(cli.clock).set_index("filename")
    F["tau_sim"] = F.filename.map(clock.tau_sim)
    F["tau_exp"] = F.filename.map(clock.tau_exp)
    assert F[["tau_sim", "tau_exp"]].notna().all().all(), "frames missing from the clock table"

    cond = build_frame_conditions(cli.encoder, IMAGE_SPECIMENS)
    V = np.stack([cond[r.exp][int(r.timestep)].numpy() for r in F.itertuples()])  # [N, 260]
    gz = np.load("Data/DamagePattern/derived/gru_g.npz")
    gi = {f: i for i, f in enumerate(gz["filename"])}
    ts, te_ = F[["tau_sim"]].values, F[["tau_exp"]].values
    reps = {
        "GRU g": gz["g"][[gi[f] for f in F.filename]],
        "tau_sim": ts,
        "tau_exp": te_,
        "v11 scalars": V[:, -4:],
        "v11 scalars + tau_sim": np.c_[V[:, -4:], ts],
        "v11 scalars + tau_exp": np.c_[V[:, -4:], te_],
        "v11 [z | s]": V,
        "v11 [z | s] + tau_sim": np.c_[V, ts],
        "v11 [z | s] + tau_exp": np.c_[V, te_],
    }
    trm = F.exp.isin(TR).values
    rows = []
    for name, X in reps.items():
        X = X.astype(np.float64)
        X = (X - X[trm].mean(0)) / (X[trm].std(0) + 1e-6)
        for pname, fn in (("ridge", ridge), ("mlp", mlp)):
            r = score(F, fn(X, F, trm, device), name)
            r["probe"] = pname
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    tab = pd.DataFrame(rows).set_index(["probe", "rep"]).round(3)
    txt = ("R_spall (anylabel) probe error = predicted - actual (mean over frames of the level); "
           f"n_frames {len(F)}\n\n{tab.to_string()}\n")
    print("\n" + txt)
    open(os.path.join(cli.out, "report.txt"), "w").write(txt)
    json.dump(rows, open(os.path.join(cli.out, "probe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
