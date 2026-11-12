#!/usr/bin/env python3
"""Can the 1/4-resolution learned memory hold thin cracks? E + decoder D alone, no refiner (a few hundred steps).

Per walk of real frames: inputs x_t = memory_input(I_t, I_0, "hpnew") (optionally rescaled), memory after frame t
m_t = max_{s<=t} E(x_s), target y_t = 1[max_{s<=t} (-x_s) > tau] (accumulated new thin dark structure, the binary
version of the refiner's decoder target). D(m_t) is scored on the val walks against y_t:
  F1 / IoU at 0.5, F1 with 2 px tolerance, positive rate of y and of the prediction.
Variants (name: channels, loss, input):
  c4_l1_raw    4, normalised L1 on the continuous target (the ILF_hpnew_aux recipe)
  c4_bd_s      4, BCE (pos_weight) + Dice, input / 0.05
  c8_bd_s      8, BCE + Dice, input / 0.05
  c8_bd_lcn    8, BCE + Dice, hpnew on locally contrast-normalised frames (fog-invariant)

  python3 scripts/eval/ilf_ae_probe.py --steps 600 --out results/ilf_memory_probe/ae_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics.dino_mmd import load_real_uint8
from src.model.cfm.refiner_memory import MemoryEncoder, _gauss_1d, _highpass, memory_input

TAU = 0.03
VARIANTS = {
    "c4_l1_raw": dict(ch=4, loss="l1", inp="raw"),
    "c4_bd_s": dict(ch=4, loss="bd", inp="scaled"),
    "c8_bd_s": dict(ch=8, loss="bd", inp="scaled"),
    "c8_bd_lcn": dict(ch=8, loss="bd", inp="lcn"),
}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--walks_per_step", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--variants", nargs="*", default=list(VARIANTS))
    p.add_argument("--data_path", default="Data/DamagePattern/id_list_val_R307.json")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _blur(x, sigma):
    k, r = _gauss_1d(sigma, x.device, x.dtype)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))


def _lcn(g, sigma=8.0, eps=0.02):
    """Local contrast normalisation: (g - local mean) / local std, so a fogged frame keeps its crack depth."""
    mu = _blur(g, sigma)
    sd = _blur((g - mu) ** 2, sigma).clamp_min(0).sqrt()
    return (g - mu) / (sd + eps)


def _walks(split, ids, lab, device):
    names = set(ids[split])
    out = []
    for _, rows in lab[lab.filename.isin(names)].groupby(["exp", "direction"], sort=True):
        rows = rows.sort_values("timestep", kind="stable")
        u8 = load_real_uint8(rows["filename"].tolist(), "Data/DamagePattern/images")
        g = torch.from_numpy(u8).float().div(255).unsqueeze(1).to(device)
        hpn = memory_input(g, g[:1], "hpnew")  # [T, 1, H, W] <= 0
        y = ((-hpn - TAU) > 0).float().cummax(dim=0).values  # accumulated binary target
        yc = (-hpn - TAU).clamp(min=0).cummax(dim=0).values  # continuous target (L1 recipe)
        z = _lcn(g)
        f, f0 = _highpass(z).clamp(max=0), _highpass(z[:1]).clamp(max=0).expand_as(z)
        lcn = (f + F.max_pool2d(-f0, 5, stride=1, padding=2)).clamp(max=0)
        out.append(dict(raw=hpn, scaled=hpn / 0.05, lcn=lcn, y=y, yc=yc))
    return out


def _forward(enc, x):
    e = enc(x)  # [T, C, h, w]
    m = e.cummax(dim=0).values
    return enc.decoder(m)  # logits / pre-softplus [T, 1, H, W]


def _dice(p, y, eps=1.0):
    return 1 - (2 * (p * y).sum() + eps) / (p.sum() + y.sum() + eps)


def _score(enc, walks, key, loss):
    tp = fp = fn = tp_tol = fp_tol = fn_tol = pos = pred_pos = n = 0.0
    with torch.no_grad():
        for w in walks:
            out = _forward(enc, w[key])
            prob = F.softplus(out) if loss == "l1" else torch.sigmoid(out)
            p = (prob > 0.5).float() if loss == "bd" else (prob > 0.02).float()  # l1: continuous depth
            y = w["y"]
            tp += float((p * y).sum()); fp += float((p * (1 - y)).sum()); fn += float(((1 - p) * y).sum())
            yd = F.max_pool2d(y, 5, 1, 2); pd_ = F.max_pool2d(p, 5, 1, 2)
            tp_tol += float((p * yd).sum()); fp_tol += float((p * (1 - yd)).sum()); fn_tol += float((y * (1 - pd_)).sum())
            pos += float(y.sum()); pred_pos += float(p.sum()); n += y.numel()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"f1": f1, "iou": tp / max(tp + fp + fn, 1),
            "f1_tol2px": 2 * tp_tol / max(2 * tp_tol + fp_tol + fn_tol, 1),
            "pos_rate": pos / n, "pred_rate": pred_pos / n}


def main():
    cli = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = json.load(open(cli.data_path))
    lab = pd.read_csv("Data/DamagePattern/label.csv")
    t0 = time.time()
    train, val = _walks("train", ids, lab, device), _walks("val", ids, lab, device)
    print(f"[ae] {len(train)} train / {len(val)} val walks loaded in {time.time() - t0:.0f}s on {device}", flush=True)
    pos = float(np.mean([float(w["y"].mean()) for w in train]))
    report = {"pos_rate_train": pos}
    for name in cli.variants:
        v = VARIANTS[name]
        torch.manual_seed(0)
        enc = MemoryEncoder(v["ch"], input_mode="hpnew", aux=True).to(device)
        opt = torch.optim.Adam(enc.parameters(), lr=cli.lr)
        g = torch.Generator().manual_seed(0)
        hist = []
        for step in range(1, cli.steps + 1):
            loss_sum = 0.0
            opt.zero_grad()
            for wi in torch.randint(len(train), (cli.walks_per_step,), generator=g).tolist():
                w = train[wi]
                out = _forward(enc, w[v["inp"]])
                if v["loss"] == "l1":
                    yc = w["yc"]
                    loss = (F.softplus(out) - yc).abs().sum() / yc.sum().clamp_min(1.0)
                else:
                    y = w["y"]
                    pw = torch.tensor((1 - pos) / max(pos, 1e-4), device=device).clamp(max=50)
                    loss = F.binary_cross_entropy_with_logits(out, y, pos_weight=pw) + _dice(torch.sigmoid(out), y)
                (loss / cli.walks_per_step).backward()
                loss_sum += float(loss) / cli.walks_per_step
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
            if step % 100 == 0 or step == cli.steps:
                s = _score(enc, val, v["inp"], v["loss"])
                hist.append({"step": step, "loss": loss_sum, **s})
                print(f"[ae] {name} step {step}: loss {loss_sum:.4f} | val F1 {s['f1']:.3f} IoU {s['iou']:.3f} "
                      f"F1@2px {s['f1_tol2px']:.3f} | pos {s['pos_rate']:.4f} pred {s['pred_rate']:.4f}", flush=True)
        report[name] = {"cfg": v, "hist": hist}
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as fh:
            json.dump(report, fh, indent=2)
    print(f"[ae] wrote {cli.out}")


if __name__ == "__main__":
    main()
