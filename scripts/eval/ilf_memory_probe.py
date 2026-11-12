#!/usr/bin/env python3
"""Does the learned ILF memory collapse to the last frame? (No training.)

Training memory for target t = max over the walk's earlier writing frames of E(frame) (loss_mixin
``_refiner_learned_memory_hint``). The gradient of that max reaches only the argmax frame per (channel, cell). Damage
grows monotonically, so the argmax may nearly always be frame t-1, which makes the memory a re-encoding of S_{t-1}
and starves the earlier frames of signal. Per checkpoint and split, over every (target t, channel, cell) of the
real frames (gate none / kaiser as trained):
  lag share     argmax = t-1 (lag 0), lag 1, 2-5, >= 6 frames back
  n_eff(r)      effective number of frames an unnormalised log-sum-exp pool with temperature r would send gradient
                to (1 / sum softmax(r e)^2), r = 5 / 20 / 50, mean over (t, channel, cell)
  redundancy    Pearson r between the memory field and E(t-1) alone (1 = the memory adds nothing to S)
  saturation    mean field, share of cells within 10% of the channel's max, spatial std / mean (0 = flat field)
  growth        mean field at the first / middle / last third of the walk

  python3 scripts/eval/ilf_memory_probe.py --run <result_dir> --step 1000 --step 4000 --split train --split val
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_spec = importlib.util.spec_from_file_location(
    "_s_probe", os.path.join(_ROOT, "scripts", "eval", "refiner_s_probe.py"))
_sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sp)

RS = (5.0, 20.0, 50.0)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--step", type=int, action="append", required=True)
    p.add_argument("--split", action="append", choices=("train", "val", "test"))
    p.add_argument("--out", required=True)
    return p.parse_args()


@torch.no_grad()
def _probe(cfm, ds, dl, device):
    from src.model.cfm.refiner_memory import memory_input, write_gates

    if getattr(cfm, "memory_encoder", None) is None:
        raise SystemExit("the model has no learned memory encoder")
    frames = []
    for batch in dl:
        real = cfm._unpack_generation_batch(batch)[0]
        frames.append(cfm._encode_to_target_space(real.to(device)))
    frames = torch.cat(frames)
    lab = ds.label
    gate_mode = str(getattr(cfm, "refiner_memory_gate", "none"))
    acc = {"lag0": 0.0, "lag1": 0.0, "lag2_5": 0.0, "lag6p": 0.0, "n": 0.0,
           **{f"neff_r{r:g}": 0.0 for r in RS}, "red_num": [], "mean": [], "near_max": [], "flat": []}
    thirds = {0: [], 1: [], 2: []}
    for _, rows in lab.groupby(["exp", "direction"], sort=True):
        rows = rows.sort_values("timestep", kind="stable")
        idx = rows.index.to_numpy()
        gates = np.asarray(write_gates(rows["filename"].tolist(), gate_mode), bool)
        x = frames[idx].float()
        e = cfm.memory_encoder(memory_input(x, x[:1], getattr(cfm.memory_encoder, "input_mode", "raw")))  # [T, C, h, w]
        T = len(idx)
        for t in range(1, T):
            w = np.flatnonzero(gates[:t])
            if len(w) == 0:
                continue
            ew = e[w]  # earlier writing frames
            field, arg = ew.max(dim=0)
            lag = (t - 1) - torch.as_tensor(w, device=device)[arg]  # 0 = frame t-1
            acc["lag0"] += float((lag == 0).float().sum())
            acc["lag1"] += float((lag == 1).float().sum())
            acc["lag2_5"] += float(((lag >= 2) & (lag <= 5)).float().sum())
            acc["lag6p"] += float((lag >= 6).float().sum())
            acc["n"] += lag.numel()
            for r in RS:
                sm = torch.softmax(r * ew, dim=0)
                acc[f"neff_r{r:g}"] += float((1.0 / (sm ** 2).sum(dim=0)).sum())
            last = e[t - 1].flatten()
            f = field.flatten()
            acc["red_num"].append(float(torch.corrcoef(torch.stack([f, last]))[0, 1]) if f.std() > 0 and last.std() > 0
                                  else float("nan"))
            fc = field.flatten(1)  # [C, cells]
            acc["mean"].append(float(fc.mean()))
            acc["near_max"].append(float((fc >= 0.9 * fc.max(dim=1, keepdim=True).values).float().mean()))
            acc["flat"].append(float((fc.std(dim=1) / fc.mean(dim=1).clamp_min(1e-6)).mean()))
            thirds[min(2, 3 * t // T)].append(float(fc.mean()))
    n = max(acc["n"], 1.0)
    return {
        "lag_share": {k: acc[k] / n for k in ("lag0", "lag1", "lag2_5", "lag6p")},
        "n_eff": {f"r{r:g}": acc[f"neff_r{r:g}"] / n for r in RS},
        "redundancy_r_with_E_prev": float(np.nanmean(acc["red_num"])),
        "field_mean": float(np.mean(acc["mean"])),
        "near_max_share": float(np.mean(acc["near_max"])),
        "spatial_cv": float(np.mean(acc["flat"])),
        "growth_thirds": [float(np.mean(v)) if v else float("nan") for v in thirds.values()],
        "e_stats": {"min": float(e.min()), "max": float(e.max())},
    }


def main():
    cli = _parse_args()
    device = torch.device("cuda")
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    report = {}
    for split in cli.split or ["train", "val"]:
        for step in cli.step:
            ckpt = os.path.join(cli.run, f"checkpoint_step_{step:06d}.pt")
            name, cfm, ds, dl, _lay, got = _sp._build(f"s{step}={cli.run}@{ckpt}", split, "mean", device)
            r = _probe(cfm, ds, dl, device)
            report[f"{split}/{step}"] = r
            ls, ne = r["lag_share"], r["n_eff"]
            print(f"[ilf_mem] {split} step {got}: argmax lag0 {ls['lag0']:.3f} lag1 {ls['lag1']:.3f} "
                  f"lag2-5 {ls['lag2_5']:.3f} lag6+ {ls['lag6p']:.3f} | n_eff r5 {ne['r5']:.2f} r20 {ne['r20']:.2f} "
                  f"r50 {ne['r50']:.2f} | corr(field, E(t-1)) {r['redundancy_r_with_E_prev']:.3f} | "
                  f"mean {r['field_mean']:.3f} near-max {r['near_max_share']:.3f} spatial cv {r['spatial_cv']:.3f} | "
                  f"thirds {[round(x, 3) for x in r['growth_thirds']]} | E range {r['e_stats']}", flush=True)
            del cfm
            torch.cuda.empty_cache()
            with open(cli.out, "w") as fh:
                json.dump(report, fh, indent=2)
    print(f"[ilf_mem] wrote {cli.out}")


if __name__ == "__main__":
    main()
