#!/usr/bin/env python3
"""Does the refiner read its memory or S_{t-1}, and how does that change with the state's noise? (No training.)

The training loss of frame t (refiner_implicit_flow_matching_loss, eval mode: no state noise, no memory drop, no
contrast aug) with the Resampling Forcing state pinned to one level t_s (S_{t-1} = the real frame t-1 noised to t_s
and integrated back by the model; ``tf`` = the real frame t-1, teacher forcing). Per level, the same batches, flow
times, start noises and resample draws are scored with
  real      the memory as trained (real frames before t)
  mem_zero  memory set to 0 (the "nothing written yet" state)
  mem_swap  memory of another sample of the shuffled batch (other time / face)
  S_swap    S_{t-1} of another sample, the memory kept
and the paired loss increase over ``real`` is reported overall and per flow-time third (tau < 1/3 is the noisy end).
Only rows whose S was resampled count at t_s < 1 (rows without a frame t-2 keep the real frame).
Reading: a curve of mem_zero / mem_swap that rises as t_s falls = the model leans on the memory when S is unreliable;
S_swap against mem_swap says which input carries more of frame t.

  python3 scripts/eval/memory_reliance_probe.py --out results/ilf_memory_probe/reliance.json \
      --model rfm=<result_dir>@<checkpoint> --model ILF_bd=<result_dir>@<checkpoint>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_spec = importlib.util.spec_from_file_location(
    "_s_probe", os.path.join(_ROOT, "scripts", "eval", "refiner_s_probe.py"))
_sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sp)


LEVELS = ("tf", 0.9, 0.7, 0.5, 0.3, 0.1)
VARIANTS = ("real", "mem_zero", "mem_swap", "S_swap")
TBINS = ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0))


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", action="append", required=True, help="name=result_dir@checkpoint; repeatable")
    p.add_argument("--split", action="append", choices=("val", "test"))
    p.add_argument("--repeats", type=int, default=6, help="passes over the split (new shuffle / noise each)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _attach_memory(cfm, ds, args):
    """The split's own memory, wired as train_mixed does (val-only table, offset 0)."""
    from src.model.cfm.refiner_memory import build_memory_cache, build_memory_frames

    transform = ds.transform  # the dataset caches prefixed (resized) frames: its own transform, as train_mixed
    gate = str(getattr(args, "refiner_memory_gate", "none"))
    if bool(getattr(args, "refiner_memory_encoder", False)):
        frames, rows_of = build_memory_frames(ds, transform, 0, gate, str(getattr(args, "refiner_memory_input", "raw")))
        ds.attach_memory_rows(rows_of)
        cfm.attach_memory_frames(frames)
    else:
        ds.attach_memory_cache(build_memory_cache(
            ds, transform, float(args.refiner_memory_sigma), float(args.refiner_memory_tau),
            int(args.refiner_memory_window), gate, int(getattr(args, "refiner_memory_ttl", 0)),
            float(getattr(args, "refiner_memory_novelty", 0.0))))


class _Hooks:
    """Instance-level patches: the variant, and capture of per-sample loss / flow time / resampled rows."""

    def __init__(self, cfm):
        self.cfm, self.variant, self.rec = cfm, "real", {}
        learned = bool(cfm.refiner_memory_encoder)
        hint_name = "_refiner_learned_memory_hint" if learned else "_refiner_memory_hint"
        orig_hint = getattr(cfm, hint_name)
        orig_track = cfm._refiner_state_track
        orig_pred = cfm._predict_velocity
        orig_mse = cfm._cfm_mse

        def hint(ar_pack, b, device, dtype, *a, **k):
            mem = orig_hint(ar_pack, b, device, dtype, *a, **k)
            rows_key = a[0] if a else k.get("rows_key", "memory_rows")
            if learned and rows_key != "memory_rows":
                return mem  # the resample hop's memory (before t-1) stays real: S is the same in every variant
            if self.variant == "mem_zero":
                return torch.zeros_like(mem)
            if self.variant == "mem_swap":
                return torch.roll(mem, 1, dims=0)
            return mem

        def track(*a, **k):
            has_prev, s_in = orig_track(*a, **k)
            self.rec["has_prev"] = has_prev.detach()
            self.rec["conf"] = cfm._refiner_state_conf_rows.detach().clone()
            if self.variant == "S_swap":
                s_in = torch.roll(s_in, 1, dims=0)
            return has_prev, s_in

        def pred(zt, t, *a, **k):
            self.rec["t"] = t.detach()
            return orig_pred(zt, t, *a, **k)

        def mse(*a, **k):
            out = orig_mse(*a, **k)
            self.rec["loss"] = out.detach()
            return out

        setattr(cfm, hint_name, hint)
        cfm._refiner_state_track = track
        cfm._predict_velocity = pred
        cfm._cfm_mse = mse


def _set_level(cfm, level):
    if level == "tf":
        cfm.refiner_resample_warmup = 10 ** 9  # _refiner_resampled_prev returns the real frame t-1
        return
    cfm.refiner_resample_warmup, cfm.refiner_resample_ramp = 0, 0
    cfm.refiner_resample_mu, cfm.refiner_resample_sigma = math.log(level / (1.0 - level)), 0.0


@torch.no_grad()
def _probe(name, cfm, ds, cli, device):
    hk = _Hooks(cfm)
    # rows -> per (level, variant): per-sample losses and flow times, paired across variants
    res = {}
    for level in LEVELS:
        _set_level(cfm, level)
        acc = {v: [] for v in VARIANTS}
        tt = []
        for rep in range(cli.repeats):
            g = torch.Generator().manual_seed(cli.seed + 1000 * rep)
            dl = DataLoader(ds, batch_size=cli.batch, shuffle=True, generator=g, num_workers=1, drop_last=True)
            for bi, batch in enumerate(dl):
                real, hs, dr, exp, ts, di, info, valid_mask, cond_mask, ar_pack = cfm._unpack_generation_batch(
                    _to(batch, device))
                c, gru = cfm._unpack_conditions(info, hs, dr, di, exp=exp)
                z = cfm._encode_to_target_space(real)
                cond_mask = cfm._align_spatial_to_target(cond_mask)
                valid_mask = cfm._align_spatial_to_target(valid_mask)
                keep = None
                for v in VARIANTS:
                    hk.variant = v
                    s = cli.seed + 7919 * rep + bi
                    torch.manual_seed(s)
                    torch.cuda.manual_seed_all(s)
                    cfm._flow_matching_loss_for_batch(z, c, gru, valid_mask, cond_mask, ar_pack, info=info)
                    if keep is None:
                        keep = hk.rec["has_prev"].bool()
                        if level != "tf":
                            keep = keep & (hk.rec["conf"] < 1.0)
                        tt.append(hk.rec["t"][keep].float().cpu())
                    acc[v].append(hk.rec["loss"][keep].float().cpu())
        L = {v: torch.cat(acc[v]).numpy() for v in VARIANTS}
        T = torch.cat(tt).numpy()
        row = {"n": int(len(T)), "loss_real": float(L["real"].mean())}
        for v in VARIANTS[1:]:
            d = L[v] - L["real"]
            row[v] = {"dloss": float(d.mean()), "se": float(d.std() / max(math.sqrt(len(d)), 1.0)),
                      "rel": float(d.mean() / max(L["real"].mean(), 1e-8)),
                      "by_tau": [float(d[(T >= lo) & (T < hi)].mean()) if ((T >= lo) & (T < hi)).any() else None
                                 for lo, hi in TBINS]}
        res[str(level)] = row
        print(f"[reliance] {name} t_s={level}: n {row['n']} loss {row['loss_real']:.4f} | "
              + " | ".join(f"{v} +{row[v]['dloss']:.4f}±{row[v]['se']:.4f} ({100 * row[v]['rel']:+.1f}%) "
                           f"τ-thirds {[None if x is None else round(x, 4) for x in row[v]['by_tau']]}"
                           for v in VARIANTS[1:]), flush=True)
    return res


def _to(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to(v, device) for v in x)
    return x


def main():
    cli = _parse_args()
    device = torch.device("cuda")
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    report = {}
    for split in cli.split or ["val"]:
        for spec in cli.model:
            name, cfm, ds, _dl, _lay, step = _sp._build(spec, split, "mean", device)
            if not getattr(cfm, "refiner_memory", False):
                raise SystemExit(f"{name}: no refiner memory")
            args = _sp._load_args_from_result_dir(spec.split("=", 1)[1].split("@")[0])
            _attach_memory(cfm, ds, args)
            print(f"[reliance] {name} step {step} {split}: {len(ds)} items, "
                  f"{'learned E' if cfm.refiner_memory_encoder else 'rule memory'}", flush=True)
            report[f"{split}/{name}"] = {"step": step, **_probe(name, cfm, ds, cli, device)}
            del cfm
            torch.cuda.empty_cache()
            with open(cli.out, "w") as fh:
                json.dump(report, fh, indent=2)
    print(f"[reliance] wrote {cli.out}")


if __name__ == "__main__":
    main()
