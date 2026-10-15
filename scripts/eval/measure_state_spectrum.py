#!/usr/bin/env python3
"""Radial power spectrum of the single-frame model's samples at different NFE, and how much of the
residual r = x_t - S is high-frequency.

For each (previous, target) pair of a split the frozen single-frame model samples the *previous*
frame with the same noise at several solver settings. That sample is the state S that the refiner's
state track sees (one Euler step in most training batches). Reported per setting:

  band power of S vs the real previous frame   low / mid / high (radial frequency in cycles per image)
  retention                                    E_high(S) / E_high(GT_{t-1})
  residual r = GT_t - S                        E_total(r), E_high(r) / E_total(r), E_high(r) / E_high(GT_t)
  same after the best integer shift of S       tests how much of r is grid misalignment

Reference rows: S = real previous frame (the true inter-frame change), and the shift search alone.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_required_match

# (name, solver, steps); NFE = steps for euler, 2 * steps for heun.
SETTINGS = (
    ("1 NFE (Euler 1)", "euler", 1),
    ("10 NFE (Heun 5)", "heun", 5),
    ("30 NFE (Heun 15)", "heun", 15),
    ("60 NFE (Heun 30, default)", "heun", 30),
)
BANDS = {"low": (0, 8), "mid": (8, 24), "high": (24, 1e9)}  # radial frequency, Nyquist = 64 at 128 px
BANDS_FINE = {"very_low(<4)": (0, 4), "low(4-8)": (4, 8), "mid(8-24)": (8, 24), "high(>=24)": (24, 1e9)}
SHIFT = 2


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="")
    return p.parse_args()


def _gray(x):
    return x[:, 0:1].float().clamp(-1, 1)  # (B,1,H,W), model range [-1, 1]


class Spec:
    """Hann-windowed rfft2 power with radial band masks."""

    def __init__(self, size, device):
        self.win = (torch.hann_window(size, periodic=False, device=device)[:, None]
                    * torch.hann_window(size, periodic=False, device=device)[None, :])
        fy = torch.fft.fftfreq(size, d=1.0 / size, device=device)[:, None]
        fx = torch.fft.rfftfreq(size, d=1.0 / size).to(device)[None, :]
        self.k = torch.sqrt(fx ** 2 + fy ** 2)
        self.masks = {n: ((self.k >= lo) & (self.k < hi)).float() for n, (lo, hi) in BANDS.items()}
        self.bins = self.k.round().long().clamp(max=size // 2)
        self.nbin = size // 2 + 1
        self.size = size

    def power(self, x):  # x (B,1,H,W) -> (B,H,W//2+1)
        x = x[:, 0]
        x = (x - x.mean(dim=(-2, -1), keepdim=True)) * self.win
        return torch.fft.rfft2(x).abs() ** 2

    def fft(self, x):  # complex (B,H,W//2+1)
        x = x[:, 0]
        x = (x - x.mean(dim=(-2, -1), keepdim=True)) * self.win
        return torch.fft.rfft2(x)

    def band_pair_stats(self, s, g):
        """Per band, per sample: normalised correlation of S with G, E(G - S) / E(G), E(S) / E(G)."""
        fs, fg = self.fft(s), self.fft(g)
        out = {}
        for n, (lo, hi) in BANDS_FINE.items():
            m = ((self.k >= lo) & (self.k < hi)).float()
            cross = (fs * fg.conj()).real
            ps, pg = fs.abs() ** 2, fg.abs() ** 2
            c = (cross * m).sum(dim=(-2, -1)) / torch.sqrt((ps * m).sum(dim=(-2, -1)) * (pg * m).sum(dim=(-2, -1)) + 1e-12)
            e_r = ((fg - fs).abs() ** 2 * m).sum(dim=(-2, -1)) / ((pg * m).sum(dim=(-2, -1)) + 1e-12)
            e_s = (ps * m).sum(dim=(-2, -1)) / ((pg * m).sum(dim=(-2, -1)) + 1e-12)
            out[n] = (c, e_r, e_s)
        return out

    def bands(self, x):  # -> dict of (B,)
        p = self.power(x)
        return {n: (p * m).sum(dim=(-2, -1)) for n, m in self.masks.items()}

    def radial(self, x):  # -> (B, nbin)
        p = self.power(x).reshape(x.shape[0], -1)
        idx = self.bins.reshape(-1)
        out = torch.zeros(x.shape[0], self.nbin, device=x.device)
        cnt = torch.zeros(self.nbin, device=x.device)
        out.index_add_(1, idx, p)
        cnt.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        return out / cnt.clamp_min(1)


def _highpass(x, k=9):
    return x - F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="reflect"), k, stride=1)


def _best_shift(s, target):
    """Integer shift of s (|dx|,|dy| <= SHIFT) that minimises the high-passed MSE to target. Per sample."""
    hs, ht = _highpass(s), _highpass(target)
    best = None
    for dy in range(-SHIFT, SHIFT + 1):
        for dx in range(-SHIFT, SHIFT + 1):
            sh = torch.roll(hs, shifts=(dy, dx), dims=(-2, -1))
            err = ((sh - ht) ** 2).mean(dim=(1, 2, 3))
            if best is None:
                best = [err, torch.full_like(err, dy), torch.full_like(err, dx)]
            else:
                m = err < best[0]
                best[0] = torch.where(m, err, best[0])
                best[1] = torch.where(m, torch.full_like(err, dy), best[1])
                best[2] = torch.where(m, torch.full_like(err, dx), best[2])
    out = torch.empty_like(s)
    for i in range(s.shape[0]):
        out[i] = torch.roll(s[i], shifts=(int(best[1][i]), int(best[2][i])), dims=(-2, -1))
    return out, best[1], best[2]


def main():
    cli = _parse_args()
    torch.manual_seed(cli.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = os.path.abspath(cli.result_dir)
    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    args.batch_size = int(cli.batch_size)

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    payload = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    w = state.get("velocity_net.scalar_mlp.0.weight")
    if w is not None and int(w.shape[1]) == 3 and bool(getattr(args, "zero_di_condition", False)):
        args.keep_zero_di_slot = True
    state = {k: v for k, v in state.items() if "u2_vss" not in k and not k.startswith("fid.")}

    extractor_args = _load_args_from_result_dir(args.extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(
        **vars(extractor_args)
    ).to(device)
    extractor.load_state_dict(
        torch.load(_resolve_required_match(os.path.join(args.extractor_path, "**/best.ckpt"), "extractor"),
                   map_location=device, weights_only=False)["state_dict"]
    )
    extractor.eval()

    transform = build_damage_image_transform_with_loss_mask(
        image_size=args.image_size, use_augmentation=False, output_scale=True,
        normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5),
        use_grayscale_input=args.use_grayscale_input, use_clahe=args.use_clahe,
        clahe_clip_limit=args.clahe_clip_limit, clahe_tile_grid_size=args.clahe_tile_grid_size,
    )
    kwargs = dict(vars(args))
    kwargs["use_loss_mask"] = True
    kwargs["use_damage_mask_condition"] = args.architecture == "CFMADM" and bool(args.use_damage_mask_condition)
    if kwargs["use_damage_mask_condition"]:
        kwargs["use_object_loss_mask"] = False
    kwargs["use_paired_bridge"] = True  # only switches on previous-frame pairing in the dataset
    kwargs["cache_data"] = True
    ds = PatternLoopDataset(root="./Data", mode=cli.split, transform=transform, extractor=extractor,
                            extractor_args=extractor_args, **kwargs)
    dl = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=cli.num_workers)
    set_dataset_args(args, dl)

    model = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(
        regressor=None, **vars(args)
    ).to(device)
    status = model.load_state_dict(state, strict=False)
    print(f"[spec] load missing={len(status.missing_keys)} unexpected={len(status.unexpected_keys)}", flush=True)
    model.eval()
    size = int(args.image_size)
    spec = Spec(size, device)

    names = [s[0] for s in SETTINGS] + ["GT previous frame (true inter-frame change)"]
    acc = {n: {"S": [], "r": [], "r_aligned": [], "radial": [], "pair": [], "gray": []} for n in names}
    gt_acc = {"prev": [], "t": [], "radial_prev": [], "radial_t": []}
    n_pairs = 0
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"spectrum-{cli.split}"):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            has_prev = batch["has_prev"].reshape(-1) >= 0.5
            if not bool(has_prev.any()):
                continue
            idx = torch.nonzero(has_prev).reshape(-1)
            info = batch["info"]
            mask = model._align_spatial_to_target(batch["mask_combined"])
            c_p, g_p = model._unpack_conditions(
                info, batch["hidden_state_prev"], batch["direction_prev"], batch["DI_prev"]
            )
            b = batch["image_target"].shape[0]
            z0 = torch.randn(b, int(model.model_in_channels), size, size, device=device)
            gt_t, gt_p = _gray(batch["image_target"])[idx], _gray(batch["image_prev"])[idx]
            gt_acc["prev"].append({k: v.cpu() for k, v in spec.bands(gt_p).items()})
            gt_acc["t"].append({k: v.cpu() for k, v in spec.bands(gt_t).items()})
            gt_acc["radial_prev"].append(spec.radial(gt_p).cpu())
            gt_acc["radial_t"].append(spec.radial(gt_t).cpu())
            n_pairs += int(idx.numel())

            states = {}
            solver0 = model.cfm_solver
            for name, solver, steps in SETTINGS:
                model.cfm_solver = solver
                states[name] = _gray(model.sample_with_ode(
                    c_p, z0=z0.clone(), gru_feat=g_p, num_steps=steps, damage_mask=mask
                ))[idx]
            model.cfm_solver = solver0
            states[names[-1]] = gt_p

            for name in names:
                s = states[name]
                r = gt_t - s
                s_al, _, _ = _best_shift(s, gt_t)
                r_al = gt_t - s_al
                acc[name]["S"].append({k: v.cpu() for k, v in spec.bands(s).items()})
                acc[name]["r"].append({k: v.cpu() for k, v in spec.bands(r).items()})
                acc[name]["r_aligned"].append({k: v.cpu() for k, v in spec.bands(r_al).items()})
                acc[name]["radial"].append(spec.radial(s).cpu())
                acc[name]["pair"].append({k: tuple(t.cpu() for t in v) for k, v in spec.band_pair_stats(s, gt_t).items()})
                acc[name]["gray"].append((s.mean(dim=(1, 2, 3)) - gt_t.mean(dim=(1, 2, 3))).abs().cpu())

    def cat(lst):
        return {k: torch.cat([d[k] for d in lst]) for k in BANDS}

    gp, gt = cat(gt_acc["prev"]), cat(gt_acc["t"])
    tot = lambda d: d["low"] + d["mid"] + d["high"]
    summary = {}
    for name in names:
        S, R, RA = cat(acc[name]["S"]), cat(acc[name]["r"]), cat(acc[name]["r_aligned"])
        summary[name] = {
            "S_band_fraction": {k: float((S[k] / tot(S)).mean()) for k in BANDS},
            "retention_E_high(S)/E_high(GT_prev)": float(S["high"].mean() / gp["high"].mean()),
            "retention_E_mid(S)/E_mid(GT_prev)": float(S["mid"].mean() / gp["mid"].mean()),
            "r_E_total_rel_to_GT_t": float(tot(R).mean() / tot(gt).mean()),
            "r_high_fraction": float((R["high"] / tot(R)).mean()),
            "r_E_high/E_high(GT_t)": float(R["high"].mean() / gt["high"].mean()),
            "r_aligned_E_total_rel_to_GT_t": float(tot(RA).mean() / tot(gt).mean()),
            "r_aligned_E_high/E_high(GT_t)": float(RA["high"].mean() / gt["high"].mean()),
        }
    for name in names:
        pairs = acc[name]["pair"]
        summary[name]["per_band_corr(S,GT_t)"] = {
            k: float(torch.cat([d[k][0] for d in pairs]).mean()) for k in BANDS_FINE}
        summary[name]["per_band_E(GT_t-S)/E(GT_t)"] = {
            k: float(torch.cat([d[k][1] for d in pairs]).mean()) for k in BANDS_FINE}
        summary[name]["per_band_E(S)/E(GT_t)"] = {
            k: float(torch.cat([d[k][2] for d in pairs]).mean()) for k in BANDS_FINE}
        summary[name]["mean_gray_abs_diff"] = float(torch.cat(acc[name]["gray"]).mean())
    summary["GT_prev_band_fraction"] = {k: float((gp[k] / tot(gp)).mean()) for k in BANDS}
    summary["GT_t_band_fraction"] = {k: float((gt[k] / tot(gt)).mean()) for k in BANDS}

    radial = {n: torch.cat(acc[n]["radial"]).mean(0).numpy() for n in names}
    radial["GT previous frame"] = torch.cat(gt_acc["radial_prev"]).mean(0).numpy()

    out_dir = cli.out_dir or os.path.join(result_dir, "eval_state_spectrum")
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"split": cli.split, "pairs": n_pairs, "image_size": size, "bands": BANDS,
               "shift_search_px": SHIFT, "summary": summary},
              open(os.path.join(out_dir, f"spectrum_{cli.split}.json"), "w"), indent=1)
    np.savez(os.path.join(out_dir, f"radial_{cli.split}.npz"), **{k: v for k, v in radial.items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    k = np.arange(len(radial["GT previous frame"]))
    for n in ["GT previous frame"] + [s[0] for s in SETTINGS]:
        ax[0].loglog(k[1:], radial[n][1:], label=n, lw=2 if n.startswith("GT") else 1.3)
    ax[0].set_xlabel("radial frequency (cycles/image)")
    ax[0].set_ylabel("mean power")
    ax[0].legend(fontsize=7)
    ax[0].set_title(f"Power spectrum of S ({cli.split}, n={n_pairs})")
    for n in [s[0] for s in SETTINGS]:
        ax[1].semilogx(k[1:], radial[n][1:] / radial["GT previous frame"][1:], label=n)
    ax[1].axhline(1.0, color="k", lw=0.8)
    ax[1].set_xlabel("radial frequency (cycles/image)")
    ax[1].set_ylabel("power ratio vs GT")
    ax[1].legend(fontsize=7)
    ax[1].set_title("Ratio to the real previous frame")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"spectrum_{cli.split}.png"), dpi=140)

    print(f"\n[spec] {cli.split}: pairs={n_pairs}, bands low<8, 8<=mid<24, high>=24 (Nyquist 64)")
    print(f"  GT prev band fraction {summary['GT_prev_band_fraction']}")
    print(f"  GT t    band fraction {summary['GT_t_band_fraction']}")
    for name in names:
        s = summary[name]
        print(f"  {name}")
        print(f"    S band fraction {s['S_band_fraction']}")
        print(f"    retention E_high {s['retention_E_high(S)/E_high(GT_prev)']:.3f}  E_mid {s['retention_E_mid(S)/E_mid(GT_prev)']:.3f}")
        print(f"    r: E_total/E_total(GT_t) {s['r_E_total_rel_to_GT_t']:.3f}  high fraction {s['r_high_fraction']:.3f}  "
              f"E_high/E_high(GT_t) {s['r_E_high/E_high(GT_t)']:.3f}")
        print(f"    r after best +-{SHIFT}px shift: E_total {s['r_aligned_E_total_rel_to_GT_t']:.3f}  "
              f"E_high {s['r_aligned_E_high/E_high(GT_t)']:.3f}")
        print("    corr(S,GT_t) per band   " + "  ".join(f"{k} {v:+.3f}" for k, v in s["per_band_corr(S,GT_t)"].items()))
        print("    E(GT_t-S)/E(GT_t)       " + "  ".join(f"{k} {v:.3f}" for k, v in s["per_band_E(GT_t-S)/E(GT_t)"].items()))
        print(f"    |mean gray S - GT_t| {s['mean_gray_abs_diff']:.4f}  (image range [-1,1])")
    print(f"[spec] saved to {out_dir}")


if __name__ == "__main__":
    main()
