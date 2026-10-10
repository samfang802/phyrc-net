#!/usr/bin/env python3
"""Compose 5 test samples from Laplacian D1/D2/D3 last.ckpt (independent ODE → reconstruct).

Default: CTR1, DI ∈ {0, 0.4, 0.6, 0.8, 1.0}, ODE=30, same panel style as CFM fixed-DI.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.model.cfm.condition import build_generation_condition, build_physics_condition, parse_fixed_di_values
from src.model.exp.laplacian_module import LaplacianBandCFM, compose_laplacian_sample
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


def _latest_run_dir(band: str) -> str:
    pattern = os.path.join(ROOT, "results/DamagePatternGeneration", f"LaplacianCFM_{band}", "*")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError(f"No run dir for band={band}: {pattern}")
    return dirs[-1]


def _last_ckpt(run_dir: str) -> str:
    # Lightning last.ckpt: csv_logs (no wandb) or wandb run dir.
    recursive = sorted(
        glob.glob(os.path.join(run_dir, "**", "last.ckpt"), recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )
    cands = recursive + [
        os.path.join(run_dir, "csv_logs/version_0/checkpoints/last-v1.ckpt"),
        os.path.join(run_dir, "checkpoint_step_014000.pt"),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"No last checkpoint under {run_dir}")


def _load_band_model(band: str, run_dir: str, ckpt_path: str, device):
    args = _load_args_from_result_dir(run_dir)
    args.result_dir = run_dir
    args.laplacian_band = band
    # condition_dim / physics_dim filled after one batch in training; restore from args.json
    model = LaplacianBandCFM(regressor=None, **vars(args))
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    status = model.load_state_dict(payload["state_dict"], strict=False)
    step = int(payload.get("global_step", -1))
    print(
        f"[lap_infer] {band}: {ckpt_path} step={step} "
        f"missing={len(status.missing_keys)} unexpected={len(status.unexpected_keys)}",
        flush=True,
    )
    model = model.to(device).eval()
    model.cfm_ode_steps = int(getattr(args, "cfm_ode_steps", 30))
    model.cfm_solver = str(getattr(args, "cfm_solver", "heun")).lower()
    return model, args, step


def _exp_str(exp):
    if isinstance(exp, (list, tuple)):
        return str(exp[0])
    if torch.is_tensor(exp):
        return str(int(exp.view(-1)[0].item())) if exp.numel() else "?"
    return str(exp)


def _collect_anchors(dataloader, di_values, anchor_exp="CTR1"):
    """Pick one sample per target DI: nearest DI among ``anchor_exp`` specimens."""
    pool = []
    for batch in dataloader:
        real, hidden, direction, exp, timestep, DI, info = batch[:7]
        for i in range(real.shape[0]):
            e = _exp_str(exp[i] if not isinstance(exp, (list, tuple)) else exp)
            if str(e) != str(anchor_exp):
                continue
            pool.append(
                {
                    "real": real[i].detach().cpu(),
                    "hidden_state": hidden[i].detach().cpu(),
                    "direction": direction[i].detach().cpu(),
                    "DI": DI[i].detach().cpu(),
                    "info": info[i].detach().cpu(),
                    "exp": e,
                    "timestep": int(timestep[i].item()) if torch.is_tensor(timestep) else int(timestep),
                    "di_gt": float(DI[i].view(-1)[0].item()),
                }
            )
    if not pool:
        raise RuntimeError(f"No test samples for exp={anchor_exp}")
    anchors = []
    used = set()
    for di in di_values:
        best_j, best_d = None, 1e9
        for j, a in enumerate(pool):
            if j in used:
                continue
            d = abs(a["di_gt"] - float(di))
            if d < best_d:
                best_d, best_j = d, j
        if best_j is None:
            # allow reuse if pool smaller than DI list
            best_j = int(np.argmin([abs(a["di_gt"] - float(di)) for a in pool]))
        used.add(best_j)
        anchors.append(pool[best_j])
        print(
            f"[lap_infer] DI={float(di):.2f} → GT_DI={pool[best_j]['di_gt']:.3f} "
            f"t={pool[best_j]['timestep']} exp={pool[best_j]['exp']}",
            flush=True,
        )
    return anchors


@torch.no_grad()
def _sample_one(models, args_ref, anchor, di_value, device, ode_steps):
    """models: dict band -> LaplacianBandCFM. Condition uses target DI (zero_di may force 0)."""
    b = 1
    hidden = anchor["hidden_state"].unsqueeze(0).to(device)
    direction = anchor["direction"].unsqueeze(0).to(device)
    info = anchor["info"].unsqueeze(0).to(device)
    di = torch.full((b, 1), float(di_value), device=device, dtype=hidden.dtype)
    # Match training: split + optional zero_di_condition
    c = build_generation_condition(
        info,
        hidden,
        direction,
        di,
        bool(getattr(args_ref, "latent_conditioning", True)),
        bool(getattr(args_ref, "direction_conditioning", True)),
        bool(getattr(args_ref, "include_di_conditioning", True)),
        str(getattr(args_ref, "condition_mode", "split")),
        zero_di_condition=bool(getattr(args_ref, "zero_di_condition", True)),
    )
    gru_feat = build_physics_condition(hidden, str(getattr(args_ref, "condition_mode", "split")))

    bands = {}
    for band, model in models.items():
        bands[band] = model.sample_band(c, gru_feat, steps=ode_steps)
    x = compose_laplacian_sample(bands["d1"], bands["d2"], bands["d3"])
    x_vis = (x * 0.5 + 0.5).clamp(0, 1)[0].detach().cpu()
    gt_vis = (anchor["real"] * 0.5 + 0.5).clamp(0, 1)
    return {
        "gt": gt_vis,
        "sampled": x_vis,
        "x1": bands["d1"][0].detach().cpu(),
        "x2": bands["d2"][0].detach().cpu(),
        "x3": bands["d3"][0].detach().cpu(),
        "di_value": float(di_value),
        "timestep": anchor["timestep"],
        "di_gt": anchor["di_gt"],
    }


def _to_gray01(t):
    if t.ndim == 3:
        t = t.mean(dim=0)
    # residuals: center for display
    return t


def _save_panel(items, out_path, step, ode_steps, anchor_exp):
    n = len(items)
    fig, axes = plt.subplots(2, n, figsize=(max(4, n * 2.2), 5.0))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for col, item in enumerate(items):
        gt = item["gt"].permute(1, 2, 0).numpy() if item["gt"].shape[0] == 3 else item["gt"].numpy()
        sm = item["sampled"].permute(1, 2, 0).numpy() if item["sampled"].shape[0] == 3 else item["sampled"].numpy()
        axes[0, col].imshow(gt)
        axes[0, col].set_title(
            f"GT DI={item['di_value']:.2f}\n(t={item['timestep']}, GT_DI={item['di_gt']:.2f})",
            fontsize=8,
        )
        axes[0, col].axis("off")
        axes[1, col].imshow(sm)
        axes[1, col].set_title(f"LapCFM compose\nODE={ode_steps}", fontsize=8)
        axes[1, col].axis("off")
    fig.suptitle(
        f"Laplacian CFM (D1+D2+D3 last.ckpt)  ·  test {anchor_exp}  ·  step={step}",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[lap_infer] saved {out_path}", flush=True)


def _save_band_breakdown(items, out_dir, step):
    """Optional per-DI band visualization (x3/x2/x1/compose)."""
    for item in items:
        di_tag = f"{item['di_value']:.2f}".replace(".", "p")
        fig, axes = plt.subplots(1, 5, figsize=(14, 3.0))
        gt = item["gt"].mean(0).numpy()
        x3 = _to_gray01(item["x3"]).numpy()
        x2 = _to_gray01(item["x2"]).numpy()
        x1 = _to_gray01(item["x1"]).numpy()
        sm = item["sampled"].mean(0).numpy()
        # upsample residuals for display
        x3u = torch.nn.functional.interpolate(
            item["x3"].unsqueeze(0).mean(1, keepdim=True), size=128, mode="bilinear", align_corners=False
        )[0, 0].numpy()
        x2u = torch.nn.functional.interpolate(
            item["x2"].unsqueeze(0).mean(1, keepdim=True), size=128, mode="bilinear", align_corners=False
        )[0, 0].numpy()
        panels = [
            (gt, "GT"),
            ((x3u - x3u.min()) / (x3u.max() - x3u.min() + 1e-8), r"$x^{(3)}$↑"),
            (np.clip(0.5 + 2.0 * x2u, 0, 1), r"$x^{(2)}$↑"),
            (np.clip(0.5 + 2.0 * x1, 0, 1), r"$x^{(1)}$"),
            (sm, "compose"),
        ]
        for ax, (im, title) in zip(axes, panels):
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Laplacian CFM bands  DI={item['di_value']:.2f}  step={step}", fontsize=10)
        plt.tight_layout()
        path = os.path.join(out_dir, f"laplacian_infer_bands_di{di_tag}.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[lap_infer] saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed_di_values", type=str, default="0,0.4,0.6,0.8,1")
    parser.add_argument("--anchor_exp", type=str, default="CTR1")
    parser.add_argument("--cfm_ode_steps", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index via CUDA_VISIBLE_DEVICES")
    args_cli = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    di_values = parse_fixed_di_values(args_cli.fixed_di_values, 5)

    runs = {b: _latest_run_dir(b) for b in ("d1", "d2", "d3")}
    ckpts = {b: _last_ckpt(runs[b]) for b in ("d1", "d2", "d3")}
    models = {}
    steps = {}
    args_ref = None
    for b in ("d1", "d2", "d3"):
        models[b], a, steps[b] = _load_band_model(b, runs[b], ckpts[b], device)
        args_ref = a

    # Build test dataloader (reuse d1 args / extractor)
    extractor_path = args_ref.extractor_path
    extractor_args = _load_args_from_result_dir(extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(**vars(extractor_args))
    extractor = extractor.to(device)
    extractor_ckpt = _resolve_required_match(
        os.path.join(extractor_path, "**/best.ckpt"), "extractor checkpoint"
    )
    extractor.load_state_dict(
        torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"]
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False

    val_transform = build_damage_image_transform_with_loss_mask(
        image_size=args_ref.image_size,
        use_augmentation=False,
        output_scale=True,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
        use_grayscale_input=args_ref.use_grayscale_input,
        use_clahe=args_ref.use_clahe,
        clahe_clip_limit=args_ref.clahe_clip_limit,
        clahe_tile_grid_size=args_ref.clahe_tile_grid_size,
    )
    dataset_kwargs = dict(vars(args_ref))
    dataset_kwargs["use_loss_mask"] = True
    dataset_kwargs["use_damage_mask_condition"] = False
    dataset_kwargs["use_object_loss_mask"] = False
    print("[lap_infer] Building test dataset...", flush=True)
    test_ds = PatternLoopDataset(
        root="./Data",
        mode="test",
        transform=val_transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **dataset_kwargs,
    )
    # Fill condition_dim like training
    loader_probe = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)
    set_dataset_args(args_ref, loader_probe)

    loader = DataLoader(
        test_ds,
        batch_size=args_cli.batch_size,
        shuffle=False,
        num_workers=_resolve_num_workers(args_cli.num_workers),
    )
    anchors = _collect_anchors(loader, di_values, anchor_exp=args_cli.anchor_exp)

    items = []
    for anchor, di in zip(anchors, di_values):
        items.append(
            _sample_one(models, args_ref, anchor, di, device, args_cli.cfm_ode_steps)
        )

    out_dir = os.path.join(runs["d1"], "inference_samples")
    step = steps["d1"]
    panel = os.path.join(
        out_dir, f"fixed_di_test_step_{step:05d}_ode{args_cli.cfm_ode_steps}_laplacian.png"
    )
    _save_panel(items, panel, step, args_cli.cfm_ode_steps, args_cli.anchor_exp)
    _save_band_breakdown(items, out_dir, step)
    print("[lap_infer] done.", flush=True)


if __name__ == "__main__":
    main()
