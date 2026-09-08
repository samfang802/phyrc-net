#!/usr/bin/env python3
"""Demo: OT-path Heun inpainting.

U-Net spatial condition = all-ones (clean pier).
ODE replacement mask = real Data/DamagePattern/mask (SAM).
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset

# Repo root on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


def _parse_args():
    p = argparse.ArgumentParser(description="Run CFM OT-path inpaint ODE demo")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="results/DamagePatternGeneration/CFMADM/selfcondnoDI/checkpoint_step_014000.pt",
    )
    p.add_argument("--result_dir", type=str, default="")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--ode_steps", type=int, default=30)
    p.add_argument("--split", type=str, default="test", choices=("test", "val", "train"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def _load_periodic_checkpoint(model, checkpoint_path, device):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    status = model.load_state_dict(payload["state_dict"], strict=True)
    step = int(payload.get("global_step", 0))
    print(f"[inpaint] Loaded {checkpoint_path}", flush=True)
    print(f"[inpaint] load_state_dict: {status}", flush=True)
    print(f"[inpaint] global_step={step}", flush=True)
    return step


def _load_damage_pattern_mask(dataset, filename, image_size, device, dtype):
    """
    Load Data/DamagePattern/mask with the same convention as PatternLoopDataset:
    concrete/valid=1 (KEEP), rebar=0 (INPAINT).
    """
    mask_path = dataset._resolve_object_mask_path(filename)
    if mask_path is None:
        raise FileNotFoundError(
            f"DamagePattern mask not found for {filename} under {dataset.mask_dir}"
        )
    mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
    if mask_tensor.shape[-2:] != (image_size, image_size):
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="nearest",
        ).squeeze(0)
    # Invert white-rebar PNG → concrete=1, rebar=0
    valid_mask = 1.0 - (mask_tensor / 255.0)
    return valid_mask.clamp_(0.0, 1.0).to(device=device, dtype=dtype).unsqueeze(0)


def _sample_filename(dataset, subset_index):
    """Resolve original label filename for a Subset index into PatternLoopDataset."""
    real_idx = int(subset_index) % int(dataset.base_size)
    filename = dataset.label.iloc[real_idx, 0]
    return str(filename)


def _to_vis(x):
    return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)


def main():
    cli = _parse_args()
    checkpoint_path = os.path.abspath(os.path.join(_ROOT, cli.checkpoint))
    if not os.path.isfile(checkpoint_path):
        checkpoint_path = os.path.abspath(cli.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {cli.checkpoint}")

    result_dir = cli.result_dir.strip() or os.path.dirname(checkpoint_path)
    result_dir = os.path.abspath(result_dir)
    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    if int(cli.ode_steps) > 0:
        args.cfm_ode_steps = int(cli.ode_steps)

    device = torch.device(
        cli.device
        if cli.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    g = torch.Generator(device="cpu").manual_seed(int(cli.seed))
    torch.manual_seed(int(cli.seed))
    np.random.seed(int(cli.seed))

    print(f"[inpaint] result_dir={result_dir}", flush=True)
    print(
        f"[inpaint] ode_steps={args.cfm_ode_steps} solver={getattr(args, 'cfm_solver', 'heun')} "
        f"self_cond={getattr(args, 'use_self_conditioning', False)} "
        f"damage_mask_cond={getattr(args, 'use_damage_mask_condition', False)} "
        f"UNet_spatial=all-ones replace=Data/DamagePattern/mask",
        flush=True,
    )

    extractor_path = args.extractor_path
    extractor_args = _load_args_from_result_dir(extractor_path)
    extractor = importlib.import_module("src.model.HystereticPrediction").__dict__[
        extractor_args.architecture
    ](**vars(extractor_args))
    extractor = extractor.to(device)
    extractor_ckpt = _resolve_required_match(
        os.path.join(extractor_path, "**/best.ckpt"), "extractor checkpoint"
    )
    extractor.load_state_dict(
        torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"]
    )
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad = False

    val_transform = build_damage_image_transform_with_loss_mask(
        image_size=args.image_size,
        use_augmentation=False,
        output_scale=True,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
        use_grayscale_input=args.use_grayscale_input,
        use_clahe=args.use_clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid_size=args.clahe_tile_grid_size,
    )

    dataset_kwargs = dict(vars(args))
    dataset_kwargs["use_loss_mask"] = True
    dataset_kwargs["use_damage_mask_condition"] = (
        args.architecture == "CFMADM" and bool(args.use_damage_mask_condition)
    )
    # Keep object-loss mask off for batch packing parity with training args;
    # inpaint mask is loaded separately from Data/DamagePattern/mask.
    if dataset_kwargs["use_damage_mask_condition"]:
        dataset_kwargs["use_object_loss_mask"] = False

    print(f"[inpaint] Building {cli.split} dataset...", flush=True)
    ds = PatternLoopDataset(
        root=os.path.join(_ROOT, "Data"),
        mode=cli.split,
        transform=val_transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **dataset_kwargs,
    )
    n = len(ds)
    k = min(int(cli.num_samples), n)
    indices = torch.randperm(n, generator=g)[:k].tolist()
    print(f"[inpaint] Sampling {k} indices from {cli.split} ({n}): {indices}", flush=True)

    dl = DataLoader(
        Subset(ds, indices),
        batch_size=1,
        shuffle=False,
        num_workers=_resolve_num_workers(cli.num_workers),
    )

    model = importlib.import_module("src.model.ImageGeneration").__dict__[args.architecture](
        regressor=None,
        **vars(args),
    )
    step = _load_periodic_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    model.eval()

    save_dir = os.path.join(result_dir, "inference_samples")
    os.makedirs(save_dir, exist_ok=True)
    panel_path = os.path.join(
        save_dir,
        f"inpaint_dpmask_{cli.split}_step_{step:05d}_ode{int(args.cfm_ode_steps)}_n{k}.png",
    )
    single_dir = os.path.join(
        save_dir,
        f"inpaint_dpmask_{cli.split}_step_{step:05d}_ode{int(args.cfm_ode_steps)}_singles",
    )
    os.makedirs(single_dir, exist_ok=True)

    fig, axes = plt.subplots(3, k, figsize=(max(2.2 * k, 6), 7.2))
    if k == 1:
        axes = np.array(axes).reshape(3, 1)

    with torch.no_grad():
        for col, batch in enumerate(dl):
            real_data, hidden_state, direction, exp, timestep, DI, info, *_rest = (
                model._unpack_generation_batch(batch)
            )
            real_data = real_data.to(device)
            hidden_state = hidden_state.to(device)
            direction = direction.to(device)
            DI = DI.to(device)
            info = info.to(device)

            c, gru_feat = model._unpack_conditions(info, hidden_state, direction, DI)
            z1_gt = model._encode_to_target_space(real_data)
            z0 = torch.randn_like(z1_gt)

            filename = _sample_filename(ds, indices[col])
            inpaint_mask = _load_damage_pattern_mask(
                ds,
                filename,
                image_size=int(args.image_size),
                device=device,
                dtype=z1_gt.dtype,
            )
            if inpaint_mask.shape[-2:] != z1_gt.shape[-2:]:
                inpaint_mask = F.interpolate(
                    inpaint_mask, size=z1_gt.shape[-2:], mode="nearest"
                )

            hole_pct = 100.0 * float((inpaint_mask < 0.5).float().mean().item())
            # Preview dilated hole for visualization (same op as inside ODE).
            dilated_preview = 1.0 - F.max_pool2d(
                1.0 - inpaint_mask, kernel_size=11, stride=1, padding=5
            )
            dil_hole_pct = 100.0 * float((dilated_preview < 0.5).float().mean().item())
            print(
                f"[inpaint] sample={col:02d} file={filename} "
                f"c={tuple(c.shape)} gru={tuple(gru_feat.shape) if gru_feat is not None else None} "
                f"sam_hole={hole_pct:.1f}%→dil5={dil_hole_pct:.1f}% "
                f"(U-Net spatial cond=all-ones; ODE replace=dilated SAM)",
                flush=True,
            )

            z_out = model.sample_with_ode_inpaint(
                c,
                z1_gt=z1_gt,
                inpaint_mask=inpaint_mask,
                z0=z0,
                num_steps=int(args.cfm_ode_steps),
                gru_feat=gru_feat,
            )
            pred_img = model._decode_from_target_space(z_out)
            gt_vis = _to_vis(real_data)[0].detach().cpu()
            pred_vis = _to_vis(pred_img)[0].detach().cpu()
            mask_cpu = dilated_preview[0, 0].detach().cpu()
            # Visualize dilated INPAINT region (rebar hole) as dark overlay on GT
            masked_vis = gt_vis.clone()
            hole = mask_cpu < 0.5
            for ch in range(masked_vis.shape[0]):
                masked_vis[ch][hole] = 0.15

            di_val = float(DI[0].detach().cpu().item()) if DI.ndim else float(DI.item())
            exp_str = str(exp[0] if isinstance(exp, (list, tuple)) else exp)
            if torch.is_tensor(exp):
                exp_str = str(exp[0].item() if exp.ndim else exp.item())
            t_str = str(timestep[0] if not torch.is_tensor(timestep) else int(timestep[0].item()))

            axes[0, col].imshow(gt_vis.permute(1, 2, 0).numpy())
            axes[0, col].set_title(f"GT\n{exp_str} t={t_str}\nDI={di_val:.2f}", fontsize=7)
            axes[0, col].axis("off")

            axes[1, col].imshow(masked_vis.permute(1, 2, 0).numpy())
            axes[1, col].set_title(f"SAM dil5\n{dil_hole_pct:.0f}%", fontsize=7)
            axes[1, col].axis("off")

            axes[2, col].imshow(pred_vis.permute(1, 2, 0).numpy())
            axes[2, col].set_title("Inpainted", fontsize=7)
            axes[2, col].axis("off")

            single_path = os.path.join(single_dir, f"sample_{col:02d}.png")
            fig_s, ax_s = plt.subplots(1, 3, figsize=(9, 3.2))
            ax_s[0].imshow(gt_vis.permute(1, 2, 0).numpy())
            ax_s[0].set_title("GT")
            ax_s[0].axis("off")
            ax_s[1].imshow(masked_vis.permute(1, 2, 0).numpy())
            ax_s[1].set_title(f"SAM dil5 hole={dil_hole_pct:.0f}%")
            ax_s[1].axis("off")
            ax_s[2].imshow(pred_vis.permute(1, 2, 0).numpy())
            ax_s[2].set_title("Inpainted")
            ax_s[2].axis("off")
            fig_s.suptitle(
                f"step {step} · {exp_str} · t={t_str} · DI={di_val:.2f} · ODE={args.cfm_ode_steps} · {filename}",
                fontsize=8,
            )
            fig_s.tight_layout()
            fig_s.savefig(single_path, dpi=140, bbox_inches="tight")
            plt.close(fig_s)
            print(f"[inpaint] Saved {single_path} ({filename})", flush=True)

    fig.suptitle(
        f"OT-path Heun inpaint · U-Net cond=all-1 · replace=SAM dilate5 · step {step} · {cli.split} · "
        f"ODE={args.cfm_ode_steps}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[inpaint] Panel: {panel_path}", flush=True)
    print(f"[inpaint] Singles: {single_dir}", flush=True)


if __name__ == "__main__":
    main()
