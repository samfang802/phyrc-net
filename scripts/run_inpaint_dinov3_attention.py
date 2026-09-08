#!/usr/bin/env python3
"""Run CFM inpaint → DINOv3 CLS attention maps for 10 random test samples."""

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

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match

try:
    from transformers import AutoImageProcessor, AutoModel
except ImportError as exc:
    raise ImportError("pip install transformers") from exc


def _parse_args():
    p = argparse.ArgumentParser(description="DINOv3 attention maps on CFM inpainted images")
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
    p.add_argument(
        "--dino_model_id",
        type=str,
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
    )
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def _load_periodic_checkpoint(model, checkpoint_path, device):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    status = model.load_state_dict(payload["state_dict"], strict=True)
    step = int(payload.get("global_step", 0))
    print(f"[dino_attn] Loaded {checkpoint_path}", flush=True)
    print(f"[dino_attn] load_state_dict: {status}", flush=True)
    print(f"[dino_attn] global_step={step}", flush=True)
    return step


def _load_damage_pattern_mask(dataset, filename, image_size, device, dtype):
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
    valid_mask = 1.0 - (mask_tensor / 255.0)
    return valid_mask.clamp_(0.0, 1.0).to(device=device, dtype=dtype).unsqueeze(0)


def _sample_filename(dataset, subset_index):
    real_idx = int(subset_index) % int(dataset.base_size)
    return str(dataset.label.iloc[real_idx, 0])


def _to_vis(x):
    return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)


def _tensor_to_pil(vis_chw: torch.Tensor) -> Image.Image:
    arr = (vis_chw.detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr)


def _cls_to_patch_attention_map(attentions, num_register_tokens: int, grid_h: int, grid_w: int):
    """
    attentions: last layer (B, heads, seq, seq)
    Token layout: [CLS | register*R | patches...]
    Returns (H_grid, W_grid) map averaged over heads, CLS→patch.
    """
    attn = attentions  # (B, heads, S, S)
    n_prefix = 1 + int(num_register_tokens)
    expected = grid_h * grid_w
    cls_to_patch = attn[:, :, 0, n_prefix : n_prefix + expected]  # (B, heads, P)
    if cls_to_patch.shape[-1] != expected:
        raise ValueError(
            f"Patch count mismatch: got {cls_to_patch.shape[-1]}, expected {expected}"
        )
    # Mean over heads, take batch 0
    amap = cls_to_patch[0].mean(dim=0).reshape(grid_h, grid_w)
    amap = amap - amap.min()
    amap = amap / (amap.max() + 1e-8)
    return amap


def _overlay_attention(pil_rgb: Image.Image, attn_hw: torch.Tensor, alpha: float = 0.45):
    h, w = pil_rgb.size[1], pil_rgb.size[0]
    attn = F.interpolate(
        attn_hw.view(1, 1, *attn_hw.shape).float(),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    attn_np = attn.detach().cpu().numpy()
    heat = plt.cm.inferno(attn_np)[..., :3]  # RGB float
    base = np.asarray(pil_rgb).astype(np.float32) / 255.0
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)
    blend = (1.0 - alpha) * base + alpha * heat
    blend = (np.clip(blend, 0, 1) * 255.0).astype(np.uint8)
    return Image.fromarray(blend), attn_np


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
        cli.device if cli.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    g = torch.Generator(device="cpu").manual_seed(int(cli.seed))
    torch.manual_seed(int(cli.seed))
    np.random.seed(int(cli.seed))

    # HF auth check
    try:
        from huggingface_hub import whoami

        info = whoami()
        print(f"[dino_attn] HF logged in as: {info.get('name') or info.get('fullname')}", flush=True)
    except Exception as exc:
        print(f"[dino_attn] HF whoami failed ({exc}); continuing with cached auth.", flush=True)

    print(f"[dino_attn] result_dir={result_dir}", flush=True)
    print(f"[dino_attn] dino={cli.dino_model_id}", flush=True)

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
    if dataset_kwargs["use_damage_mask_condition"]:
        dataset_kwargs["use_object_loss_mask"] = False

    print(f"[dino_attn] Building {cli.split} dataset...", flush=True)
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
    print(f"[dino_attn] Sampling {k} indices: {indices}", flush=True)

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

    print(f"[dino_attn] Loading DINOv3...", flush=True)
    dino_processor = AutoImageProcessor.from_pretrained(cli.dino_model_id)
    dino_model = AutoModel.from_pretrained(cli.dino_model_id).to(device).eval()
    dino_model.config._attn_implementation = "eager"
    num_register = int(getattr(dino_model.config, "num_register_tokens", 0) or 0)
    patch_size = int(dino_model.config.patch_size)

    save_dir = os.path.join(result_dir, "inference_samples")
    os.makedirs(save_dir, exist_ok=True)
    panel_path = os.path.join(
        save_dir,
        f"inpaint_dinov3_attn_{cli.split}_step_{step:05d}_ode{int(args.cfm_ode_steps)}_n{k}.png",
    )
    single_dir = os.path.join(
        save_dir,
        f"inpaint_dinov3_attn_{cli.split}_step_{step:05d}_ode{int(args.cfm_ode_steps)}_singles",
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

            z_out = model.sample_with_ode_inpaint(
                c,
                z1_gt=z1_gt,
                inpaint_mask=inpaint_mask,
                z0=z0,
                num_steps=int(args.cfm_ode_steps),
                gru_feat=gru_feat,
            )
            pred_img = model._decode_from_target_space(z_out)
            gt_vis = _to_vis(real_data)[0]
            pred_vis = _to_vis(pred_img)[0]
            inpaint_pil = _tensor_to_pil(pred_vis)

            # DINOv3 attention on inpainted image
            dino_in = dino_processor(images=inpaint_pil, return_tensors="pt")
            pixel_values = dino_in["pixel_values"].to(device)
            h_in, w_in = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
            grid_h, grid_w = h_in // patch_size, w_in // patch_size

            dino_out = dino_model(pixel_values=pixel_values, output_attentions=True)
            if dino_out.attentions is None:
                raise RuntimeError(
                    "DINOv3 returned attentions=None; ensure attn_implementation=eager"
                )
            attn_map = _cls_to_patch_attention_map(
                dino_out.attentions[-1],
                num_register_tokens=num_register,
                grid_h=grid_h,
                grid_w=grid_w,
            )
            overlay_pil, attn_np = _overlay_attention(inpaint_pil, attn_map)

            di_val = float(DI[0].detach().cpu().item()) if DI.ndim else float(DI.item())
            exp_str = str(exp[0] if isinstance(exp, (list, tuple)) else exp)
            if torch.is_tensor(exp):
                exp_str = str(exp[0].item() if exp.ndim else exp.item())
            t_str = str(
                timestep[0] if not torch.is_tensor(timestep) else int(timestep[0].item())
            )

            axes[0, col].imshow(gt_vis.detach().cpu().permute(1, 2, 0).numpy())
            axes[0, col].set_title(f"GT\n{exp_str} t={t_str}\nDI={di_val:.2f}", fontsize=7)
            axes[0, col].axis("off")

            axes[1, col].imshow(np.asarray(inpaint_pil))
            axes[1, col].set_title("Inpainted", fontsize=7)
            axes[1, col].axis("off")

            axes[2, col].imshow(np.asarray(overlay_pil))
            axes[2, col].set_title("DINOv3 attn", fontsize=7)
            axes[2, col].axis("off")

            single_path = os.path.join(single_dir, f"sample_{col:02d}.png")
            fig_s, ax_s = plt.subplots(1, 4, figsize=(12, 3.2))
            ax_s[0].imshow(gt_vis.detach().cpu().permute(1, 2, 0).numpy())
            ax_s[0].set_title("GT")
            ax_s[0].axis("off")
            ax_s[1].imshow(np.asarray(inpaint_pil))
            ax_s[1].set_title("Inpainted")
            ax_s[1].axis("off")
            ax_s[2].imshow(attn_np, cmap="inferno")
            ax_s[2].set_title("Attn map")
            ax_s[2].axis("off")
            ax_s[3].imshow(np.asarray(overlay_pil))
            ax_s[3].set_title("Overlay")
            ax_s[3].axis("off")
            fig_s.suptitle(
                f"DINOv3 CLS→patch · step {step} · {exp_str} · t={t_str} · DI={di_val:.2f} · {filename}",
                fontsize=8,
            )
            fig_s.tight_layout()
            fig_s.savefig(single_path, dpi=140, bbox_inches="tight")
            plt.close(fig_s)
            print(f"[dino_attn] Saved {single_path}", flush=True)

    fig.suptitle(
        f"Inpaint → DINOv3 attention · step {step} · {cli.split} · ODE={args.cfm_ode_steps} · "
        f"{cli.dino_model_id.split('/')[-1]}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[dino_attn] Panel: {panel_path}", flush=True)
    print(f"[dino_attn] Singles: {single_dir}", flush=True)


if __name__ == "__main__":
    main()
