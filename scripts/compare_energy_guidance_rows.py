#!/usr/bin/env python3
"""5-row comparison panel: GT vs custom-prior z0_new (+ energy guidance variants).

Rows
----
1. Ground Truth (val, nearest DI)
2. z0_new only (Top-3 HysGRU + Slerp, no energy J)
3. z0_new + J_TV (Total Variation + 8-neighbor island penalty)
4. z0_new + J_texture (Laplacian high-frequency / roughness)
5. z0_new + J_spatial (plastic-hinge spatial weight)
"""

from __future__ import annotations

import argparse
import importlib
import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


EnergyFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _parse_args():
    p = argparse.ArgumentParser(description="5-row energy-guidance comparison panel")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="results/DamagePatternGeneration/CFMADM/2026-07-26_21-24-55/checkpoint_step_014000.pt",
    )
    p.add_argument("--result_dir", type=str, default="")
    p.add_argument("--fixed_di_values", type=str, default="0.0,0.4,0.75,0.95")
    p.add_argument("--val_anchor_exp", type=str, default="R615")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--ode_steps", type=int, default=0, help="0 = use args.json")
    p.add_argument("--guidance_tv", type=float, default=2.0)
    p.add_argument("--guidance_texture", type=float, default=3.0)
    p.add_argument("--guidance_spatial", type=float, default=4.0)
    p.add_argument("--island_threshold", type=float, default=0.35)
    p.add_argument("--spatial_alpha", type=float, default=3.0)
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path (default: result_dir/inference_samples/energy_guidance_rows_step_XXXXX.png)",
    )
    return p.parse_args()


def _parse_di(raw: str):
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("fixed_di_values empty")
    return vals


def _load_ckpt(model, path, device):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    payload = torch.load(path, map_location=device, weights_only=False)
    status = model.load_state_dict(payload["state_dict"], strict=True)
    step = int(payload.get("global_step", 0))
    print(f"[rows] Loaded {path}", flush=True)
    print(f"[rows] {status} | step={step}", flush=True)
    return step


def _tensor_to_vis(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)


def _slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two tensors of the same shape."""
    shape = a.shape
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    a_norm = torch.linalg.vector_norm(a_flat).clamp_min(1e-8)
    b_norm = torch.linalg.vector_norm(b_flat).clamp_min(1e-8)
    a_n = a_flat / a_norm
    b_n = b_flat / b_norm
    dot = torch.clamp(torch.dot(a_n, b_n), -1.0 + 1e-7, 1.0 - 1e-7)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(1e-8)
    coef_a = torch.sin((1.0 - t) * omega) / sin_omega
    coef_b = torch.sin(t * omega) / sin_omega
    out = coef_a * a_flat + coef_b * b_flat
    # Preserve average radius of the two endpoints.
    target_norm = (1.0 - t) * a_norm + t * b_norm
    out = out / torch.linalg.vector_norm(out).clamp_min(1e-8) * target_norm
    return out.reshape(shape).to(dtype=a.dtype)


def _slerp_average(tensors):
    if len(tensors) == 1:
        return tensors[0]
    acc = tensors[0]
    for i, z in enumerate(tensors[1:], start=1):
        t = 1.0 / float(i + 1)
        acc = _slerp(acc, z, t)
    return acc


def _build_coords(model, batch_size, device, dtype):
    if not model.use_patch_diffusion:
        return None
    grid_x, grid_y = model._build_full_coordinate_maps(
        model.model_spatial_size,
        model.model_spatial_size,
        device=device,
        dtype=dtype,
    )
    return (
        torch.stack([grid_x, grid_y], dim=0)
        .unsqueeze(0)
        .expand(batch_size, -1, -1, -1)
        .contiguous()
    )


def _build_condition_mask(model, batch_size, spatial, device, dtype):
    if not model.use_damage_mask_condition:
        return None
    h, w = spatial
    return torch.ones((batch_size, 1, h, w), device=device, dtype=dtype)


def reverse_ode_to_z0(model, z1, c, gru_feat=None, num_steps=None):
    """Integrate CFM ODE backward from t=1 (data) to t=0 (noise)."""
    if num_steps is None:
        num_steps = model.cfm_ode_steps
    z = z1
    coords = _build_coords(model, c.shape[0], z.device, z.dtype)
    condition_mask = _build_condition_mask(model, c.shape[0], z.shape[-2:], z.device, z.dtype)
    dt = 1.0 / float(num_steps)
    # Walk i = N-1 ... 0; at step i integrate from t=(i+1)/N to t=i/N using -v.
    for i in reversed(range(num_steps)):
        t_scalar = float(i + 1) / float(num_steps)
        t = torch.full((c.shape[0],), t_scalar, device=c.device)
        with torch.no_grad():
            v_t = model._predict_velocity(
                z, t, c, coords=coords, damage_mask=condition_mask, gru_feat=gru_feat
            )
            if model.cfm_solver == "heun":
                z_pred = z - dt * v_t
                t_prev_scalar = float(i) / float(num_steps)
                t_prev = torch.full((c.shape[0],), t_prev_scalar, device=c.device)
                v_prev = model._predict_velocity(
                    z_pred, t_prev, c, coords=coords, damage_mask=condition_mask, gru_feat=gru_feat
                )
                z = z - 0.5 * dt * (v_t + v_prev)
            else:
                z = z - dt * v_t
    return z


def sample_ode_guided(
    model,
    c,
    z0,
    gru_feat=None,
    num_steps=None,
    energy_fn: Optional[EnergyFn] = None,
    guidance_scale: float = 0.0,
    di_tensor: Optional[torch.Tensor] = None,
):
    """Forward ODE from z0; optional energy guidance v <- v - λ ∇_z J(decode(z))."""
    if num_steps is None:
        num_steps = model.cfm_ode_steps
    z = z0
    coords = _build_coords(model, c.shape[0], z.device, z.dtype)
    condition_mask = _build_condition_mask(model, c.shape[0], z.shape[-2:], z.device, z.dtype)
    dt = 1.0 / float(num_steps)

    def _velocity(z_in, t_in):
        v = model._predict_velocity(
            z_in, t_in, c, coords=coords, damage_mask=condition_mask, gru_feat=gru_feat
        )
        if energy_fn is None or guidance_scale == 0.0:
            return v
        z_req = z_in.detach().requires_grad_(True)
        img = model._decode_from_target_space(z_req)
        img_vis = torch.clamp(img * 0.5 + 0.5, 0.0, 1.0)
        di = di_tensor if di_tensor is not None else torch.zeros(c.shape[0], device=c.device)
        j = energy_fn(img_vis, di)
        grad = torch.autograd.grad(j, z_req, create_graph=False)[0]
        return v - guidance_scale * grad

    for i in range(num_steps):
        t_scalar = float(i) / float(num_steps)
        t = torch.full((c.shape[0],), t_scalar, device=c.device)
        if energy_fn is None or guidance_scale == 0.0:
            with torch.no_grad():
                v_t = _velocity(z, t)
                if model.cfm_solver == "heun":
                    z_pred = z + dt * v_t
                    t_next = torch.full(
                        (c.shape[0],), float(i + 1) / float(num_steps), device=c.device
                    )
                    v_next = _velocity(z_pred, t_next)
                    z = z + 0.5 * dt * (v_t + v_next)
                else:
                    z = z + dt * v_t
        else:
            v_t = _velocity(z, t)
            if model.cfm_solver == "heun":
                z_pred = (z + dt * v_t).detach()
                t_next = torch.full(
                    (c.shape[0],), float(i + 1) / float(num_steps), device=c.device
                )
                v_next = _velocity(z_pred, t_next)
                z = (z + 0.5 * dt * (v_t + v_next)).detach()
            else:
                z = (z + dt * v_t).detach()
    return z.detach()


# ---------------------------------------------------------------------------
# Energy functions J(image_vis in [0,1], di)
# ---------------------------------------------------------------------------

def energy_tv_island(img_vis: torch.Tensor, di: torch.Tensor, island_thr: float = 0.35) -> torch.Tensor:
    """Total Variation + isolated dark-pixel (8-neighborhood) penalty."""
    # TV
    dx = (img_vis[:, :, :, 1:] - img_vis[:, :, :, :-1]).abs().mean()
    dy = (img_vis[:, :, 1:, :] - img_vis[:, :, :-1, :]).abs().mean()
    j_tv = dx + dy

    gray = img_vis.mean(dim=1, keepdim=True)
    dark = (gray < island_thr).float()
    kernel = torch.ones((1, 1, 3, 3), device=img_vis.device, dtype=img_vis.dtype)
    kernel[0, 0, 1, 1] = 0.0
    neighbor = F.conv2d(dark, kernel, padding=1)
    isolated = dark * (neighbor < 0.5)
    j_island = isolated.mean()
    return j_tv + 5.0 * j_island


def energy_laplacian_roughness(img_vis: torch.Tensor, di: torch.Tensor) -> torch.Tensor:
    """Penalize insufficient high-frequency energy at high DI."""
    gray = img_vis.mean(dim=1, keepdim=True)
    lap_k = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=img_vis.device,
        dtype=img_vis.dtype,
    ).view(1, 1, 3, 3)
    resp = F.conv2d(gray, lap_k, padding=1)
    # Per-sample spatial variance of Laplacian response.
    var = resp.flatten(1).var(dim=1, unbiased=False)
    # Target roughness grows with DI (empirically scaled for 128x128 damage images).
    di_flat = di.view(-1).to(dtype=var.dtype)
    target = 0.02 + 0.18 * di_flat.clamp(0.0, 1.5)
    deficit = (target - var).clamp(min=0.0)
    return (deficit ** 2).mean()


def make_energy_spatial(alpha: float = 3.0) -> EnergyFn:
    """Plastic-hinge prior: weight 0 at image bottom, rises toward mid/upper column."""

    def energy_spatial(img_vis: torch.Tensor, di: torch.Tensor) -> torch.Tensor:
        b, _, h, w = img_vis.shape
        gray = img_vis.mean(dim=1, keepdim=True)
        damage = (1.0 - gray).clamp(0.0, 1.0)
        # Row 0 = top of image / upper column; row H-1 = bottom / plastic hinge.
        rows = torch.linspace(0.0, 1.0, h, device=img_vis.device, dtype=img_vis.dtype)
        h_from_bottom = 1.0 - rows  # 0 at bottom, 1 at top
        w_1d = (torch.exp(alpha * h_from_bottom) - 1.0) / (torch.exp(torch.tensor(alpha, device=img_vis.device)) - 1.0)
        w_map = w_1d.view(1, 1, h, 1).expand(b, 1, h, w)
        # Soft area penalty: large dark blobs where weight is high.
        return (damage * w_map).mean()

    return energy_spatial


def _collect_gallery(dataloader, max_items=None):
    """Collect (hidden, image[-1,1], direction, exp, timestep, DI, info) from a split."""
    gallery = []
    with torch.no_grad():
        for batch in dataloader:
            real_data, hidden_state, direction, exp, timestep, DI, info = batch[:7]
            for i in range(real_data.shape[0]):
                e = exp[i] if isinstance(exp, (list, tuple)) else exp[i]
                if torch.is_tensor(e):
                    e = e.item() if e.ndim == 0 else str(e)
                gallery.append(
                    {
                        "hidden": hidden_state[i].detach().cpu(),
                        "image": real_data[i].detach().cpu(),
                        "direction": direction[i].detach().cpu(),
                        "exp": str(e),
                        "timestep": int(timestep[i].item() if torch.is_tensor(timestep[i]) else timestep[i]),
                        "di": float(DI[i].item() if torch.is_tensor(DI[i]) else DI[i]),
                        "info": info[i].detach().cpu(),
                    }
                )
                if max_items is not None and len(gallery) >= max_items:
                    return gallery
    return gallery


def _top_k_by_hidden(gallery, query_hidden, k=3):
    q = query_hidden.float().reshape(-1)
    q = q / q.norm().clamp_min(1e-8)
    scores = []
    for idx, item in enumerate(gallery):
        h = item["hidden"].float().reshape(-1)
        h = h / h.norm().clamp_min(1e-8)
        scores.append((float(torch.dot(q, h)), idx))
    scores.sort(key=lambda x: -x[0])
    return [gallery[i] for _, i in scores[:k]]


def _build_dataloader(args, extractor, extractor_args, mode, batch_size, num_workers):
    transform = build_damage_image_transform_with_loss_mask(
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
    # Gallery / inference: no virtual repeat (avoid 521 → 52100).
    dataset_kwargs["train_virtual_repeat"] = 1
    dataset_kwargs["use_damage_mask_condition"] = (
        args.architecture == "CFMADM" and bool(args.use_damage_mask_condition)
    )
    if dataset_kwargs["use_damage_mask_condition"]:
        dataset_kwargs["use_object_loss_mask"] = False
    ds = PatternLoopDataset(
        root="./Data",
        mode=mode,
        transform=transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **dataset_kwargs,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return ds, dl


def _save_panel(rows, di_values, row_labels, out_path, step, val_exp):
    n_rows = len(rows)
    n_cols = len(di_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(4, n_cols * 2.4), n_rows * 2.3))
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for r, (label, images) in enumerate(zip(row_labels, rows)):
        for c, (img, di) in enumerate(zip(images, di_values)):
            arr = img.detach().cpu()
            if arr.ndim == 3:
                arr = arr.permute(1, 2, 0).numpy()
            axes[r, c].imshow(np.clip(arr, 0, 1))
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"DI={di:.2f}", fontsize=9)
            if c == 0:
                axes[r, c].text(
                    -0.08,
                    0.5,
                    label,
                    transform=axes[r, c].transAxes,
                    va="center",
                    ha="right",
                    fontsize=7,
                    rotation=90,
                )

    fig.suptitle(
        f"step {step} · val {val_exp} · custom prior $z_{{0,new}}$ + energy guidance",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[rows] Saved panel: {out_path}", flush=True)


def main():
    cli = _parse_args()
    ckpt = os.path.abspath(cli.checkpoint)
    result_dir = os.path.abspath(cli.result_dir.strip() or os.path.dirname(ckpt))
    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    if cli.batch_size > 0:
        args.batch_size = cli.batch_size
    di_values = _parse_di(cli.fixed_di_values)
    args.inference_num_images = len(di_values)
    args.fixed_di_values = cli.fixed_di_values
    args.inference_val_anchor_exp = cli.val_anchor_exp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = _resolve_num_workers(cli.num_workers)
    ode_steps = cli.ode_steps if cli.ode_steps > 0 else None

    extractor_args = _load_args_from_result_dir(args.extractor_path)
    extractor = importlib.import_module("src.model.HystereticPrediction").__dict__[
        extractor_args.architecture
    ](**vars(extractor_args)).to(device)
    extractor_ckpt = _resolve_required_match(
        os.path.join(args.extractor_path, "**/best.ckpt"), "extractor checkpoint"
    )
    extractor.load_state_dict(
        torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"]
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False
    print("[rows] Extractor loaded.", flush=True)

    print("[rows] Building train gallery...", flush=True)
    _, train_dl = _build_dataloader(
        args, extractor, extractor_args, "train", args.batch_size, num_workers
    )
    set_dataset_args(args, train_dl)
    train_gallery = _collect_gallery(train_dl)
    print(f"[rows] train gallery={len(train_gallery)}", flush=True)

    print("[rows] Building val set...", flush=True)
    _, val_dl = _build_dataloader(
        args, extractor, extractor_args, "val", args.batch_size, num_workers
    )
    val_gallery = _collect_gallery(val_dl)
    print(f"[rows] val gallery={len(val_gallery)}", flush=True)

    model = importlib.import_module("src.model.ImageGeneration").__dict__[args.architecture](
        regressor=None,
        **vars(args),
    ).to(device)
    step = _load_ckpt(model, ckpt, device)
    model.eval()

    # Select val anchors per DI (same logic as training inference panel).
    model.validation_outputs = {
        "gt_DI": [],
        "pred_DI": [],
        "train_anchor_candidates": [],
        "val_anchor_candidates": [],
        "test_anchor_candidates": [],
    }
    for item in val_gallery:
        model.validation_outputs["val_anchor_candidates"].append(
            {
                "split": "val",
                "real": _tensor_to_vis(item["image"]),
                "hidden_state": item["hidden"],
                "direction": item["direction"],
                "info": item["info"],
                "exp": item["exp"],
                "timestep": item["timestep"],
                "gt_di": item["di"],
                "sample_uid": f"val_{item['exp']}_{item['timestep']}",
            }
        )
    if hasattr(model, "_fixed_val_inference_anchors"):
        delattr(model, "_fixed_val_inference_anchors")
    val_anchors = model._select_anchors_for_di_values(
        model.validation_outputs["val_anchor_candidates"],
        di_values,
        split="val",
        anchor_exp=cli.val_anchor_exp,
    )
    if len(val_anchors) != len(di_values):
        raise RuntimeError("Failed to select val anchors for all DI values")

    energy_spatial = make_energy_spatial(alpha=cli.spatial_alpha)
    island_thr = cli.island_threshold

    def j_tv(img, di):
        return energy_tv_island(img, di, island_thr=island_thr)

    row_gt = []
    row_prior = []
    row_tv = []
    row_texture = []
    row_spatial = []

    for di_value, anchor in zip(di_values, val_anchors):
        print(
            f"[rows] DI={di_value:.2f} exp={anchor['exp']} t={anchor['timestep']} "
            f"GT_DI={anchor['gt_di']:.3f}",
            flush=True,
        )
        row_gt.append(anchor["real"])

        query_h = anchor["hidden_state"]
        topk = _top_k_by_hidden(train_gallery, query_h, k=cli.top_k)
        print(
            "[rows]   Top-{} train: {}".format(
                cli.top_k,
                ", ".join(f"{t['exp']}@t{t['timestep']}(DI={t['di']:.2f})" for t in topk),
            ),
            flush=True,
        )

        # Reverse each Top-k real image to z0 under its own condition, then Slerp.
        z0_list = []
        for t_item in topk:
            img = t_item["image"].unsqueeze(0).to(device)
            hidden = t_item["hidden"].unsqueeze(0).to(device)
            direction = t_item["direction"].unsqueeze(0).to(device)
            info = t_item["info"].unsqueeze(0).to(device)
            di = torch.tensor([[t_item["di"]]], device=device, dtype=info.dtype)
            c_k, gru_k = model._unpack_conditions(info, hidden, direction, di)
            z1 = model._encode_to_target_space(img)
            z0_k = reverse_ode_to_z0(model, z1, c_k, gru_feat=gru_k, num_steps=ode_steps)
            z0_list.append(z0_k[0].detach())
        z0_new = _slerp_average(z0_list).unsqueeze(0).to(device)

        # Target (val) condition for forward generation.
        hidden = anchor["hidden_state"].unsqueeze(0).to(device)
        direction = anchor["direction"].unsqueeze(0).to(device)
        info = anchor["info"].unsqueeze(0).to(device)
        di = torch.tensor([[float(di_value)]], device=device, dtype=info.dtype)
        c, gru_feat = model._unpack_conditions(info, hidden, direction, di)

        def _gen(energy_fn=None, scale=0.0):
            z = sample_ode_guided(
                model,
                c,
                z0_new.clone(),
                gru_feat=gru_feat,
                num_steps=ode_steps,
                energy_fn=energy_fn,
                guidance_scale=scale,
                di_tensor=di,
            )
            img = model._decode_from_target_space(z)
            return _tensor_to_vis(img)[0].detach().cpu()

        row_prior.append(_gen(None, 0.0))
        row_tv.append(_gen(j_tv, cli.guidance_tv))
        row_texture.append(_gen(energy_laplacian_roughness, cli.guidance_texture))
        row_spatial.append(_gen(energy_spatial, cli.guidance_spatial))

    row_labels = [
        "Row1 GT",
        r"Row2 $z_{0,new}$",
        r"Row3 $z_{0,new}$+TV&Island",
        r"Row4 $z_{0,new}$+Laplacian",
        r"Row5 $z_{0,new}$+PlasticHinge",
    ]
    out_path = cli.output.strip() or os.path.join(
        result_dir,
        "inference_samples",
        f"energy_guidance_rows_step_{step:05d}.png",
    )
    _save_panel(
        [row_gt, row_prior, row_tv, row_texture, row_spatial],
        di_values,
        row_labels,
        out_path,
        step,
        cli.val_anchor_exp,
    )
    print("[rows] Done.", flush=True)


if __name__ == "__main__":
    main()
