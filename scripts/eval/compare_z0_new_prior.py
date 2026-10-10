#!/usr/bin/env python3
"""Generate (1) GT + z0_new prior inference and (2) Top-3 train real images per DI.

Retrieval score
---------------
Score = W_DI * (1 - |DI_target - DI_train|) + W_GRU * CosineSim(g_target, g_train)
with W_DI=0.8, W_GRU=0.2. Candidates with |DI_target - DI_train| > 0.1 are discarded.

Before reverse-ODE, train images are OpenCV-inpainted using the interference mask
(white=255 = interference) so rebar/clutter is filled with surrounding concrete texture.

Spatial damage mask at inference is forced to all-ones (full domain).
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_energy_guidance_rows import (  # noqa: E402
    _build_dataloader,
    _load_ckpt,
    _parse_di,
    _slerp_average,
    _tensor_to_vis,
    reverse_ode_to_z0,
    sample_ode_guided,
)
from argument.argument_mixed import set_dataset_args
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


def _parse_args():
    p = argparse.ArgumentParser(description="z0_new prior + Top-3 with DI/GRU score + inpaint")
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
    p.add_argument("--ode_steps", type=int, default=0)
    p.add_argument("--w_di", type=float, default=0.8)
    p.add_argument("--w_gru", type=float, default=0.2)
    p.add_argument("--di_max_diff", type=float, default=0.1)
    p.add_argument("--inpaint_radius", type=int, default=5)
    return p.parse_args()


def _resolve_mask_path(mask_dir, filename):
    stem, _ = os.path.splitext(str(filename))
    for path in (
        os.path.join(mask_dir, f"{stem}_mask.png"),
        os.path.join(mask_dir, f"{stem}.png"),
        os.path.join(mask_dir, str(filename)),
    ):
        if os.path.isfile(path):
            return path
    return None


def _inpaint_interference(image_rgb_u8: np.ndarray, mask_u8: np.ndarray, radius: int = 5) -> np.ndarray:
    """OpenCV Telea inpaint. mask: interference region = 255."""
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[..., 0]
    if mask_u8.shape[:2] != image_rgb_u8.shape[:2]:
        mask_u8 = cv2.resize(
            mask_u8,
            (image_rgb_u8.shape[1], image_rgb_u8.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    binary = (mask_u8 > 127).astype(np.uint8) * 255
    if int(binary.sum()) == 0:
        return image_rgb_u8
    return cv2.inpaint(image_rgb_u8, binary, inpaintRadius=int(radius), flags=cv2.INPAINT_TELEA)


def _collect_gallery_with_filename(dataset):
    """Iterate unique samples; store filename for mask lookup + inpaint."""
    gallery = []
    n = int(getattr(dataset, "base_size", len(dataset)))
    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]
            real_data, hidden_state, direction, exp, timestep, DI, info = sample[:7]
            filename = str(dataset.label.iloc[i % dataset.base_size]["filename"])
            e = exp
            if torch.is_tensor(e):
                e = e.item() if e.ndim == 0 else str(e)
            gallery.append(
                {
                    "filename": filename,
                    "hidden": hidden_state.detach().cpu(),
                    "image": real_data.detach().cpu(),
                    "direction": direction.detach().cpu()
                    if torch.is_tensor(direction)
                    else direction,
                    "exp": str(e),
                    "timestep": int(timestep.item() if torch.is_tensor(timestep) else timestep),
                    "di": float(DI.item() if torch.is_tensor(DI) else DI[0] if hasattr(DI, "__len__") else DI),
                    "info": info.detach().cpu(),
                }
            )
    return gallery


def _top_k_by_score(
    gallery,
    query_hidden,
    target_di,
    k=3,
    w_di=0.8,
    w_gru=0.2,
    di_max_diff=0.1,
):
    """Score = W_DI*(1-|dDI|) + W_GRU*cos; drop |dDI| > di_max_diff; unique (exp,t)."""
    q = query_hidden.float().reshape(-1)
    q = q / q.norm().clamp_min(1e-8)
    scored = []
    for idx, item in enumerate(gallery):
        ddi = abs(float(item["di"]) - float(target_di))
        if ddi > di_max_diff:
            continue
        h = item["hidden"].float().reshape(-1)
        h = h / h.norm().clamp_min(1e-8)
        cos = float(torch.dot(q, h))
        score = float(w_di) * (1.0 - ddi) + float(w_gru) * cos
        scored.append((score, cos, ddi, idx))
    scored.sort(key=lambda x: -x[0])

    selected = []
    seen = set()
    for score, cos, ddi, idx in scored:
        item = gallery[idx]
        key = (str(item["exp"]), int(item["timestep"]))
        if key in seen:
            continue
        seen.add(key)
        selected.append((score, cos, ddi, item))
        if len(selected) >= k:
            break
    return selected


def _prepare_inpainted_tensor(item, transform, image_root, mask_dir, image_size, inpaint_radius):
    """Load raw X_train + interference mask, OpenCV inpaint, then apply model transform → [-1,1]."""
    img_path = os.path.join(image_root, item["filename"])
    mask_path = _resolve_mask_path(mask_dir, item["filename"])
    pil = Image.open(img_path).convert("RGB")
    rgb = np.array(pil)
    if mask_path is None:
        inpainted = rgb
        print(f"[z0_new]   mask missing for {item['filename']}; skip inpaint", flush=True)
    else:
        mask = np.array(Image.open(mask_path).convert("L"))
        inpainted = _inpaint_interference(rgb, mask, radius=inpaint_radius)

    # Resize to model size before transform (transform also resizes, but keep consistent).
    inpainted_pil = Image.fromarray(inpainted).resize((image_size, image_size), Image.BILINEAR)
    out = transform(inpainted_pil)
    if isinstance(out, dict):
        tensor = out["target_image"]
    else:
        tensor = out
    return tensor


def _save_gt_prior_panel(gt_imgs, prior_imgs, di_values, out_path, step, val_exp):
    n_cols = len(di_values)
    fig, axes = plt.subplots(2, n_cols, figsize=(max(4, n_cols * 2.4), 4.8))
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    row_labels = ["GT (val)", r"$z_{0,new}$ prior"]
    for r, (label, images) in enumerate(zip(row_labels, [gt_imgs, prior_imgs])):
        for c, (img, di) in enumerate(zip(images, di_values)):
            arr = img.detach().cpu().permute(1, 2, 0).numpy()
            axes[r, c].imshow(np.clip(arr, 0, 1))
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"DI={di:.2f}", fontsize=10)
            if c == 0:
                axes[r, c].text(
                    -0.08,
                    0.5,
                    label,
                    transform=axes[r, c].transAxes,
                    va="center",
                    ha="right",
                    fontsize=9,
                    rotation=90,
                )
    fig.suptitle(
        f"step {step} · val {val_exp} · mask=all-ones · "
        r"$z_{0,new}$ (score=$0.8\cdot DI+0.2\cdot$GRU, inpaint)",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[z0_new] Saved prior panel: {out_path}", flush=True)


def _save_topk_panel(topk_per_di, di_values, out_path, step, val_exp, top_k, show_inpainted=None):
    """show_inpainted: optional list[list[CHW tensor]] aligned with topk_per_di (inpainted views)."""
    n_cols = len(di_values)
    fig, axes = plt.subplots(top_k, n_cols, figsize=(max(4, n_cols * 2.6), top_k * 2.4))
    if top_k == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(top_k, 1)

    for c, (di, ranked) in enumerate(zip(di_values, topk_per_di)):
        for r in range(top_k):
            ax = axes[r, c]
            if r >= len(ranked):
                ax.axis("off")
                continue
            score, cos, ddi, item = ranked[r]
            if show_inpainted is not None and show_inpainted[c][r] is not None:
                arr = _tensor_to_vis(show_inpainted[c][r]).permute(1, 2, 0).numpy()
            else:
                arr = _tensor_to_vis(item["image"]).permute(1, 2, 0).numpy()
            ax.imshow(np.clip(arr, 0, 1))
            ax.axis("off")
            ax.set_title(
                f"#{r+1} {item['exp']} t={item['timestep']}\n"
                f"DI={item['di']:.2f} score={score:.3f}\n"
                f"cos={cos:.3f} |dDI|={ddi:.3f}",
                fontsize=6.5,
            )
            if r == 0:
                ax.text(
                    0.5,
                    1.36,
                    f"Query DI={di:.2f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
            if c == 0:
                ax.text(
                    -0.08,
                    0.5,
                    f"Top-{r+1}",
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=9,
                    rotation=90,
                )

    fig.suptitle(
        f"step {step} · val {val_exp} · Top-{top_k} train "
        r"(score=$W_{DI}(1-|dDI|)+W_{GRU}$cos, filter $|dDI|\leq 0.1$, inpainted)",
        fontsize=10,
        y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[z0_new] Saved Top-{top_k} gallery panel: {out_path}", flush=True)


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
    print(
        f"[z0_new] device={device} | W_DI={cli.w_di} W_GRU={cli.w_gru} "
        f"|dDI|<={cli.di_max_diff} | inpaint=OpenCV TELEA r={cli.inpaint_radius}",
        flush=True,
    )

    extractor_args = _load_args_from_result_dir(args.extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(**vars(extractor_args)).to(device)
    extractor_ckpt = _resolve_required_match(
        os.path.join(args.extractor_path, "**/best.ckpt"), "extractor checkpoint"
    )
    extractor.load_state_dict(
        torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"]
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False

    print("[z0_new] Building train gallery...", flush=True)
    train_ds, train_dl = _build_dataloader(
        args, extractor, extractor_args, "train", args.batch_size, num_workers
    )
    set_dataset_args(args, train_dl)
    # Rebuild a no-aug transform for inpainted images (same as val).
    from src.utils import build_damage_image_transform_with_loss_mask

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
    train_gallery = _collect_gallery_with_filename(train_ds)
    print(f"[z0_new] train gallery={len(train_gallery)}", flush=True)

    print("[z0_new] Building val set...", flush=True)
    val_ds, _val_dl = _build_dataloader(
        args, extractor, extractor_args, "val", args.batch_size, num_workers
    )
    val_gallery = _collect_gallery_with_filename(val_ds)

    model = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(
        regressor=None,
        **vars(args),
    ).to(device)
    step = _load_ckpt(model, ckpt, device)
    model.eval()

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

    image_root = os.path.join("./Data", "DamagePattern", "images")
    mask_dir = os.path.join("./Data", "DamagePattern", "mask")
    save_dir = os.path.join(result_dir, "inference_samples")
    gt_imgs, prior_imgs, topk_per_di, inpainted_views = [], [], [], []

    for di_value, anchor in zip(di_values, val_anchors):
        print(
            f"[z0_new] DI={di_value:.2f} exp={anchor['exp']} t={anchor['timestep']} "
            f"GT_DI={anchor['gt_di']:.3f}",
            flush=True,
        )
        gt_imgs.append(anchor["real"])

        ranked = _top_k_by_score(
            train_gallery,
            anchor["hidden_state"],
            target_di=di_value,
            k=cli.top_k,
            w_di=cli.w_di,
            w_gru=cli.w_gru,
            di_max_diff=cli.di_max_diff,
        )
        if len(ranked) < cli.top_k:
            print(
                f"[z0_new]   WARNING: only {len(ranked)}/{cli.top_k} candidates "
                f"with |dDI|<={cli.di_max_diff}",
                flush=True,
            )
        topk_per_di.append(ranked)
        print(
            "[z0_new]   Top-{}: {}".format(
                cli.top_k,
                ", ".join(
                    f"{it['exp']}@t{it['timestep']}(DI={it['di']:.2f},"
                    f"score={sc:.3f},cos={cos:.3f},|dDI|={ddi:.3f})"
                    for sc, cos, ddi, it in ranked
                ),
            ),
            flush=True,
        )

        z0_list = []
        col_inpainted = []
        for _score, _cos, _ddi, t_item in ranked:
            inpainted = _prepare_inpainted_tensor(
                t_item,
                val_transform,
                image_root,
                mask_dir,
                args.image_size,
                cli.inpaint_radius,
            )
            col_inpainted.append(inpainted.detach().cpu())
            img = inpainted.unsqueeze(0).to(device)
            hidden = t_item["hidden"].unsqueeze(0).to(device)
            direction = t_item["direction"].unsqueeze(0).to(device)
            info = t_item["info"].unsqueeze(0).to(device)
            di = torch.tensor([[t_item["di"]]], device=device, dtype=info.dtype)
            c_k, gru_k = model._unpack_conditions(info, hidden, direction, di)
            z1 = model._encode_to_target_space(img)
            z0_k = reverse_ode_to_z0(model, z1, c_k, gru_feat=gru_k, num_steps=ode_steps)
            z0_list.append(z0_k[0].detach())
        # Pad empty slots for panel alignment.
        while len(col_inpainted) < cli.top_k:
            col_inpainted.append(None)
        inpainted_views.append(col_inpainted)

        if not z0_list:
            raise RuntimeError(f"No Top-k candidates for DI={di_value}")
        z0_new = _slerp_average(z0_list).unsqueeze(0).to(device)

        hidden = anchor["hidden_state"].unsqueeze(0).to(device)
        direction = anchor["direction"].unsqueeze(0).to(device)
        info = anchor["info"].unsqueeze(0).to(device)
        di = torch.tensor([[float(di_value)]], device=device, dtype=info.dtype)
        c, gru_feat = model._unpack_conditions(info, hidden, direction, di)

        z = sample_ode_guided(
            model,
            c,
            z0_new.clone(),
            gru_feat=gru_feat,
            num_steps=ode_steps,
            energy_fn=None,
            guidance_scale=0.0,
            di_tensor=di,
        )
        prior_imgs.append(_tensor_to_vis(model._decode_from_target_space(z))[0].detach().cpu())

    prior_path = os.path.join(save_dir, f"z0_new_prior_step_{step:05d}.png")
    topk_path = os.path.join(save_dir, f"z0_new_topk_train_step_{step:05d}.png")
    _save_gt_prior_panel(gt_imgs, prior_imgs, di_values, prior_path, step, cli.val_anchor_exp)
    _save_topk_panel(
        topk_per_di,
        di_values,
        topk_path,
        step,
        cli.val_anchor_exp,
        cli.top_k,
        show_inpainted=inpainted_views,
    )
    print("[z0_new] Done.", flush=True)


if __name__ == "__main__":
    main()
