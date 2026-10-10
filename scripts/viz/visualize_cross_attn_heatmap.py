#!/usr/bin/env python3
"""Overlay Mid / U3 / U2 cross-attention heatmaps on CFM-generated column images.

Uses PyTorch forward hooks on ``mid_cross_attn``, ``u3_cross_attn``, ``u2_cross_attn``.
Attention is reshaped to each layer's spatial grid, upsampled to image size, and
averaged over ODE steps (and attention heads / physics tokens).
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
from torch.utils.data import DataLoader

# repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.model.adm import CrossAttention2d
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


LAYER_NAMES = ("mid", "u3", "u2")


def _parse_args():
    p = argparse.ArgumentParser(description="Cross-attn heatmap overlay on CFM samples")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="results/DamagePatternGeneration/CFMADM/2026-07-26_21-24-55/checkpoint_step_014000.pt",
    )
    p.add_argument(
        "--result_dir",
        type=str,
        default="results/DamagePatternGeneration/CFMADM/2026-07-26_21-24-55",
    )
    p.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    p.add_argument("--anchor_exp", type=str, default="CTR1")
    p.add_argument("--fixed_di_values", type=str, default="0.0,0.4,0.75,1.0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ode_tail_frac",
        type=float,
        default=0.35,
        help="Average attn only over the last this fraction of ODE velocity evals",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--gpu", type=int, default=6)
    return p.parse_args()


def _parse_di(raw: str):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _load_ckpt(model, path, device):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    payload = torch.load(path, map_location=device, weights_only=False)
    status = model.load_state_dict(payload["state_dict"], strict=True)
    step = int(payload.get("global_step", 0))
    print(f"[attn_vis] loaded {path} step={step} status={status}", flush=True)
    return step


class CrossAttnCapture:
    """Enable store_attn + forward hooks on Mid/U3/U2 CrossAttention2d modules."""

    def __init__(self, velocity_net):
        self.net = velocity_net
        self.modules = {
            "mid": velocity_net.mid_cross_attn,
            "u3": velocity_net.u3_cross_attn,
            "u2": velocity_net.u2_cross_attn,
        }
        for name, mod in self.modules.items():
            if not isinstance(mod, CrossAttention2d):
                raise TypeError(
                    f"{name}_cross_attn is {type(mod).__name__}; "
                    "need condition_mode=split with CrossAttention2d"
                )
        self.buffers = {k: [] for k in LAYER_NAMES}
        self.handles = []
        self.enabled = False

    def _make_hook(self, name):
        def hook(module, _inp, _out):
            if not self.enabled:
                return
            if module.last_attn is None or module.last_hw is None:
                return
            # [B, heads, HW, L] → spatial mass: mean heads, sum tokens
            attn = module.last_attn[0].float().cpu()  # [heads, HW, L]
            spatial = attn.mean(dim=0).sum(dim=-1)  # [HW]
            h, w = module.last_hw
            self.buffers[name].append(spatial.reshape(h, w).numpy())

        return hook

    def __enter__(self):
        for name, mod in self.modules.items():
            mod.store_attn = True
            mod.last_attn = None
            self.handles.append(mod.register_forward_hook(self._make_hook(name)))
        self.clear()
        self.enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.enabled = False
        for h in self.handles:
            h.remove()
        self.handles.clear()
        for mod in self.modules.values():
            mod.store_attn = False
            mod.last_attn = None

    def clear(self):
        self.buffers = {k: [] for k in LAYER_NAMES}

    def aggregate(self, tail_frac: float = 0.35) -> dict:
        """Mean spatial maps over the last ``tail_frac`` of captured forwards."""
        out = {}
        for name, maps in self.buffers.items():
            if not maps:
                out[name] = None
                continue
            n = len(maps)
            start = max(0, n - max(1, int(round(n * float(tail_frac)))))
            stack = np.stack(maps[start:], axis=0).astype(np.float64)
            heat = stack.mean(axis=0)
            # normalize to [0,1]
            heat = heat - heat.min()
            denom = heat.max() + 1e-8
            out[name] = (heat / denom).astype(np.float32)
            out[f"{name}_n_captures"] = n
            out[f"{name}_n_used"] = n - start
        return out


def _overlay_heatmap(rgb_u8: np.ndarray, heat: np.ndarray, alpha: float = 0.45):
    """rgb_u8 HxWx3 uint8; heat 2D in [0,1] any resolution → upsample."""
    h, w = rgb_u8.shape[:2]
    t = torch.from_numpy(heat)[None, None].float()
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    heat_r = t.numpy()
    cmap = plt.get_cmap("jet")
    color = (cmap(heat_r)[..., :3] * 255.0).astype(np.float32)
    base = rgb_u8.astype(np.float32)
    blend = (1.0 - alpha) * base + alpha * color
    return np.clip(blend, 0, 255).astype(np.uint8), heat_r


def _collect_anchors(model, dataloader, split, anchor_exp, di_values):
    pool_key = f"{split}_anchor_candidates"
    model.validation_outputs = {
        "gt_DI": [],
        "pred_DI": [],
        "train_anchor_candidates": [],
        "val_anchor_candidates": [],
        pool_key: [],
    }
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            real_data, hidden_state, direction, exp, timestep, DI, info = batch[:7]
            real_vis = torch.clamp(real_data * 0.5 + 0.5, 0.0, 1.0)
            model._collect_inference_anchor_candidates(
                pool_key,
                split,
                real_vis,
                hidden_state,
                direction,
                exp,
                timestep,
                DI,
                info,
            )
            if len(model.validation_outputs[pool_key]) >= model._inference_candidate_cap():
                break
    anchors = model._select_anchors_for_di_values(
        model.validation_outputs[pool_key],
        di_values,
        split=split,
        anchor_exp=anchor_exp,
    )
    return anchors


@torch.no_grad()
def _sample_with_attn(model, anchor, di_value, capture: CrossAttnCapture, seed: int):
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    hidden_state = anchor["hidden_state"].unsqueeze(0).to(device)
    direction = anchor["direction"].unsqueeze(0).to(device)
    info = anchor["info"].unsqueeze(0).to(device)
    di = torch.tensor([[float(di_value)]], device=device, dtype=info.dtype)
    c, gru_feat = model._unpack_conditions(info, hidden_state, direction, di)

    capture.clear()
    capture.enabled = True
    sampled = model.sample_with_ode(c, gru_feat=gru_feat)
    capture.enabled = False

    image = model._decode_from_target_space(sampled)
    vis = torch.clamp(image * 0.5 + 0.5, 0.0, 1.0)[0].detach().cpu()
    rgb = (vis.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    # grayscale-trained images are 3ch duplicated; keep as RGB for overlay
    maps = capture.aggregate(tail_frac=capture.tail_frac)
    return rgb, maps, anchor


def _plot_panel(rows, di_values, step, out_path, anchor_exp):
    """rows: list of dicts with rgb, maps, di, gt_di, t"""
    n = len(rows)
    # columns: GT | Gen | Mid | U3 | U2
    fig, axes = plt.subplots(n, 5, figsize=(14.5, 2.6 * n))
    if n == 1:
        axes = np.array([axes])

    titles = ["GT", "Generated", "Mid cross-attn", "U3 cross-attn", "U2 cross-attn"]
    for r, row in enumerate(rows):
        gt = row["gt"]
        gen = row["rgb"]
        axes[r, 0].imshow(gt)
        axes[r, 1].imshow(gen)
        for j, layer in enumerate(LAYER_NAMES):
            heat = row["maps"].get(layer)
            if heat is None:
                axes[r, 2 + j].imshow(gen)
                axes[r, 2 + j].set_title(f"{layer}: missing", fontsize=8)
            else:
                overlay, _ = _overlay_heatmap(gen, heat, alpha=0.5)
                axes[r, 2 + j].imshow(overlay)
                n_used = row["maps"].get(f"{layer}_n_used", "?")
                axes[r, 2 + j].set_title(f"{layer} (avg {n_used} fwd)", fontsize=8)
        axes[r, 0].set_ylabel(
            f"DI={row['di']:.2f}\nt={row['t']} GT={row['gt_di']:.2f}",
            fontsize=8,
        )
        for c in range(5):
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=9)

    fig.suptitle(
        f"Cross-attn heatmaps · step {step} · {anchor_exp} · DI={di_values}",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[attn_vis] Saved {out_path}", flush=True)


def main():
    cli = _parse_args()
    if cli.gpu is not None and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.gpu)

    result_dir = os.path.abspath(cli.result_dir)
    ckpt = os.path.abspath(cli.checkpoint)
    out_dir = cli.output_dir.strip() or os.path.join(result_dir, "analysis", "cross_attn_heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    di_values = _parse_di(cli.fixed_di_values)
    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    args.batch_size = cli.batch_size
    args.fixed_di_values = cli.fixed_di_values
    args.inference_num_images = len(di_values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[attn_vis] device={device} split={cli.split} exp={cli.anchor_exp} "
        f"condition_mode={getattr(args, 'condition_mode', None)}",
        flush=True,
    )
    if str(getattr(args, "condition_mode", "")).lower() != "split":
        raise RuntimeError(
            f"Cross-attn heatmaps require condition_mode=split, got {args.condition_mode}"
        )

    extractor_args = _load_args_from_result_dir(args.extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(**vars(extractor_args)).to(device)
    extractor.load_state_dict(
        torch.load(
            _resolve_required_match(os.path.join(args.extractor_path, "**/best.ckpt"), "extractor"),
            map_location=device,
            weights_only=False,
        )["state_dict"]
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False

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
    kwargs = dict(vars(args))
    kwargs["use_loss_mask"] = True
    kwargs["use_damage_mask_condition"] = True
    kwargs["use_object_loss_mask"] = False
    kwargs["train_virtual_repeat"] = 1

    ds = PatternLoopDataset(
        root="./Data",
        mode=cli.split,
        transform=transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **kwargs,
    )
    dl = DataLoader(
        ds,
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=_resolve_num_workers(cli.num_workers),
    )
    set_dataset_args(args, dl)

    model = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(
        regressor=None, **vars(args)
    )
    model = model.to(device)
    model.eval()
    step = _load_ckpt(model, ckpt, device)

    print("[attn_vis] Collecting anchors...", flush=True)
    anchors = _collect_anchors(model, dl, cli.split, cli.anchor_exp, di_values)
    if not anchors:
        raise RuntimeError(f"No anchors for exp={cli.anchor_exp} split={cli.split}")

    capture = CrossAttnCapture(model.velocity_net)
    capture.tail_frac = float(cli.ode_tail_frac)

    rows = []
    with capture:
        for di_value, anchor in zip(di_values, anchors):
            print(
                f"[attn_vis] Sampling DI={di_value:.2f} t={anchor['timestep']} "
                f"GT_DI={anchor['gt_di']:.2f} ...",
                flush=True,
            )
            rgb, maps, _ = _sample_with_attn(
                model, anchor, di_value, capture, seed=cli.seed + int(di_value * 1000)
            )
            gt = (anchor["real"].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            rows.append(
                {
                    "rgb": rgb,
                    "gt": gt,
                    "maps": maps,
                    "di": float(di_value),
                    "gt_di": float(anchor["gt_di"]),
                    "t": int(anchor["timestep"]),
                }
            )
            # also save per-DI npz of raw heatmaps
            np.savez_compressed(
                os.path.join(out_dir, f"attn_maps_di_{di_value:.2f}.npz"),
                mid=maps["mid"],
                u3=maps["u3"],
                u2=maps["u2"],
                generated=rgb,
                gt=gt,
                di=di_value,
                timestep=anchor["timestep"],
            )

    out_png = os.path.join(
        out_dir, f"cross_attn_heatmap_{cli.split}_{cli.anchor_exp}_step_{step:05d}.png"
    )
    _plot_panel(rows, di_values, step, out_png, cli.anchor_exp)

    # Extra: single-layer comparison for DI≈0.75 if present
    print("[attn_vis] Done.", flush=True)


if __name__ == "__main__":
    main()
