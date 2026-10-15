#!/usr/bin/env python3
"""Build 3×(6×6) DI-tertile CFM inference overview panels (no per-cell labels)."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--result_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--grid_size", type=int, default=6)
    p.add_argument("--n_groups", type=int, default=3)
    p.add_argument(
        "--only_groups",
        type=str,
        default="",
        help="Comma-separated subset of Low,Mid,High (e.g. High). Empty = all",
    )
    p.add_argument("--anchor_exp", type=str, default="")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--cfm_ode_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--gt_only",
        action="store_true",
        help="Only export the GT 6×6 for the same tertile selection (no ODE).",
    )
    p.add_argument(
        "--gt_from_meta",
        type=str,
        default="",
        help="Previous overview_di_*.json; GT cells follow that items[] order exactly.",
    )
    p.add_argument(
        "--flowedit",
        action="store_true",
        help="Sequence by FlowEdit on the single-frame G. No M.",
    )
    p.add_argument(
        "--flowedit_t_start",
        type=float,
        default=0.5,
        help="Our t where the velocity difference starts. 0 is full noise, closer to 1 is a weaker edit.",
    )
    p.add_argument(
        "--flowedit_steps",
        type=int,
        default=15,
        help="Euler steps of the velocity difference, from flowedit_t_start to 1.",
    )
    p.add_argument(
        "--bridge_corrector_ckpt",
        type=str,
        default="",
        help="Frozen single-frame checkpoint. After each bridge step, re-noise and denoise with this model.",
    )
    p.add_argument(
        "--bridge_corrector_s",
        type=float,
        default=0.35,
        help="Gaussian fraction before the single-frame corrector. 0.35 starts at t=0.65.",
    )
    p.add_argument(
        "--flowedit_refine_t",
        type=float,
        default=0.0,
        help="After the velocity difference, noise to this t and integrate v(g_t) to 1. 0 skips it.",
    )
    p.add_argument(
        "--spall_sdedit",
        action="store_true",
        help="SDEdit: start from the previous frame. M's sigmoid is the per-pixel edit strength.",
    )
    p.add_argument(
        "--spall_sdedit_floor",
        type=float,
        default=0.0,
        help="Sigmoid values at or below this become zero edit strength.",
    )
    p.add_argument(
        "--spall_mask_ckpt",
        type=str,
        default="",
        help="Load this spall_mask_head after the CFM checkpoint.",
    )
    p.add_argument(
        "--spall_guide_m",
        action="store_true",
        help="Inference: G paints only M's solid extent. The rest stays on the previous frame.",
    )
    p.add_argument(
        "--spall_panel",
        type=str,
        default="",
        choices=["", "g", "mg", "both"],
        help="g: single-frame G. mg: M then G along each column. both: write both grids.",
    )
    p.add_argument(
        "--ref_guidance_mode",
        type=str,
        default="none",
        choices=["none", "latent", "velocity", "inversion"],
        help="Structure reuse: inversion = DI=0→z0_inv then forward",
    )
    p.add_argument("--ref_guidance_weight", type=float, default=0.0)
    p.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Short name shown in overview title (default: basename of result_dir)",
    )
    p.add_argument(
        "--sampler",
        type=str,
        default="ode",
        choices=["ode", "sde"],
        help="ode = probability-flow Heun; sde = interpolant Euler–Maruyama",
    )
    p.add_argument("--sde_sigma", type=float, default=1.0)
    p.add_argument(
        "--feat_guidance_scale",
        type=float,
        default=0.0,
        help="Classifier-style λ on ∇_{x_t} L_feat (0=off). Uses Siamese embed→ridge→g.",
    )
    p.add_argument(
        "--feat_guidance_metric",
        type=str,
        default="cosine",
        choices=["cosine", "l2"],
    )
    p.add_argument(
        "--feat_guidance_t_power",
        type=float,
        default=1.0,
        help="λ(t)=scale * t^p; p=0 is constant, p=1 is late-t stronger",
    )
    p.add_argument(
        "--feat_guidance_ridge",
        type=float,
        default=10.0,
    )
    p.add_argument(
        "--siamese_checkpoint",
        type=str,
        default=(
            "results/DamageQuantification/resnet18_siamese_inpaint_nomask/"
            "DIresnet18/checkpoints/last.ckpt"
        ),
    )
    p.add_argument(
        "--siamese_result_dir",
        type=str,
        default=(
            "results/DamageQuantification/resnet18_siamese_inpaint_nomask/DIresnet18"
        ),
    )
    p.add_argument(
        "--sag_scale",
        type=float,
        default=0.0,
        help="Masked self-attn guidance γ; 0=off. v ← v + γ(v − v_mask), same c.",
    )
    p.add_argument("--sag_mask_frac", type=float, default=0.3)
    p.add_argument("--sag_t_min", type=float, default=0.2)
    p.add_argument("--sag_fill", type=str, default="zero", choices=["zero", "local_mean"])
    p.add_argument("--sag_reduce", type=str, default="mean", choices=["mean", "max"])
    p.add_argument("--sag_axis", type=str, default="key", choices=["key", "query"])
    p.add_argument(
        "--zero_patch_coords",
        action="store_true",
        help="Keep coord channels (in_conv size) but feed zeros instead of XY grids.",
    )
    p.add_argument(
        "--coord_guidance_scale",
        type=float,
        default=0.0,
        help="Coord CFG: v ← v_xy + γ(v_xy − v_0). 0=off. Do not combine with --zero_patch_coords.",
    )
    p.add_argument(
        "--coord_guidance_hipass_sigma",
        type=float,
        default=0.0,
        help="If >0, high-pass Δv (Δ−pool(Δ)) so coord CFG does not grow low-freq spall.",
    )
    p.add_argument(
        "--coord_guidance_t_min",
        type=float,
        default=0.0,
        help="Apply coord CFG only for t >= this (0=from the start; late detail uses ~0.55).",
    )
    p.add_argument(
        "--coord_guidance_t_max",
        type=float,
        default=1.0,
        help="Apply coord CFG only for t < this (1=through the end).",
    )
    p.add_argument(
        "--apg_guidance_scale",
        type=float,
        default=0.0,
        help="APG CFG scale s on (v_xy, v_0). 0=off. Do not combine with --coord_guidance_scale.",
    )
    p.add_argument("--apg_eta", type=float, default=0.0)
    p.add_argument("--apg_norm_threshold", type=float, default=15.0)
    p.add_argument("--apg_momentum", type=float, default=-0.5)
    p.add_argument(
        "--coord_g0",
        type=str,
        default="",
        choices=["", "zero", "undamaged"],
        help="Coord/high-pass guide on (v_cond, v_g0) instead of the coord on/off pair. "
        "undamaged = lowest-DI frame of the same column and direction. Needs --coord_guidance_scale.",
    )
    p.add_argument(
        "--apg_g0",
        type=str,
        default="",
        choices=["", "zero", "undamaged"],
        help="APG on (v_cond, v_g0) instead of the coord on/off pair. zero = g set to 0; "
        "undamaged = lowest-DI frame of the same column and direction. Needs --apg_guidance_scale. "
        "With --facfg_wl/--facfg_wh the band weights replace s: v = v_c + (wl-1) d_low + (wh-1) d_high "
        "on the APG-processed difference.",
    )
    p.add_argument(
        "--repel_prev_scale",
        type=float,
        default=0.0,
        help="Push v away from the previous ODE step (0=off). Not APG/CFG.",
    )
    p.add_argument(
        "--repel_prev_mode",
        type=str,
        default="x1",
        choices=["x1", "state"],
        help="x1 = away from last clean estimate; state = away from last z_t",
    )
    p.add_argument(
        "--stage1_checkpoint",
        type=str,
        default="",
        help="Frozen Stage-1 (64) ckpt for residual cascade SR. Empty = args.json path.",
    )
    p.add_argument(
        "--stage1_result_dir",
        type=str,
        default="",
        help="Stage-1 run dir with args.json (default: parent of --stage1_checkpoint).",
    )
    p.add_argument(
        "--use_frecas",
        action="store_true",
        help="Training-free 64→128 FreCaS (same 64 weights if --frecas_single_model).",
    )
    p.add_argument(
        "--frecas_single_model",
        action="store_true",
        help="Use the loaded 64 checkpoint for both FreCaS stages (no extra 128 train).",
    )
    p.add_argument("--frecas_stage1_stop_index", type=int, default=-1)
    p.add_argument("--frecas_gamma", type=float, default=2.0)
    p.add_argument("--frecas_w_h", type=float, default=2.0)
    p.add_argument("--frecas_w_c", type=float, default=0.6)
    p.add_argument(
        "--facfg_wl",
        type=float,
        default=None,
        help="FA-CFG weight on the low-frequency velocity. Requires --facfg_wh.",
    )
    p.add_argument(
        "--facfg_wh",
        type=float,
        default=None,
        help="FA-CFG weight on the high-frequency velocity.",
    )
    p.add_argument(
        "--facfg_g0",
        type=str,
        default="",
        choices=["", "undamaged"],
        help="Unconditional g. undamaged = same column and direction, lowest-DI frame.",
    )
    p.add_argument(
        "--spall_area_guide",
        type=str,
        default="",
        help="g-only 16x16 spall-area head (.pt). Guides only the low band.",
    )
    p.add_argument("--lowfreq_guide_scale", type=float, default=0.02)
    p.add_argument("--lowfreq_guide_t_min", type=float, default=0.2)
    p.add_argument("--lowfreq_guide_t_max", type=float, default=0.7)
    p.add_argument(
        "--autoguide_checkpoint",
        type=str,
        default="",
        help="Frozen weaker CFM. Same g and z_t as the main model.",
    )
    p.add_argument("--autoguide_result_dir", type=str, default="")
    p.add_argument("--autoguide_weight", type=float, default=0.5)
    p.add_argument(
        "--autoguide_eta",
        type=float,
        default=0.0,
        help="Weight on the part of (v_strong-v_bad) parallel to v_strong. 0 drops it.",
    )
    p.add_argument("--autoguide_gamma", type=float, default=4.0)
    p.add_argument("--autoguide_w_min", type=float, default=0.0)
    p.add_argument(
        "--refiner_gray_state",
        action="store_true",
        help="Refiner only: average the 3 channels of every frame before it becomes the next state.",
    )
    p.add_argument("--autoguide_wl", type=float, default=1.0, help="FDG weight on low-freq (v-v_bad)")
    p.add_argument("--autoguide_wh", type=float, default=1.0, help="FDG weight on high-freq (v-v_bad)")
    return p.parse_args()


def _strip_keys(state_dict):
    drop = [
        k
        for k in list(state_dict.keys())
        if ("u2_vss" in str(k)) or str(k).startswith("fid.")
    ]
    for k in drop:
        del state_dict[k]
    return drop


def _load_cfm(model, checkpoint_path, device):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = payload["state_dict"]
    dropped = _strip_keys(state)
    if dropped:
        print(f"[tertile] stripped {len(dropped)} unused keys", flush=True)
    status = model.load_state_dict(state, strict=False)
    missing = list(status.missing_keys)
    print(
        f"[tertile] load missing={len(missing)} unexpected={len(status.unexpected_keys)}",
        flush=True,
    )
    step = int(payload.get("global_step", 0))
    epoch = int(payload.get("epoch", 0))
    print(f"[tertile] loaded step={step} epoch={epoch}", flush=True)
    return step, epoch


def _collect_candidates(model, dataloader, split, store_interference=False):
    pool_key = f"{split}_anchor_candidates"
    model.validation_outputs = {
        "gt_DI": [],
        "pred_DI": [],
        "train_anchor_candidates": [],
        "val_anchor_candidates": [],
        "test_anchor_candidates": [],
    }
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"collect-{split}", leave=False):
            if isinstance(batch, dict):
                real_data = batch["image"]
                hidden_state = batch["hidden_state"]
                direction = batch["direction"]
                exp = batch["exp"]
                timestep = batch["timestep"]
                DI = batch["DI"]
                info = batch["info"]
            else:
                real_data, hidden_state, direction, exp, timestep, DI, info = batch[:7]
            interference = None
            if store_interference and not isinstance(batch, dict) and len(batch) >= 9:
                interference = (1.0 - batch[-1]).clamp(0.0, 1.0)
            real_vis = torch.clamp(real_data * 0.5 + 0.5, 0.0, 1.0)
            pool = model.validation_outputs[pool_key]
            for sample_idx in range(real_vis.shape[0]):
                anchor_exp = model._extract_batch_field(exp, sample_idx)
                anchor_timestep = int(model._extract_batch_field(timestep, sample_idx))
                gt_di_value = float(model._extract_batch_field(DI, sample_idx))
                pool.append(
                    {
                        "split": split,
                        "real": real_vis[sample_idx].detach().cpu(),
                        "hidden_state": hidden_state[sample_idx].detach().cpu(),
                        "direction": direction[sample_idx].detach().cpu(),
                        "info": info[sample_idx].detach().cpu(),
                        "exp": anchor_exp,
                        "timestep": anchor_timestep,
                        "gt_di": gt_di_value,
                        "interference_mask": (
                            None
                            if interference is None
                            else interference[sample_idx].detach().cpu()
                        ),
                        "sample_uid": (
                            f"{split}_{anchor_exp}_{anchor_timestep}_"
                            f"{sample_idx}_{len(pool)}"
                        ),
                    }
                )
    return model.validation_outputs[pool_key]


def _load_siamese(checkpoint_path, result_dir, device):
    args = _load_args_from_result_dir(result_dir)
    from src.model.ResNet18_Siamese_Lightning import ResNet18SiameseLightning

    model = ResNet18SiameseLightning(**vars(args))
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    status = model.load_state_dict(payload["state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.image_size = int(getattr(args, "image_size", 128) or 128)
    print(f"[tertile] Siamese load: {status}", flush=True)
    return model


def _select_uniform(pool, di_lo, di_hi, n):
    """Pick n distinct anchors with GT DI as uniform as possible in [di_lo, di_hi]."""
    in_bin = [c for c in pool if di_lo <= float(c["gt_di"]) <= di_hi]
    # Fall back to nearest overall if bin too small.
    search_pool = in_bin if len(in_bin) >= n else list(pool)
    targets = np.linspace(di_lo, di_hi, n).tolist()
    selected = []
    remaining = list(search_pool)
    for t in targets:
        cand = remaining if remaining else search_pool
        # Prefer in-bin if available among remaining.
        in_rem = [c for c in cand if di_lo <= float(c["gt_di"]) <= di_hi]
        use = in_rem if in_rem else cand
        anchor = min(
            use,
            key=lambda c: (
                abs(float(c["gt_di"]) - float(t)),
                int(c["timestep"]),
                str(c["sample_uid"]),
            ),
        )
        selected.append((float(t), anchor))
        remaining = [c for c in remaining if c["sample_uid"] != anchor["sample_uid"]]
    return selected


def _match_anchor(pool, item):
    """Find the pool sample for one inference-meta cell (exp, timestep, gt_di)."""
    exp = str(item["exp"])
    ts = int(item["timestep"])
    gt = float(item["gt_di"])
    cand = [
        c
        for c in pool
        if str(c["exp"]) == exp and int(c["timestep"]) == ts
    ]
    if not cand:
        raise RuntimeError(f"no pool sample for exp={exp} timestep={ts} gt_di={gt}")
    return min(cand, key=lambda c: abs(float(c["gt_di"]) - gt))


def _pairs_from_meta_items(pool, items):
    return [(float(it.get("target_di", it["gt_di"])), _match_anchor(pool, it)) for it in items]


def _undamaged_anchor(candidates, anchor):
    """Same column and direction, the frame with the lowest GT DI."""
    key = (str(anchor["exp"]), _direction_key(anchor["direction"]))
    matches = [
        c
        for c in candidates
        if (str(c["exp"]), _direction_key(c["direction"])) == key
    ]
    if not matches:
        raise RuntimeError(f"no frames for exp={key[0]} direction={key[1]}")
    return min(matches, key=lambda c: (float(c["gt_di"]), int(c["timestep"])))


def _direction_key(direction):
    if torch.is_tensor(direction):
        return tuple(round(float(v), 4) for v in direction.detach().float().reshape(-1).tolist())
    return (str(direction),)


def _ordered_trajectory(frames):
    by_t = {}
    for cand in frames:
        t = int(cand["timestep"])
        prev = by_t.get(t)
        if prev is None or str(cand["sample_uid"]) < str(prev["sample_uid"]):
            by_t[t] = cand
    return [by_t[t] for t in sorted(by_t)]


def _sample_mg_trajectory(model, frames):
    """One column, one direction, from the first frame through ``frames``."""
    if not torch.is_grad_enabled():
        return _sample_mg_trajectory_impl(model, frames)
    with torch.no_grad():
        return _sample_mg_trajectory_impl(model, frames)


@torch.no_grad()
def _sample_refiner_trajectory_impl(model, frames):
    """Frame 0 is the frozen base model. Later frames are S_t = clamp(S_{t-1} + residual)."""
    conds = []
    gru_feats = []
    for anchor in frames:
        hidden_state = anchor["hidden_state"].unsqueeze(0).to(model.device)
        direction = anchor["direction"].unsqueeze(0).to(model.device)
        info = anchor["info"].unsqueeze(0).to(model.device)
        di = torch.tensor([[float(anchor["gt_di"])]], device=model.device, dtype=info.dtype)
        c, gru_feat = model._unpack_conditions(
            info, hidden_state, direction, di, exp=anchor.get("exp")
        )
        conds.append(c)
        gru_feats.append(gru_feat)
    print(
        f"[tertile] refiner integrating {len(frames)} frames "
        f"t={int(frames[0]['timestep'])}..{int(frames[-1]['timestep'])}",
        flush=True,
    )
    latents = model.sample_refiner_sequence(conds, gru_feats)
    out = {}
    for anchor, z in zip(frames, latents):
        ts = int(anchor["timestep"])
        image = model._decode_from_target_space(z)
        out[ts] = torch.clamp(image * 0.5 + 0.5, 0.0, 1.0)[0].detach().cpu()
    return out


def _sample_bridge_trajectory_impl(model, frames):
    conds = []
    gru_feats = []
    for anchor in frames:
        hidden_state = anchor["hidden_state"].unsqueeze(0).to(model.device)
        direction = anchor["direction"].unsqueeze(0).to(model.device)
        info = anchor["info"].unsqueeze(0).to(model.device)
        di = torch.tensor([[float(anchor["gt_di"])]], device=model.device, dtype=info.dtype)
        c, gru_feat = model._unpack_conditions(
            info, hidden_state, direction, di, exp=anchor.get("exp")
        )
        conds.append(c)
        gru_feats.append(gru_feat)
    latents = model.sample_paired_bridge_sequence(conds, gru_feats)
    out = {}
    for anchor, z in zip(frames, latents):
        ts = int(anchor["timestep"])
        image = model._decode_from_target_space(z)
        out[ts] = torch.clamp(image * 0.5 + 0.5, 0.0, 1.0)[0].detach().cpu()
    return out, None


def _sample_mg_trajectory_impl(model, frames):
    if bool(getattr(model, "use_refiner_residual", False)):
        return _sample_refiner_trajectory_impl(model, frames), None
    if bool(getattr(model, "flowedit", False)):
        return _sample_flowedit_trajectory_impl(model, frames)
    if bool(getattr(model, "use_paired_bridge", False)):
        return _sample_bridge_trajectory_impl(model, frames)
    conds = []
    gru_feats = []
    for anchor in frames:
        hidden_state = anchor["hidden_state"].unsqueeze(0).to(model.device)
        direction = anchor["direction"].unsqueeze(0).to(model.device)
        info = anchor["info"].unsqueeze(0).to(model.device)
        di = torch.tensor([[float(anchor["gt_di"])]], device=model.device, dtype=info.dtype)
        c, gru_feat = model._unpack_conditions(
            info, hidden_state, direction, di, exp=anchor.get("exp")
        )
        conds.append(c)
        gru_feats.append(gru_feat)
    latents, mask_frames = model.sample_spall_sequence(
        conds, gru_feats, return_masks=True
    )
    out = {}
    masks = {"m_prob": {}, "m_grow": {}, "lock": {}}
    for anchor, z, vis in zip(frames, latents, mask_frames):
        ts = int(anchor["timestep"])
        image = model._decode_from_target_space(z)
        out[ts] = torch.clamp(image * 0.5 + 0.5, 0.0, 1.0)[0].detach().cpu()
        for key in masks:
            masks[key][ts] = vis[key][0].detach().cpu().clamp(0.0, 1.0)
    return out, masks


def _sample_flowedit_trajectory_impl(model, frames):
    conds = []
    gru_feats = []
    for anchor in frames:
        hidden_state = anchor["hidden_state"].unsqueeze(0).to(model.device)
        direction = anchor["direction"].unsqueeze(0).to(model.device)
        info = anchor["info"].unsqueeze(0).to(model.device)
        di = torch.tensor([[float(anchor["gt_di"])]], device=model.device, dtype=info.dtype)
        c, gru_feat = model._unpack_conditions(
            info, hidden_state, direction, di, exp=anchor.get("exp")
        )
        conds.append(c)
        gru_feats.append(gru_feat)
    latents = model.sample_flowedit_sequence(
        conds,
        gru_feats,
        num_steps=int(getattr(model, "flowedit_steps", 15)),
        t_start=float(getattr(model, "flowedit_t_start", 0.5)),
    )
    out = {}
    for anchor, z in zip(frames, latents):
        ts = int(anchor["timestep"])
        image = model._decode_from_target_space(z)
        out[ts] = torch.clamp(image * 0.5 + 0.5, 0.0, 1.0)[0].detach().cpu()
    return out, None


def _default_high_meta(result_dir):
    d = os.path.join(result_dir, "inference_samples")
    preferred = os.path.join(d, "overview_di_high_selfcondOFF_step_15000_ode30.json")
    if os.path.isfile(preferred):
        return preferred
    hits = sorted(
        f
        for f in os.listdir(d)
        if f.startswith("overview_di_high_") and f.endswith(".json")
    ) if os.path.isdir(d) else []
    if not hits:
        return ""
    return os.path.join(d, hits[0])


def _print_grid_order(pairs, grid_size, tag):
    print(f"[tertile] {tag} cell order (row-major):", flush=True)
    for i, (target_di, a) in enumerate(pairs):
        r, c = divmod(i, grid_size)
        print(
            f"  [{i:02d}] r{r}c{c} exp={a['exp']} ts={int(a['timestep'])} "
            f"gt_di={float(a['gt_di']):.4f} target_di={float(target_di):.4f}",
            flush=True,
        )


def _to_vis_numpy(img_t):
    if img_t.ndim == 3 and img_t.shape[0] in (1, 3):
        vis = img_t.permute(1, 2, 0).numpy()
        if vis.shape[-1] == 1:
            vis = vis[..., 0]
        return vis
    return img_t.numpy()


def _save_overview(images, out_path, title, grid_size, vmin=None, vmax=None):
    fig, axes = plt.subplots(
        grid_size,
        grid_size,
        figsize=(grid_size * 1.2, grid_size * 1.2 + 0.35),
    )
    axes = np.atleast_2d(axes)
    for i, img in enumerate(images):
        r, c = divmod(i, grid_size)
        ax = axes[r, c]
        vis = _to_vis_numpy(img)
        kw = {}
        if vis.ndim == 2:
            kw["cmap"] = "gray"
            if vmin is not None:
                kw["vmin"] = vmin
            if vmax is not None:
                kw["vmax"] = vmax
        ax.imshow(vis, **kw)
        ax.axis("off")
    fig.suptitle(title, fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[tertile] saved {out_path}", flush=True)


def main():
    cli = _parse_args()
    torch.manual_seed(int(cli.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cli.seed))

    checkpoint_path = os.path.abspath(cli.checkpoint)
    result_dir = os.path.abspath(cli.result_dir)
    grid_size = int(cli.grid_size)
    n_groups = int(cli.n_groups)
    n_per = grid_size * grid_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    args.batch_size = int(cli.batch_size)
    args.inference_num_images = n_per * n_groups
    if int(cli.cfm_ode_steps) > 0:
        args.cfm_ode_steps = int(cli.cfm_ode_steps)
    args.ref_guidance_mode = str(cli.ref_guidance_mode).lower()
    args.ref_guidance_weight = float(cli.ref_guidance_weight)
    use_sde = str(cli.sampler).lower() == "sde"
    sde_sigma = float(cli.sde_sigma)
    # Old zero-DI ckpts kept a dummy DI channel (scalar_mlp in=3).
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    _peek = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _sd = _peek.get("state_dict", _peek)
    _w = _sd.get("velocity_net.scalar_mlp.0.weight")
    if _w is not None and int(_w.shape[1]) == 3 and bool(getattr(args, "zero_di_condition", False)):
        args.keep_zero_di_slot = True
        print("[tertile] keep_zero_di_slot=True (ckpt scalar_mlp in=3)", flush=True)

    selfcond = bool(getattr(args, "use_self_conditioning", False))
    selfcond_tag = "selfcond=ON" if selfcond else "selfcond=OFF"
    run_tag = cli.run_tag.strip() or os.path.basename(result_dir.rstrip("/"))
    mode = str(cli.ref_guidance_mode).lower()
    mode_tag = f" · {mode}" if mode not in ("none", "") else ""

    print(f"[tertile] ckpt={checkpoint_path}", flush=True)
    print(
        f"[tertile] {run_tag} · {selfcond_tag}{mode_tag} · sampler={'sde' if use_sde else 'ode'}"
        + (f" σ={sde_sigma:g}" if use_sde else ""),
        flush=True,
    )

    extractor_path = args.extractor_path
    extractor_args = _load_args_from_result_dir(extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(**vars(extractor_args)).to(device)
    extractor_ckpt = _resolve_required_match(
        os.path.join(extractor_path, "**/best.ckpt"), "extractor checkpoint"
    )
    extractor.load_state_dict(
        torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"]
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
    dataset_kwargs = dict(vars(args))
    dataset_kwargs["use_loss_mask"] = True
    dataset_kwargs["use_damage_mask_condition"] = (
        args.architecture == "CFMADM" and bool(args.use_damage_mask_condition)
    )
    if dataset_kwargs["use_damage_mask_condition"]:
        dataset_kwargs["use_object_loss_mask"] = False
    if str(getattr(cli, "spall_area_guide", "") or "").strip():
        dataset_kwargs["use_damage_mask_condition"] = True
        dataset_kwargs["eval_real_condition_mask"] = True

    num_workers = _resolve_num_workers(cli.num_workers)
    ds = PatternLoopDataset(
        root="./Data",
        mode=cli.split,
        transform=transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **dataset_kwargs,
    )
    dl = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=num_workers)
    set_dataset_args(args, dl)

    model = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(
        regressor=None,
        **vars(args),
    ).to(device)
    step, epoch = _load_cfm(model, checkpoint_path, device)
    mask_ckpt = str(getattr(cli, "spall_mask_ckpt", "") or "").strip()
    if mask_ckpt:
        from train_mixed import _load_state_prefix

        _load_state_prefix(model.spall_mask_head, os.path.abspath(mask_ckpt), "spall_mask_head.")
    model.eval()
    corrector_ckpt = str(getattr(cli, "bridge_corrector_ckpt", "") or "").strip()
    if corrector_ckpt:
        corrector = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(
            regressor=None,
            **vars(args),
        ).to(device)
        _load_cfm(corrector, os.path.abspath(corrector_ckpt), device)
        corrector.eval()
        model._paired_bridge_corrector = corrector
        model._paired_bridge_frame0 = corrector
        model.bridge_corrector_s = float(cli.bridge_corrector_s)
        print(
            f"[tertile] bridge corrector s={model.bridge_corrector_s:g} "
            f"ckpt={os.path.abspath(corrector_ckpt)}",
            flush=True,
        )
    ag_ckpt = str(getattr(cli, "autoguide_checkpoint", "") or "").strip()
    if ag_ckpt:
        model.autoguide_checkpoint = ag_ckpt
        model.autoguide_result_dir = str(getattr(cli, "autoguide_result_dir", "") or "").strip()
        model.autoguide_weight = float(cli.autoguide_weight)
        model.autoguide_eta = float(cli.autoguide_eta)
        model.autoguide_gamma = float(cli.autoguide_gamma)
        model.autoguide_w_min = float(cli.autoguide_w_min)
        model.autoguide_wl = float(cli.autoguide_wl)
        model.autoguide_wh = float(cli.autoguide_wh)
        print(
            f"[tertile] autoguide w={model.autoguide_weight:g} eta={model.autoguide_eta:g} "
            f"gamma={model.autoguide_gamma:g} w_min={model.autoguide_w_min:g} "
            f"bad={os.path.abspath(ag_ckpt)}",
            flush=True,
        )
    if bool(getattr(cli, "flowedit", False)):
        model.flowedit = True
        model.flowedit_t_start = float(cli.flowedit_t_start)
        model.flowedit_steps = int(cli.flowedit_steps)
        model.flowedit_refine_t = float(cli.flowedit_refine_t)
        print(
            f"[tertile] FlowEdit t_start={model.flowedit_t_start:g} "
            f"steps={model.flowedit_steps} refine_t={model.flowedit_refine_t:g}",
            flush=True,
        )
    if bool(getattr(cli, "spall_sdedit", False)):
        model.spall_sdedit = True
        model.spall_sdedit_floor = float(getattr(cli, "spall_sdedit_floor", 0.0))
        print(
            f"[tertile] SDEdit from the previous frame; strength is M sigmoid "
            f"above {model.spall_sdedit_floor:g}",
            flush=True,
        )
    if bool(getattr(cli, "spall_guide_m", False)):
        model.spall_guide_m = True
        print("[tertile] G paints M's solid extent; other pixels stay on the previous frame", flush=True)
    model.use_sde_sample = bool(use_sde)
    model.cfm_sde_sigma = float(sde_sigma)
    model.sag_scale = float(cli.sag_scale)
    model.sag_mask_frac = float(cli.sag_mask_frac)
    model.sag_t_min = float(cli.sag_t_min)
    model.sag_fill = str(cli.sag_fill)
    model.sag_reduce = str(cli.sag_reduce)
    model.sag_axis = str(cli.sag_axis)
    model.zero_patch_coords = bool(cli.zero_patch_coords)
    model.coord_guidance_scale = float(cli.coord_guidance_scale)
    model.coord_guidance_hipass_sigma = float(cli.coord_guidance_hipass_sigma)
    model.coord_guidance_t_min = float(cli.coord_guidance_t_min)
    model.coord_guidance_t_max = float(cli.coord_guidance_t_max)
    model.apg_guidance_scale = float(cli.apg_guidance_scale)
    model.apg_eta = float(cli.apg_eta)
    model.apg_norm_threshold = float(cli.apg_norm_threshold)
    model.apg_momentum = float(cli.apg_momentum)
    model.apg_g0 = str(cli.apg_g0)
    model.coord_g0 = str(cli.coord_g0)
    model.repel_prev_scale = float(cli.repel_prev_scale)
    model.refiner_gray_state = bool(cli.refiner_gray_state)
    model.repel_prev_mode = str(cli.repel_prev_mode).lower()
    model.use_frecas = bool(cli.use_frecas)
    model.frecas_single_model = bool(cli.frecas_single_model)
    if model.use_frecas:
        if use_sde:
            raise ValueError("FreCaS is ODE-only; use --sampler ode")
        model.frecas_stage1_stop_index = int(cli.frecas_stage1_stop_index)
        model.frecas_gamma = float(cli.frecas_gamma)
        model.frecas_w_h = float(cli.frecas_w_h)
        model.frecas_w_c = float(cli.frecas_w_c)
        mode = "single-64" if model.frecas_single_model else "dual-ckpt"
        print(
            f"[tertile] FreCaS ON ({mode}) L={model.frecas_stage1_stop_index} "
            f"γ={model.frecas_gamma:g} w_h={model.frecas_w_h:g} w_c={model.frecas_w_c:g}",
            flush=True,
        )
        if not model.frecas_single_model:
            raise ValueError("tertile FreCaS currently supports --frecas_single_model only")
    s1_ckpt = str(cli.stage1_checkpoint or "").strip()
    if s1_ckpt:
        model.cascade_stage1_checkpoint = os.path.abspath(s1_ckpt)
        s1_dir = str(cli.stage1_result_dir or "").strip()
        model.cascade_stage1_result_dir = os.path.abspath(
            s1_dir or os.path.dirname(model.cascade_stage1_checkpoint)
        )
    if bool(getattr(model, "use_cascade_sr", False)):
        if model.use_frecas:
            raise ValueError("FreCaS cannot combine with residual cascade SR")
        model._attach_cascade_stage1()
        stage1 = model._get_cascade_stage1()
        if stage1 is None:
            raise RuntimeError("use_cascade_sr=True but Stage-1 failed to attach")
        print(
            f"[tertile] cascade SR Stage-1 attached "
            f"lr={getattr(model, 'cascade_lr_size', '?')} "
            f"residual={getattr(model, 'cascade_predict_residual', False)} "
            f"s1_size={getattr(stage1, 'image_size', getattr(stage1, 'model_spatial_size', '?'))}",
            flush=True,
        )
    if model.zero_patch_coords and (
        float(cli.coord_guidance_scale) != 0.0 or float(cli.apg_guidance_scale) != 0.0
    ):
        raise ValueError("--zero_patch_coords cannot combine with coord/APG guidance")
    if float(cli.coord_guidance_scale) != 0.0 and float(cli.apg_guidance_scale) != 0.0:
        raise ValueError("use --coord_guidance_scale or --apg_guidance_scale, not both")
    if model.zero_patch_coords:
        print("[tertile] zero_patch_coords=True (XY channels all 0)", flush=True)
    if float(cli.coord_guidance_scale) != 0.0:
        if use_sde:
            raise ValueError("coord guidance is ODE-only; use --sampler ode")
        print(
            f"[tertile] coord CFG γ={float(cli.coord_guidance_scale):g} "
            f"hipassσ={float(cli.coord_guidance_hipass_sigma):g} "
            f"t=[{float(cli.coord_guidance_t_min):g},{float(cli.coord_guidance_t_max):g}) "
            "(v_xy vs coords=0, same c)",
            flush=True,
        )
    if float(cli.apg_guidance_scale) != 0.0:
        if use_sde:
            raise ValueError("APG is ODE-only; use --sampler ode")
        print(
            f"[tertile] APG s={float(cli.apg_guidance_scale):g} eta={float(cli.apg_eta):g} "
            f"norm_thr={float(cli.apg_norm_threshold):g} mom={float(cli.apg_momentum):g} "
            + ("(v_xy vs coords=0)" if not cli.apg_g0 else f"(v_cond vs g0={cli.apg_g0})"),
            flush=True,
        )
    if cli.coord_g0 and float(cli.coord_guidance_scale) == 0.0:
        raise ValueError("--coord_g0 needs --coord_guidance_scale != 0")
    if cli.coord_g0 and cli.apg_g0:
        raise ValueError("use --coord_g0 or --apg_g0, not both")
    if cli.apg_g0 and float(cli.apg_guidance_scale) == 0.0:
        raise ValueError("--apg_g0 needs --apg_guidance_scale != 0")
    sag_scale = float(cli.sag_scale)
    if sag_scale != 0.0 and use_sde:
        raise ValueError("SAG is ODE-only; use --sampler ode")
    if sag_scale != 0.0:
        print(
            f"[tertile] SAG γ={sag_scale:g} mask_frac={cli.sag_mask_frac:g} "
            f"t_min={cli.sag_t_min:g} fill={cli.sag_fill} "
            f"reduce={cli.sag_reduce} axis={cli.sag_axis}",
            flush=True,
        )
    if float(cli.repel_prev_scale) != 0.0:
        if use_sde:
            raise ValueError("repel-prev is ODE-only; use --sampler ode")
        print(
            f"[tertile] repel_prev γ={float(cli.repel_prev_scale):g} "
            f"mode={cli.repel_prev_mode} (away from previous ODE step)",
            flush=True,
        )

    candidates = _collect_candidates(
        model,
        dl,
        cli.split,
        store_interference=bool(str(getattr(cli, "spall_area_guide", "") or "").strip()),
    )
    required_exp = str(cli.anchor_exp or "").strip()
    pool = [c for c in candidates if (not required_exp or str(c["exp"]) == required_exp)]
    if len(pool) < n_per:
        raise RuntimeError(f"pool too small: {len(pool)} < {n_per}")

    feat_scale = float(cli.feat_guidance_scale)
    if feat_scale != 0.0 and use_sde:
        raise ValueError("feat guidance is ODE-only; use --sampler ode")
    if feat_scale != 0.0:
        from src.model.exp.feat_guidance import SiameseGruFeatGuide, collect_ridge_pairs

        siamese_ckpt = os.path.abspath(cli.siamese_checkpoint)
        siamese_dir = os.path.abspath(cli.siamese_result_dir)
        siamese = _load_siamese(siamese_ckpt, siamese_dir, device)
        guide = SiameseGruFeatGuide(siamese, metric=str(cli.feat_guidance_metric))
        W = collect_ridge_pairs(
            model, guide, pool, device, ridge=float(cli.feat_guidance_ridge)
        )
        guide.W = W.to(device=device)
        model.feat_guidance_fn = guide
        model.feat_guidance_scale = feat_scale
        model.feat_guidance_t_power = float(cli.feat_guidance_t_power)
        print(
            f"[tertile] feat guidance λ={feat_scale:g} metric={cli.feat_guidance_metric} "
            f"t^{cli.feat_guidance_t_power:g} W={tuple(W.shape)} ridge={cli.feat_guidance_ridge:g} "
            f"N={len(pool)}",
            flush=True,
        )

    gt_all = np.asarray([float(c["gt_di"]) for c in pool], dtype=np.float64)
    # Equal-count DI tertiles (size-ordered groups).
    edges = np.quantile(gt_all, np.linspace(0.0, 1.0, n_groups + 1)).tolist()
    # Nudge last edge so max is included.
    edges[-1] = float(gt_all.max()) + 1e-9
    group_names = ["Low", "Mid", "High"][:n_groups]
    only = {
        s.strip().capitalize()
        for s in str(cli.only_groups).split(",")
        if s.strip()
    }
    if only:
        # Allow "high" → "High"
        only = {x[0].upper() + x[1:].lower() if x else x for x in only}
        group_names = [g for g in group_names if g in only]
        if not group_names:
            raise ValueError(f"--only_groups matched nothing; got {cli.only_groups!r}")
    print(f"[tertile] DI quantile edges={edges}", flush=True)
    print(f"[tertile] groups={group_names}", flush=True)

    meta_path = str(getattr(cli, "gt_from_meta", "") or "").strip()
    if bool(cli.gt_only) and not meta_path:
        meta_path = _default_high_meta(result_dir)
    meta_items_by_group = None
    if meta_path:
        meta_path = os.path.abspath(meta_path)
        with open(meta_path, encoding="utf-8") as f:
            prev_meta = json.load(f)
        meta_items_by_group = {
            str(g["name"]): list(g.get("items") or []) for g in prev_meta.get("groups", [])
        }
        print(f"[tertile] cell order locked to {meta_path}", flush=True)
        print(f"[tertile] meta groups={list(meta_items_by_group)}", flush=True)

    # Shared structural z0 from undamaged DI=0 via ODE inversion (optional).
    z0_shared = None
    if mode == "inversion" and not bool(cli.gt_only):
        # Use lowest-GT-DI anchors as undamaged source for inversion.
        di_targets0 = [0.0]
        # Fake a single-anchor list: lowest DI in pool.
        low_anchor = min(pool, key=lambda c: float(c["gt_di"]))
        z0_shared, _ref = model._build_inverted_z0_from_undamaged(
            [low_anchor], di_targets0
        )
        if z0_shared is None:
            print("[tertile] inversion failed; continuing with randn z0", flush=True)
        else:
            print("[tertile] using shared z0_inv for all selected tiles", flush=True)

    ode = int(args.cfm_ode_steps)
    save_dir = os.path.join(result_dir, "inference_samples")
    os.makedirs(save_dir, exist_ok=True)
    meta_all = {
        "checkpoint": checkpoint_path,
        "result_dir": result_dir,
        "run_tag": run_tag,
        "selfcond": selfcond,
        "ref_guidance_mode": mode,
        "ref_guidance_weight": float(cli.ref_guidance_weight),
        "sde_sigma": sde_sigma if use_sde else None,
        "feat_guidance_scale": feat_scale,
        "feat_guidance_metric": str(cli.feat_guidance_metric),
        "feat_guidance_t_power": float(cli.feat_guidance_t_power),
        "split": cli.split,
        "global_step": step,
        "epoch": epoch,
        "cfm_ode_steps": ode,
        "grid_size": grid_size,
        "edges": edges,
        "groups": [],
    }

    # Map name → tertile index for edges.
    name_to_idx = {"Low": 0, "Mid": 1, "High": 2}
    used_uids = set()
    for g_name in group_names:
        g_idx = name_to_idx[g_name]
        di_lo = float(edges[g_idx])
        di_hi = float(edges[g_idx + 1])
        # Exclude already used anchors across groups.
        avail = [c for c in pool if c["sample_uid"] not in used_uids]
        if meta_items_by_group is not None:
            if g_name not in meta_items_by_group:
                raise ValueError(
                    f"meta has no group {g_name!r}; have {list(meta_items_by_group)}"
                )
            pairs = _pairs_from_meta_items(pool, meta_items_by_group[g_name])
            if len(pairs) != n_per:
                print(
                    f"[tertile] warning: meta {g_name} has {len(pairs)} cells, "
                    f"grid wants {n_per}",
                    flush=True,
                )
        else:
            pairs = _select_uniform(
                avail,
                di_lo,
                di_hi - 1e-12 if g_idx < n_groups - 1 else di_hi,
                n_per,
            )
        _print_grid_order(pairs, grid_size, g_name)
        for _, a in pairs:
            used_uids.add(a["sample_uid"])

        gt_images = [anchor["real"] for _, anchor in pairs]
        exp_tag = str(cli.anchor_exp or "all").strip() or "all"
        gt_path = os.path.join(
            save_dir,
            f"overview_di_{g_idx+1}_{g_name.lower()}_{grid_size}x{grid_size}_"
            f"GT_{cli.split}_{exp_tag}.png",
        )
        gt_hi = float(min(di_hi, float(gt_all.max())))
        _save_overview(
            gt_images,
            gt_path,
            (
                f"GT · {g_name} DI [{di_lo:.2f}, {gt_hi:.2f}] · "
                f"{cli.split} {exp_tag} · {grid_size}×{grid_size}"
            ),
            grid_size,
        )
        if bool(cli.gt_only):
            meta_all["groups"].append(
                {
                    "name": g_name,
                    "di_lo": di_lo,
                    "di_hi": gt_hi,
                    "figure_gt": gt_path,
                    "items": [
                        {
                            "target_di": float(t),
                            "gt_di": float(a["gt_di"]),
                            "exp": str(a["exp"]),
                            "timestep": int(a["timestep"]),
                        }
                        for t, a in pairs
                    ],
                }
            )
            continue

        spall_panel = str(getattr(cli, "spall_panel", "") or "").strip().lower()
        want_g = spall_panel in ("", "g", "both")
        want_mg = spall_panel in ("mg", "both")
        images_g = []
        images_mg = []
        images_m_prob = []
        images_m_grow = []
        images_m_lock = []
        rows = []
        mg_cache = {}
        mg_masks = {}
        refiner_cache = {}
        use_refiner = bool(getattr(model, "use_refiner_residual", False))
        if use_refiner:
            needed = {}
            for _, anchor in pairs:
                key = (str(anchor["exp"]), _direction_key(anchor["direction"]))
                needed[key] = max(needed.get(key, 0), int(anchor["timestep"]))
            grouped = {}
            for cand in candidates:
                key = (str(cand["exp"]), _direction_key(cand["direction"]))
                if key in needed:
                    grouped.setdefault(key, []).append(cand)
            for key, tmax in needed.items():
                ordered = [
                    c for c in _ordered_trajectory(grouped.get(key, []))
                    if int(c["timestep"]) <= tmax
                ]
                if not ordered:
                    raise RuntimeError(
                        f"no refiner trajectory for exp={key[0]} direction={key[1]}"
                    )
                print(
                    f"[tertile] refiner {key[0]} dir={key[1]} frames={len(ordered)} "
                    f"t={int(ordered[0]['timestep'])}..{int(ordered[-1]['timestep'])}",
                    flush=True,
                )
                refiner_cache[key] = _sample_refiner_trajectory_impl(model, ordered)
        if want_mg:
            if not bool(getattr(model, "use_spall_growth_mask", False)) and not bool(
                getattr(model, "flowedit", False)
            ) and not bool(getattr(model, "use_paired_bridge", False)):
                raise RuntimeError(
                    "--spall_panel mg requires use_spall_growth_mask, FlowEdit, or a paired bridge"
                )
            needed = {}
            for _, anchor in pairs:
                key = (str(anchor["exp"]), _direction_key(anchor["direction"]))
                needed[key] = max(needed.get(key, 0), int(anchor["timestep"]))
            grouped = {}
            for cand in candidates:
                key = (str(cand["exp"]), _direction_key(cand["direction"]))
                if key in needed:
                    grouped.setdefault(key, []).append(cand)
            for key, tmax in needed.items():
                ordered = [
                    c for c in _ordered_trajectory(grouped.get(key, []))
                    if int(c["timestep"]) <= tmax
                ]
                if not ordered:
                    raise RuntimeError(f"no spall trajectory for exp={key[0]} direction={key[1]}")
                print(
                    f"[tertile] MG {key[0]} dir={key[1]} frames={len(ordered)} "
                    f"t={int(ordered[0]['timestep'])}..{int(ordered[-1]['timestep'])}",
                    flush=True,
                )
                mg_cache[key], mg_masks[key] = _sample_mg_trajectory(model, ordered)
        use_facfg = cli.facfg_wl is not None or cli.facfg_wh is not None or bool(cli.facfg_g0)
        if use_facfg:
            if cli.facfg_wl is None or cli.facfg_wh is None or cli.facfg_g0 != "undamaged":
                raise ValueError("FA-CFG needs --facfg_wl, --facfg_wh, and --facfg_g0 undamaged")
            if bool(getattr(model, "use_frecas", False)):
                raise ValueError("FA-CFG is separate from --use_frecas")
            model.facfg_wl = float(cli.facfg_wl)
            model.facfg_wh = float(cli.facfg_wh)
            model._facfg_logged = False
            print(
                f"[tertile] FA-CFG wl={model.facfg_wl:g} wh={model.facfg_wh:g} "
                "g0=undamaged frame of the same column and direction",
                flush=True,
            )
        guide_path = str(getattr(cli, "spall_area_guide", "") or "").strip()
        if guide_path:
            from src.model.cfm.lowfreq_guide import SpallAreaHead

            payload = torch.load(os.path.abspath(guide_path), map_location=device, weights_only=False)
            head = SpallAreaHead(
                physics_dim=int(payload.get("physics_dim", 1024)),
                grid=int(payload.get("grid", 16)),
            ).to(device)
            head.load_state_dict(payload["state_dict"])
            head.eval()
            for p in head.parameters():
                p.requires_grad_(False)
            model.spall_area_head = head
            model.lowfreq_guide_scale = float(cli.lowfreq_guide_scale)
            model.lowfreq_guide_t_min = float(cli.lowfreq_guide_t_min)
            model.lowfreq_guide_t_max = float(cli.lowfreq_guide_t_max)
            model._lowfreq_logged = False
            print(
                f"[tertile] lowfreq guide scale={model.lowfreq_guide_scale:g} "
                f"t=[{model.lowfreq_guide_t_min:g},{model.lowfreq_guide_t_max:g}) "
                f"head={os.path.abspath(guide_path)}",
                flush=True,
            )
        with torch.no_grad():
            for target_di, anchor in tqdm(pairs, desc=f"ode-{g_name}", leave=True):
                if use_refiner:
                    key = (str(anchor["exp"]), _direction_key(anchor["direction"]))
                    ts = int(anchor["timestep"])
                    if ts not in refiner_cache[key]:
                        raise RuntimeError(f"refiner cache missing {key} t={ts}")
                    images_g.append(refiner_cache[key][ts])
                elif want_g:
                    inter = anchor.get("interference_mask")
                    if inter is None:
                        model._lowfreq_interference = None
                    else:
                        model._lowfreq_interference = inter.unsqueeze(0).to(model.device)
                    if use_facfg or cli.apg_g0 == "undamaged" or cli.coord_g0 == "undamaged":
                        g0_anchor = _undamaged_anchor(candidates, anchor)
                        g0 = model.process_physics_condition(
                            g0_anchor["hidden_state"].unsqueeze(0).to(model.device)
                        )
                        model._facfg_gru_unc = g0
                        print(
                            f"[tertile] FA-CFG g0 exp={g0_anchor['exp']} "
                            f"t={int(g0_anchor['timestep'])} gt_di={float(g0_anchor['gt_di']):.4f} "
                            f"for target t={int(anchor['timestep'])}",
                            flush=True,
                        )
                    item = model._run_inference_for_anchor(
                        anchor,
                        target_di,
                        z0=z0_shared,
                        guidance_weight=0.0 if mode == "inversion" else float(cli.ref_guidance_weight),
                    )
                    images_g.append(item["sampled_vis"])
                if want_mg:
                    key = (str(anchor["exp"]), _direction_key(anchor["direction"]))
                    ts = int(anchor["timestep"])
                    if ts not in mg_cache[key]:
                        raise RuntimeError(f"MG cache missing {key} t={ts}")
                    images_mg.append(mg_cache[key][ts])
                    vis = None if mg_masks[key] is None else mg_masks[key]
                    if vis is not None:
                        images_m_prob.append(vis["m_prob"][ts])
                        images_m_grow.append(vis["m_grow"][ts])
                        images_m_lock.append(vis["lock"][ts])
                rows.append(
                    {
                        "target_di": float(target_di),
                        "gt_di": float(anchor["gt_di"]),
                        "exp": str(anchor["exp"]),
                        "timestep": int(anchor["timestep"]),
                    }
                )
        images = images_g if images_g else images_mg

        mode_suffix = ""
        if bool(getattr(model, "use_paired_bridge", False)):
            mode_suffix += "_bridge"
            corr_s = float(getattr(model, "bridge_corrector_s", 0.0) or 0.0)
            if getattr(model, "_paired_bridge_corrector", None) is not None and corr_s > 0.0:
                mode_suffix += f"_corr{corr_s:.2f}".replace(".", "p")
        refine_t = float(getattr(model, "flowedit_refine_t", 0.0) or 0.0)
        if bool(getattr(model, "flowedit", False)) and refine_t > 0.0:
            mode_suffix += f"_refine{refine_t:.2f}".replace(".", "p")
        if mode == "inversion":
            mode_suffix = "_odeinv"
        elif mode in ("latent", "velocity") and float(cli.ref_guidance_weight) > 0:
            mode_suffix = (
                f"_refg{float(cli.ref_guidance_weight):.2f}".replace(".", "p")
                + f"_{mode}"
            )
        if use_sde:
            mode_suffix += f"_sde{sde_sigma:.2f}".replace(".", "p")
        if feat_scale != 0.0:
            mode_suffix += (
                f"_featg{feat_scale:.2f}".replace(".", "p")
                + f"_{cli.feat_guidance_metric}"
            )
        if sag_scale != 0.0:
            mode_suffix += f"_sag{sag_scale:.2f}".replace(".", "p")
        if bool(cli.zero_patch_coords):
            mode_suffix += "_nocoord"
        if float(cli.coord_guidance_scale) != 0.0:
            mode_suffix += f"_coordg{float(cli.coord_guidance_scale):.2f}".replace(".", "p")
            if float(cli.coord_guidance_hipass_sigma) > 0.0:
                mode_suffix += f"_hp{float(cli.coord_guidance_hipass_sigma):.1f}".replace(".", "p")
            if float(cli.coord_guidance_t_min) > 0.0:
                mode_suffix += f"_tmin{float(cli.coord_guidance_t_min):.2f}".replace(".", "p")
            if float(cli.coord_guidance_t_max) < 1.0:
                mode_suffix += f"_tmax{float(cli.coord_guidance_t_max):.2f}".replace(".", "p")
            if cli.coord_g0:
                mode_suffix += f"_g0{cli.coord_g0}"
        if float(cli.apg_guidance_scale) != 0.0:
            mode_suffix += f"_apg{float(cli.apg_guidance_scale):.2f}".replace(".", "p")
            if float(cli.apg_eta) == 1.0 and float(cli.apg_norm_threshold) <= 0.0:
                mode_suffix += "_mom"
            if cli.apg_g0:
                mode_suffix += f"_g0{cli.apg_g0}"
        if float(cli.repel_prev_scale) != 0.0:
            mode_suffix += (
                f"_repel{float(cli.repel_prev_scale):.2f}".replace(".", "p")
                + f"_{cli.repel_prev_mode}"
            )
        if cli.facfg_wl is not None and cli.facfg_wh is not None:
            mode_suffix += (
                f"_facfg_wl{float(cli.facfg_wl):.2f}_wh{float(cli.facfg_wh):.2f}".replace(".", "p")
                + "_g0undamaged"
            )
        if str(getattr(cli, "spall_area_guide", "") or "").strip():
            mode_suffix += f"_lfguide{float(cli.lowfreq_guide_scale):.2f}_ifix".replace(".", "p")
        if str(getattr(cli, "autoguide_checkpoint", "") or "").strip():
            mode_suffix += f"_ag{float(cli.autoguide_weight):.2f}".replace(".", "p")
            if float(cli.autoguide_eta) != 1.0:
                mode_suffix += f"_eta{float(cli.autoguide_eta):.2f}".replace(".", "p")
            if float(cli.autoguide_gamma) != 0.0:
                mode_suffix += f"_g{float(cli.autoguide_gamma):.1f}".replace(".", "p")
            _bad_step = re.search(r"(\d+)", os.path.basename(cli.autoguide_checkpoint))
            if _bad_step:
                mode_suffix += f"_bad{int(_bad_step.group(1))}"
            if float(cli.autoguide_wl) != 1.0 or float(cli.autoguide_wh) != 1.0:
                mode_suffix += (
                    f"_fdgL{float(cli.autoguide_wl):.2f}H{float(cli.autoguide_wh):.2f}"
                ).replace(".", "p")
        if bool(getattr(model, "use_refiner_residual", False)):
            mode_suffix += "_refiner" + ("_gray" if bool(cli.refiner_gray_state) else "")
        if bool(getattr(model, "use_cascade_sr", False)):
            mode_suffix += "_cascade"
        if bool(getattr(model, "use_frecas", False)):
            mode_suffix += "_frecas"
            if bool(getattr(model, "frecas_single_model", False)):
                mode_suffix += "_64only"
        sampler_tag = f"SDE σ={sde_sigma:g}" if use_sde else f"ODE={ode}"
        if feat_scale != 0.0:
            sampler_tag += f" · featg λ={feat_scale:g}"
        if sag_scale != 0.0:
            sampler_tag += f" · sag γ={sag_scale:g}"
        if bool(cli.zero_patch_coords):
            sampler_tag += " · coords=0"
        if float(cli.coord_guidance_scale) != 0.0:
            sampler_tag += f" · coordg γ={float(cli.coord_guidance_scale):g}"
            if float(cli.coord_guidance_hipass_sigma) > 0.0:
                sampler_tag += f" hpσ={float(cli.coord_guidance_hipass_sigma):g}"
            if cli.coord_g0:
                sampler_tag += f" g0={cli.coord_g0}"
        if float(cli.apg_guidance_scale) != 0.0:
            sampler_tag += f" · APG s={float(cli.apg_guidance_scale):g}"
            if float(cli.apg_eta) == 1.0 and float(cli.apg_norm_threshold) <= 0.0:
                sampler_tag += " mom-only"
            if cli.apg_g0:
                sampler_tag += f" g0={cli.apg_g0}"
        if float(cli.repel_prev_scale) != 0.0:
            sampler_tag += (
                f" · repel γ={float(cli.repel_prev_scale):g} {cli.repel_prev_mode}"
            )
        if cli.facfg_wl is not None and cli.facfg_wh is not None:
            sampler_tag += (
                f" · FA-CFG wl={float(cli.facfg_wl):g} wh={float(cli.facfg_wh):g} g0=undamaged"
            )
        if str(getattr(cli, "spall_area_guide", "") or "").strip():
            sampler_tag += f" · LFguide s={float(cli.lowfreq_guide_scale):g}"
        if str(getattr(cli, "autoguide_checkpoint", "") or "").strip():
            sampler_tag += (
                f" · autoguide w={float(cli.autoguide_weight):g} eta={float(cli.autoguide_eta):g}"
                f" γ={float(cli.autoguide_gamma):g}"
            )
        if bool(getattr(model, "use_refiner_residual", False)):
            sampler_tag += " · residual refiner"
        if bool(getattr(model, "use_cascade_sr", False)):
            sampler_tag += (
                f" · cascade {getattr(model, 'cascade_lr_size', '?')}→"
                f"{getattr(model, 'model_spatial_size', '?')}"
            )
        if bool(getattr(model, "use_frecas", False)):
            sampler_tag += " · FreCaS 64→128"
            if bool(getattr(model, "frecas_single_model", False)):
                sampler_tag += " 64-only"
        stem = (
            f"overview_di_{g_idx+1}_{g_name.lower()}_6x6_"
            f"{'selfcondON' if selfcond else 'selfcondOFF'}_"
            f"step_{step:05d}_ode{ode}{mode_suffix}"
        )
        di_hi_show = min(di_hi, float(gt_all.max()))
        panel_saves = []
        if spall_panel == "":
            panel_saves.append(("", images, "", None, None))
        else:
            if want_g:
                panel_saves.append(("_gonly", images_g, " · G only", None, None))
            if want_mg:
                panel_saves.append(("_mg", images_mg, " · M+G", None, None))
                if images_m_prob:
                    panel_saves.append(
                        ("_mg_mprob", images_m_prob, " · M sigmoid", 0.0, 1.0)
                    )
                    panel_saves.append(
                        ("_mg_mmask", images_m_grow, " · M growth → G", 0.0, 1.0)
                    )
                    panel_saves.append(
                        ("_mg_lock", images_m_lock, " · RePaint lock", 0.0, 1.0)
                    )
        out_path = ""
        for suffix, panel_images, panel_note, vmin, vmax in panel_saves:
            out_path = os.path.join(save_dir, f"{stem}{suffix}.png")
            title = (
                f"{run_tag} · {selfcond_tag}{mode_tag}{panel_note} · {g_name} DI "
                f"[{di_lo:.2f}, {di_hi_show:.2f}] · "
                f"{cli.split} · step {step} · {sampler_tag} · {grid_size}×{grid_size}"
            )
            _save_overview(
                panel_images, out_path, title, grid_size, vmin=vmin, vmax=vmax
            )
        meta_all["groups"].append(
            {
                "name": g_name,
                "di_lo": di_lo,
                "di_hi": float(min(di_hi, float(gt_all.max()))),
                "figure": out_path,
                "figure_gt": gt_path,
                "items": rows,
            }
        )

    meta_path = os.path.join(
        save_dir,
        f"overview_di_{'_'.join(g.lower() for g in group_names)}_"
        f"{'selfcondON' if selfcond else 'selfcondOFF'}_"
        f"step_{step:05d}_ode{ode}"
        f"{'_odeinv' if mode == 'inversion' else ''}.json",
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_all, f, indent=2)
    print(f"[tertile] meta → {meta_path}", flush=True)


if __name__ == "__main__":
    main()
