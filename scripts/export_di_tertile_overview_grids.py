#!/usr/bin/env python3
"""Build 3×(6×6) DI-tertile CFM inference overview panels (no per-cell labels)."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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


def _collect_candidates(model, dataloader, split):
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
            real_data, hidden_state, direction, exp, timestep, DI, info = batch[:7]
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
                        "sample_uid": (
                            f"{split}_{anchor_exp}_{anchor_timestep}_"
                            f"{sample_idx}_{len(pool)}"
                        ),
                    }
                )
    return model.validation_outputs[pool_key]


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


def _to_vis_numpy(img_t):
    if img_t.ndim == 3 and img_t.shape[0] in (1, 3):
        vis = img_t.permute(1, 2, 0).numpy()
        if vis.shape[-1] == 1:
            vis = vis[..., 0]
        return vis
    return img_t.numpy()


def _save_overview(images, out_path, title, grid_size):
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
        ax.imshow(vis, cmap="gray" if vis.ndim == 2 else None)
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

    selfcond = bool(getattr(args, "use_self_conditioning", False))
    selfcond_tag = "selfcond=ON" if selfcond else "selfcond=OFF"
    run_tag = cli.run_tag.strip() or os.path.basename(result_dir.rstrip("/"))
    mode = str(cli.ref_guidance_mode).lower()
    mode_tag = f" · {mode}" if mode not in ("none", "") else ""

    print(f"[tertile] ckpt={checkpoint_path}", flush=True)
    print(f"[tertile] {run_tag} · {selfcond_tag}{mode_tag}", flush=True)

    extractor_path = args.extractor_path
    extractor_args = _load_args_from_result_dir(extractor_path)
    extractor = importlib.import_module("src.model.HystereticPrediction").__dict__[
        extractor_args.architecture
    ](**vars(extractor_args)).to(device)
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

    model = importlib.import_module("src.model.ImageGeneration").__dict__[args.architecture](
        regressor=None,
        **vars(args),
    ).to(device)
    step, epoch = _load_cfm(model, checkpoint_path, device)
    model.eval()

    candidates = _collect_candidates(model, dl, cli.split)
    required_exp = str(cli.anchor_exp or "").strip()
    pool = [c for c in candidates if (not required_exp or str(c["exp"]) == required_exp)]
    if len(pool) < n_per:
        raise RuntimeError(f"pool too small: {len(pool)} < {n_per}")

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

    # Shared structural z0 from undamaged DI=0 via ODE inversion (optional).
    z0_shared = None
    if mode == "inversion":
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
        pairs = _select_uniform(
            avail,
            di_lo,
            di_hi - 1e-12 if g_idx < n_groups - 1 else di_hi,
            n_per,
        )
        for _, a in pairs:
            used_uids.add(a["sample_uid"])

        images = []
        rows = []
        with torch.no_grad():
            for target_di, anchor in tqdm(pairs, desc=f"ode-{g_name}", leave=True):
                item = model._run_inference_for_anchor(
                    anchor,
                    target_di,
                    z0=z0_shared,
                    guidance_weight=0.0 if mode == "inversion" else float(cli.ref_guidance_weight),
                )
                images.append(item["sampled_vis"])
                rows.append(
                    {
                        "target_di": float(target_di),
                        "gt_di": float(anchor["gt_di"]),
                        "exp": str(anchor["exp"]),
                        "timestep": int(anchor["timestep"]),
                    }
                )

        mode_suffix = ""
        if mode == "inversion":
            mode_suffix = "_odeinv"
        elif mode in ("latent", "velocity") and float(cli.ref_guidance_weight) > 0:
            mode_suffix = (
                f"_refg{float(cli.ref_guidance_weight):.2f}".replace(".", "p")
                + f"_{mode}"
            )
        out_path = os.path.join(
            save_dir,
            f"overview_di_{g_idx+1}_{g_name.lower()}_6x6_"
            f"{'selfcondON' if selfcond else 'selfcondOFF'}_"
            f"step_{step:05d}_ode{ode}{mode_suffix}.png",
        )
        title = (
            f"{run_tag} · {selfcond_tag}{mode_tag} · {g_name} DI "
            f"[{di_lo:.2f}, {min(di_hi, float(gt_all.max())):.2f}] · "
            f"{cli.split} · step {step} · ODE={ode} · {grid_size}×{grid_size}"
        )
        _save_overview(images, out_path, title, grid_size)
        meta_all["groups"].append(
            {
                "name": g_name,
                "di_lo": di_lo,
                "di_hi": float(min(di_hi, float(gt_all.max()))),
                "figure": out_path,
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
