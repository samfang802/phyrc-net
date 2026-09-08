#!/usr/bin/env python3
"""ODE trajectory grids for two CFM checkpoints on the test CTR1 DI=1 sample.

Captures after Heun steps 1, 3, ..., 49 (1..50, stride 2) plus step 50.
Same z0 is used for both models.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask
from test_cfm_checkpoint import _load_periodic_checkpoint
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match


def _exp_str(exp, i=0):
    if isinstance(exp, (list, tuple)):
        return str(exp[i])
    if torch.is_tensor(exp):
        return str(exp.view(-1)[i].item())
    return str(exp)


def _pick_di1_anchor(loader, exp_id="CTR1", target_di=1.0):
    best = None
    best_d = 1e9
    for batch in loader:
        real, hidden, direction, exp, timestep, DI, info = batch[:7]
        cond_mask = batch[8] if len(batch) >= 9 else None
        for i in range(real.shape[0]):
            e = _exp_str(exp, i) if not isinstance(exp, str) else str(exp)
            if str(e) != str(exp_id):
                continue
            di = float(DI[i].view(-1)[0].item())
            d = abs(di - float(target_di))
            if d < best_d:
                best_d = d
                best = {
                    "real": real[i].detach().cpu(),
                    "hidden_state": hidden[i].detach().cpu(),
                    "direction": direction[i].detach().cpu(),
                    "info": info[i].detach().cpu(),
                    "DI": DI[i].detach().cpu(),
                    "exp": e,
                    "timestep": int(timestep[i].item()) if torch.is_tensor(timestep) else int(timestep),
                    "gt_di": di,
                    "condition_mask": None
                    if cond_mask is None
                    else cond_mask[i].detach().cpu(),
                }
    if best is None:
        raise RuntimeError(f"No test sample for exp={exp_id}")
    return best


def _build_model(result_dir, ckpt, device):
    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    model = importlib.import_module("src.model.ImageGeneration").__dict__[args.architecture](
        regressor=None, **vars(args)
    )
    step, epoch = _load_periodic_checkpoint(model, ckpt, device)
    model = model.to(device).eval()
    print(
        f"[traj] loaded {ckpt} step={step} epoch={epoch} "
        f"self_cond={model.use_self_conditioning} solver={model.cfm_solver} "
        f"patch={model.use_patch_diffusion} spatial={model.model_spatial_size}",
        flush=True,
    )
    return model, args, step


@torch.no_grad()
def _ode_snapshots(model, c, gru_feat, z0, capture, damage_mask=None):
    """Heun/Euler 0→1; return dict step_k -> vis [C,H,W] in [0,1] after k updates."""
    z = z0.clone()
    n_steps = 50
    times = [float(i) / float(n_steps) for i in range(n_steps + 1)]
    coords = None
    if model.use_patch_diffusion:
        grid_x, grid_y = model._build_full_coordinate_maps(
            model.model_spatial_size, model.model_spatial_size, device=z.device, dtype=z.dtype
        )
        coords = (
            torch.stack([grid_x, grid_y], dim=0)
            .unsqueeze(0)
            .expand(c.shape[0], -1, -1, -1)
            .contiguous()
        )
    condition_mask = damage_mask
    if condition_mask is None and model.use_damage_mask_condition:
        condition_mask = torch.ones(
            (c.shape[0], 1, z.shape[-2], z.shape[-1]), device=z.device, dtype=z.dtype
        )
    self_cond = torch.zeros_like(z) if model.use_self_conditioning else None
    out = {}
    capture_set = set(int(x) for x in capture)

    def _to_vis(state):
        img = model._decode_from_target_space(state)
        return torch.clamp(img * 0.5 + 0.5, 0.0, 1.0)[0].detach().cpu()

    for i in range(n_steps):
        t_scalar = times[i]
        t_next_scalar = times[i + 1]
        dt = t_next_scalar - t_scalar
        t = torch.full((c.shape[0],), t_scalar, device=c.device)
        v_t = model._predict_velocity(
            z, t, c, coords=coords, damage_mask=condition_mask, gru_feat=gru_feat, self_cond=self_cond
        )
        z1_hat = (
            model._estimate_clean_from_velocity(z, t, v_t).detach()
            if model.use_self_conditioning
            else None
        )
        if model.cfm_solver == "heun":
            t_next = torch.full((c.shape[0],), t_next_scalar, device=c.device)
            v_next = model._predict_velocity(
                z + dt * v_t,
                t_next,
                c,
                coords=coords,
                damage_mask=condition_mask,
                gru_feat=gru_feat,
                self_cond=self_cond,
            )
            z = z + 0.5 * dt * (v_t + v_next)
        else:
            z = z + dt * v_t
        if model.use_self_conditioning:
            self_cond = z1_hat
        k = i + 1
        if k in capture_set:
            out[k] = _to_vis(z)
    return out


def _imshow(ax, img):
    if img.ndim == 3 and img.shape[0] in (1, 3):
        arr = img.permute(1, 2, 0).numpy()
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
    else:
        arr = img.numpy() if torch.is_tensor(img) else img
    ax.imshow(arr, cmap="gray" if np.ndim(arr) == 2 else None, vmin=0, vmax=1)
    ax.axis("off")


def _save_grid(frames, steps, title, path, gt=None):
    n = len(steps)
    cols = 13
    rows = int(np.ceil(n / cols))
    extra = 1 if gt is not None else 0
    fig, axes = plt.subplots(rows + extra, cols, figsize=(cols * 1.15, (rows + extra) * 1.25))
    axes = np.atleast_2d(axes)
    if gt is not None:
        _imshow(axes[0, 0], gt)
        axes[0, 0].set_title("GT", fontsize=8)
        for j in range(1, cols):
            axes[0, j].axis("off")
    base = extra
    for i, k in enumerate(steps):
        r, c = divmod(i, cols)
        ax = axes[base + r, c]
        _imshow(ax, frames[k])
        ax.set_title(f"k={k}", fontsize=7)
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[base + r, c].axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[traj] saved {path}", flush=True)


def _save_compare(a_frames, b_frames, steps, labels, gt, path):
    n = len(steps)
    fig, axes = plt.subplots(3, n, figsize=(max(12, n * 0.95), 3.6))
    _imshow(axes[0, 0], gt)
    axes[0, 0].set_title("GT DI=1", fontsize=7)
    for j in range(1, n):
        axes[0, j].axis("off")
    for j, k in enumerate(steps):
        _imshow(axes[1, j], a_frames[k])
        axes[1, j].set_title(f"{labels[0]}\nk={k}", fontsize=6)
        _imshow(axes[2, j], b_frames[k])
        axes[2, j].set_title(f"{labels[1]}\nk={k}", fontsize=6)
    fig.suptitle("ODE trajectory  ·  test CTR1  ·  DI=1  ·  steps 1..50 stride 2 (+50)", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[traj] saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--anchor_exp", type=str, default="CTR1")
    parser.add_argument("--target_di", type=float, default=1.0)
    args_cli = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs = [
        {
            "name": "1024-dadaGNsplit",
            "ckpt": os.path.join(
                ROOT,
                "results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit",
                "Opensees Based damage Patterns Generation/chjqbp9m/checkpoints/last.ckpt",
            ),
            "result_dir": os.path.join(ROOT, "results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit"),
        },
        {
            "name": "selfcondnoDI",
            "ckpt": os.path.join(
                ROOT,
                "results/DamagePatternGeneration/CFMADM/selfcondnoDI/checkpoint_step_014000.pt",
            ),
            "result_dir": os.path.join(ROOT, "results/DamagePatternGeneration/CFMADM/selfcondnoDI"),
        },
    ]
    for s in specs:
        if not os.path.isfile(s["ckpt"]):
            raise FileNotFoundError(s["ckpt"])

    capture = list(range(1, 50, 2)) + [50]  # 1,3,...,49,50

    # Dataset from selfcond run (same split / transforms as usual 128 CFM).
    args0 = _load_args_from_result_dir(specs[1]["result_dir"])
    args0.result_dir = specs[1]["result_dir"]
    os.makedirs(os.path.join(args0.result_dir, "data_check"), exist_ok=True)
    extractor_path = args0.extractor_path
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
    for p in extractor.parameters():
        p.requires_grad = False

    val_transform = build_damage_image_transform_with_loss_mask(
        image_size=args0.image_size,
        use_augmentation=False,
        output_scale=True,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
        use_grayscale_input=args0.use_grayscale_input,
        use_clahe=args0.use_clahe,
        clahe_clip_limit=args0.clahe_clip_limit,
        clahe_tile_grid_size=args0.clahe_tile_grid_size,
    )
    dataset_kwargs = dict(vars(args0))
    dataset_kwargs["use_loss_mask"] = True
    dataset_kwargs["use_damage_mask_condition"] = bool(args0.use_damage_mask_condition)
    dataset_kwargs["use_object_loss_mask"] = False
    print("[traj] Building test dataset...", flush=True)
    test_ds = PatternLoopDataset(
        root="./Data",
        mode="test",
        transform=val_transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **dataset_kwargs,
    )
    loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=_resolve_num_workers(1))
    set_dataset_args(args0, loader)
    anchor = _pick_di1_anchor(loader, exp_id=args_cli.anchor_exp, target_di=args_cli.target_di)
    print(
        f"[traj] anchor exp={anchor['exp']} t={anchor['timestep']} GT_DI={anchor['gt_di']:.3f}",
        flush=True,
    )
    gt = torch.clamp(anchor["real"] * 0.5 + 0.5, 0, 1)

    models = []
    for s in specs:
        # condition_dim from this run's args.json after set_dataset_args on same loader
        a = _load_args_from_result_dir(s["result_dir"])
        a.result_dir = s["result_dir"]
        set_dataset_args(a, loader)
        model, _, step = _build_model(s["result_dir"], s["ckpt"], device)
        models.append((s, model, step))

    z0 = torch.randn(
        1,
        models[0][1].model_in_channels,
        models[0][1].model_spatial_size,
        models[0][1].model_spatial_size,
        device=device,
    )

    all_frames = {}
    for s, model, step in models:
        hidden = anchor["hidden_state"].unsqueeze(0).to(device)
        direction = anchor["direction"].unsqueeze(0).to(device)
        info = anchor["info"].unsqueeze(0).to(device)
        di = torch.tensor([[float(args_cli.target_di)]], device=device, dtype=info.dtype)
        c, gru_feat = model._unpack_conditions(info, hidden, direction, di, exp=anchor["exp"])
        dmg = None
        if anchor.get("condition_mask") is not None:
            dmg = anchor["condition_mask"].unsqueeze(0).to(device)
            if dmg.ndim == 3:
                dmg = dmg.unsqueeze(1)
            dmg = model._align_spatial_to_target(dmg)
        print(f"[traj] ODE snapshots for {s['name']} ...", flush=True)
        frames = _ode_snapshots(model, c, gru_feat, z0, capture, damage_mask=dmg)
        all_frames[s["name"]] = frames
        out_dir = os.path.join(s["result_dir"], "inference_samples", "ode_traj_di1")
        _save_grid(
            frames,
            capture,
            f"{s['name']}  last/14k  ·  CTR1 DI=1  ·  Heun 50  ·  k=1,3,...,49,50  ·  ckpt_step={step}",
            os.path.join(out_dir, "grid_steps1to50_stride2.png"),
            gt=gt,
        )

    # Full compare is 26 cols — too wide; also save a compact every-4 plus a full 2×13.
    out_cmp = os.path.join(specs[1]["result_dir"], "inference_samples", "ode_traj_di1")
    names = [s["name"] for s in specs]
    _save_compare(
        all_frames[names[0]],
        all_frames[names[1]],
        capture,
        ["dadaGN", "selfcond"],
        gt,
        os.path.join(out_cmp, "compare_steps1to50_stride2.png"),
    )
    compact = [k for k in capture if k == 1 or k % 4 == 1 or k == 50]
    # 1,5,9,...,49,50
    compact = [k for k in capture if (k % 4 == 1) or k == 50]
    _save_compare(
        all_frames[names[0]],
        all_frames[names[1]],
        compact,
        ["dadaGN", "selfcond"],
        gt,
        os.path.join(out_cmp, "compare_compact.png"),
    )
    print("[traj] done.", flush=True)


if __name__ == "__main__":
    main()
