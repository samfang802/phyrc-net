#!/usr/bin/env python3
"""Generate K samples per test condition and save them for eval_dinov3_kid.py.

Single-frame path: same loader and `_generate_batch` as eval_cfm_generated_r2_siamese.py.
Refiners (use_refiner_residual, V10 residual or v10.5 implicit): each face of each column is walked in timestep
order with `sample_refiner_sequence` (frame 0 = frozen base model, later frames from the previous state),
K trajectories side by side in the batch, as in export_di_tertile_overview_grids.py. The frames of
a trajectory are the split's labelled frames of that face. Bridge / FlowEdit can write the same npz
with `save_samples_npz` below.
--shared_noise (single-frame path): every frame of one face of one column starts from the same
initial noise for sample k. The cheap way to make a single-frame model temporally coherent; the
control for a temporal model in eval_temporal.py.

npz layout (src/metrics/samples_io.py; all row-aligned to the split's label order, shuffle=False):
  fake      uint8 (N, K, H, W)   generated, x*0.5+0.5 → round(255·x)
  real      uint8 (N, H, W)      dataset GT through the same quantization
  cond      float32 (N, D)       GRU hidden state g (condition for the joint MMD)
  DI        float32 (N,)
  filename  str (N,)
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.metrics.dino_mmd import to_uint8_gray
from src.metrics.samples_io import save_samples_npz  # noqa: F401  (re-exported for older callers)
from src.metrics.train_kid import refiner_sequence_samples
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match

_spec = importlib.util.spec_from_file_location(
    "_eval_r2", os.path.join(_ROOT, "scripts", "eval", "eval_cfm_generated_r2_siamese.py")
)
_eval_r2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_r2)


def _parse_args():
    p = argparse.ArgumentParser(description="K samples per condition → npz")
    p.add_argument("--cfm_checkpoint", required=True)
    p.add_argument("--cfm_result_dir", required=True, help="Directory containing CFM args.json")
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--split", default="test", choices=("test", "val", "train"))
    p.add_argument("--k", type=int, default=8, help="Samples per condition")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--cfm_ode_steps", type=int, default=0, help="0 = use args.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shared_noise", action="store_true", help="Same initial noise for all frames of a face (per k)")
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="ATTR=VALUE",
        help="Sampler attributes set on the model, e.g. apg_guidance_scale=1.5 autoguide_weight=0.3",
    )
    return p.parse_args()


def _apply_overrides(model, pairs):
    for kv in pairs:
        key, val = kv.split("=", 1)
        for cast in (int, float):
            try:
                val = cast(val)
                break
            except ValueError:
                continue
        setattr(model, key, val)
        print(f"[gen] {key}={val!r}", flush=True)


def _generate_refiner(cfm, dl, ds, K, seed, device):
    """Walk every (column, direction) in timestep order; K trajectories per walk."""
    return refiner_sequence_samples(cfm, dl, ds, K, seed, device)


def main():
    cli = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = os.path.abspath(cli.cfm_result_dir)

    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    os.makedirs(os.path.join(result_dir, "data_check"), exist_ok=True)  # the dataset draws hysteresis plots there
    args.batch_size = int(cli.batch_size)
    args.train_virtual_repeat = 1  # train mode repeats the split many times; we want each image once
    if int(cli.cfm_ode_steps) > 0:
        args.cfm_ode_steps = int(cli.cfm_ode_steps)

    extractor_args = _load_args_from_result_dir(args.extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(
        **vars(extractor_args)
    ).to(device)
    extractor_ckpt = _resolve_required_match(os.path.join(args.extractor_path, "**/best.ckpt"), "extractor")
    extractor.load_state_dict(torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"])
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
    dataset_kwargs["use_damage_mask_condition"] = args.architecture == "CFMADM" and bool(
        args.use_damage_mask_condition
    )
    if dataset_kwargs["use_damage_mask_condition"]:
        dataset_kwargs["use_object_loss_mask"] = False
    ds = PatternLoopDataset(
        root="./Data", mode=cli.split, transform=transform, extractor=extractor, extractor_args=extractor_args,
        **dataset_kwargs,
    )
    dl = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=_resolve_num_workers(cli.num_workers))
    set_dataset_args(args, dl)

    cfm = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(
        regressor=None, **vars(args)
    ).to(device)
    step, _ = _eval_r2._load_cfm_checkpoint(cfm, os.path.abspath(cli.cfm_checkpoint), device)
    cfm.eval()
    _apply_overrides(cfm, cli.set)
    if bool(getattr(cfm, "use_cascade_sr", False)):
        cfm._attach_cascade_stage1()
        cfm._get_cascade_stage1().to(device).eval()

    n = len(ds)
    fake, real, cond, DI = None, [], [], []
    if bool(getattr(cfm, "use_refiner_residual", False)):
        fake, real, cond, DI = _generate_refiner(cfm, dl, ds, int(cli.k), int(cli.seed), device)
    traj_key = ds.label["exp"].astype(str) + "/" + ds.label["direction"].astype(str)
    with torch.no_grad():
        for k in range(0 if fake is not None else int(cli.k)):
            torch.manual_seed(int(cli.seed) + k)
            row = 0
            if cli.shared_noise:
                shape = (int(cfm.model_in_channels), int(cfm.model_spatial_size), int(cfm.model_spatial_size))
                noise = {}
                for j, key in enumerate(sorted(set(traj_key))):
                    g = torch.Generator().manual_seed(int(cli.seed) * 1000 + k * 97 + j)
                    noise[key] = torch.randn(shape, generator=g)
            for batch in tqdm(dl, desc=f"sample k={k}"):
                real_data, hidden_state, direction, _exp, _t, di, info, _vm, condition_mask, *_ = (
                    cfm._unpack_generation_batch(batch)
                )
                if condition_mask is not None:
                    condition_mask = condition_mask.to(device)
                if cli.shared_noise:
                    if bool(getattr(cfm, "use_cascade_sr", False)):
                        raise ValueError("--shared_noise is not implemented for the cascade sampler")
                    keys = traj_key.iloc[row : row + real_data.shape[0]]
                    z0 = torch.stack([noise[key] for key in keys]).to(device)
                    c, g = cfm._unpack_conditions(info.to(device), hidden_state.to(device), direction.to(device), di.to(device))
                    dmask = _eval_r2._eval_damage_mask(cfm, c, condition_mask)
                    img = cfm._decode_from_target_space(cfm.sample_with_ode(c, z0=z0, gru_feat=g, damage_mask=dmask))
                else:
                    img = _eval_r2._generate_batch(
                        cfm, info.to(device), hidden_state.to(device), direction.to(device), di.to(device),
                        damage_mask=condition_mask,
                    )
                u8 = to_uint8_gray(img)
                if fake is None:
                    fake = np.empty((n, int(cli.k)) + u8.shape[1:], dtype=np.uint8)
                fake[row : row + len(u8), k] = u8
                if k == 0:
                    real.append(to_uint8_gray(real_data))
                    cond.append(hidden_state.reshape(hidden_state.shape[0], -1).float().cpu().numpy())
                    DI.append(di.view(-1).float().cpu().numpy())
                row += len(u8)

    meta = {
        "cfm_checkpoint": os.path.abspath(cli.cfm_checkpoint),
        "global_step": int(step),
        "split": cli.split,
        "k": int(cli.k),
        "seed": int(cli.seed),
        "ode_steps": int(getattr(cfm, "cfm_ode_steps", 0)),
        "overrides": list(cli.set),
        "shared_noise": bool(cli.shared_noise),
        "sampler": "refiner_sequence" if bool(getattr(cfm, "use_refiner_residual", False)) else "single_frame",
        "refiner_implicit": bool(getattr(cfm, "refiner_implicit", False)),
        "refiner_shared_lowfreq_noise": bool(getattr(cfm, "refiner_shared_lowfreq_noise", False)),
    }
    save_samples_npz(
        cli.out, fake, np.concatenate(real), np.concatenate(cond), np.concatenate(DI),
        ds.label["filename"].tolist(), meta,
    )
    print(f"[gen] wrote {cli.out}  fake={fake.shape}", flush=True)


if __name__ == "__main__":
    main()
