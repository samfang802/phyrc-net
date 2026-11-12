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
import time

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
    p.add_argument("--out", required=True,
                   help="Output .npz path; with several splits / variants it must contain {split} / {name}")
    p.add_argument("--split", default="test",
                   help="test | val | train, or a comma list run in one process (e.g. test,val)")
    p.add_argument("--variant", action="append", default=[], metavar="NAME:ATTR=VALUE[,ATTR=VALUE...]",
                   help="Sampler settings for one npz each, on top of --set (repeatable; NAME fills {name} in --out). "
                        "Every variant of a split reuses the loaded dataset and model.")
    p.add_argument("--skip_existing", action="store_true", help="Skip a (split, variant) whose --out already exists")
    p.add_argument("--k", type=int, default=8, help="Samples per condition")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--cfm_ode_steps", type=int, default=0, help="0 = use args.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shared_noise", action="store_true", help="Same initial noise for all frames of a face (per k)")
    p.add_argument("--layouts", default="", help="layout npz (layout_cond models): sample k of each frame gets layout k %% L (L = 1: same for all k); may contain {split}")
    p.add_argument("--layout_source", default="npz_or_zero",
                   help="layout_cond models: npz_or_zero = --layouts, else 0 (default); gt = the dataset's ground-truth "
                        "AnyLabel layout (oracle only, single-frame)")
    p.add_argument(
        "--app_code",
        default="null",
        help="Appearance-code models only (docs/redesign/appearance_code.md): null | mean | seq:<i> (a training "
        "sequence's code) | random (one training sequence per face of the column, drawn per k)",
    )
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="ATTR=VALUE",
        help="Sampler attributes set on the model, e.g. apg_guidance_scale=1.5 autoguide_weight=0.3",
    )
    return p.parse_args()


_MISSING = object()


def _parse_variant(spec):
    name, _, rest = spec.partition(":")
    if not name:
        raise ValueError(f"--variant {spec!r}: expected NAME:ATTR=VALUE[,ATTR=VALUE...]")
    return name, [kv for kv in rest.split(",") if kv]


def _restore(model, saved):
    for key, val in saved.items():
        if val is _MISSING:
            delattr(model, key)
        else:
            setattr(model, key, val)


def _apply_overrides(model, pairs):
    """Set ATTR=VALUE pairs on the model; returns the previous values for _restore."""
    saved = {kv.split("=", 1)[0]: getattr(model, kv.split("=", 1)[0], _MISSING) for kv in pairs}
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
    return saved


def _app_code_column(mode, keys, n_seq, k, seed, key_ids):
    """Value of the trailing sequence-index column for a batch (-1 null, -2 mean, i = training sequence i)."""
    n = len(keys)
    if mode == "null":
        return np.full(n, -1.0, dtype=np.float32)
    if mode == "mean":
        return np.full(n, -2.0, dtype=np.float32)
    if mode.startswith("seq:"):
        i = int(mode.split(":", 1)[1])
        if not 0 <= i < n_seq:
            raise ValueError(f"app_code {mode}: only {n_seq} training sequences")
        return np.full(n, float(i), dtype=np.float32)
    if mode == "random":
        out = np.empty(n, dtype=np.float32)
        for r, key in enumerate(keys):
            out[r] = np.random.default_rng(int(seed) * 7919 + k * 131 + key_ids[key]).integers(n_seq)
        return out
    raise ValueError(f"unknown --app_code {mode!r}")


def _generate_refiner(cfm, dl, ds, K, seed, device, layouts=None):
    """Walk every (column, direction) in timestep order; K trajectories per walk."""
    return refiner_sequence_samples(cfm, dl, ds, K, seed, device, layouts=layouts)


def _load_layouts(path, split):
    """--layouts npz (src/data/layout_io.py) -> filename -> (K, S, S) coverage."""
    from src.data import layout_io

    d = layout_io.load_layout_npz(path)
    if d["split"] and d["split"] != split:
        raise ValueError(f"--layouts {path} is for split {d['split']!r}, not {split!r}")
    return {f: d["layout"][i] for i, f in enumerate(d["filename"])}


def _build_split(cli, args, extractor, extractor_args, split, device):
    """Dataset + loader of one split (the dataset draws hysteresis plots into result_dir/data_check)."""
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
    if int(getattr(args, "app_code_dim", 0)) > 0 and bool(getattr(args, "use_refiner_residual", False)):
        # The refiner walk reads the code from the dataset; only null / mean are supported there.
        if cli.app_code not in ("null", "mean"):
            raise ValueError(f"--app_code {cli.app_code} is not supported for refiner models (null | mean)")
        dataset_kwargs["app_unseen_code"] = -1 if cli.app_code == "null" else -2
    if split == "train":
        # Sampling on training frames: no training-time randomness in the conditions or the frame list.
        dataset_kwargs.update(condition_perturb=False, train_virtual_repeat=1, g_traj_mix=False,
                              share_photometric_aug=False, opa_lambda_ord=0.0, opa_lambda_mag=0.0)
    ds = PatternLoopDataset(
        root="./Data", mode=split, transform=transform, extractor=extractor, extractor_args=extractor_args,
        **dataset_kwargs,
    )
    dl = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=_resolve_num_workers(cli.num_workers))
    set_dataset_args(args, dl)
    return ds, dl


def _build_model(cli, args, device):
    cfm = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(
        regressor=None, **vars(args)
    ).to(device)
    step, _ = _eval_r2._load_cfm_checkpoint(cfm, os.path.abspath(cli.cfm_checkpoint), device)
    cfm.eval()
    _apply_overrides(cfm, cli.set)
    if bool(getattr(cfm, "use_cascade_sr", False)):
        cfm._attach_cascade_stage1()
        cfm._get_cascade_stage1().to(device).eval()
    return cfm, step


def _generate(cli, cfm, ds, dl, args, layouts, device):
    """K samples per row of the split -> fake (N, K, H, W), real / cond / DI batch lists."""
    n = len(ds)
    app_dim = int(getattr(args, "app_code_dim", 0))
    app_n_seq = int(getattr(args, "app_num_seq", 8))
    if app_dim > 0:
        print(f"[gen] appearance code: {cli.app_code} (sequences {getattr(ds, 'app_seq_index', None)})", flush=True)
    fake, real, cond, DI = None, [], [], []
    layout_cond = bool(getattr(cfm, "layout_cond", False))
    if cli.layouts and not layout_cond:
        raise ValueError("--layouts given but the model has no layout channel")
    if cli.layout_source not in ("npz_or_zero", "gt"):
        raise ValueError(f"--layout_source must be npz_or_zero or gt, got {cli.layout_source!r}")
    files = [str(f) for f in ds.label["filename"]]
    if layout_cond:
        # The val/test dataset appends the frame's GROUND-TRUTH AnyLabel layout to the condition mask.
        # Only --layout_source gt (an oracle run, named as such) may feed it to the model.
        print(f"[gen] layout input: {'GT oracle' if cli.layout_source == 'gt' else (cli.layouts or 'zero (dropped)')}", flush=True)
    if bool(getattr(cfm, "use_refiner_residual", False)):
        if layout_cond and cli.layout_source == "gt":
            raise ValueError("--layout_source gt is not implemented for the refiner walk; pass --layouts")
        fake, real, cond, DI = _generate_refiner(cfm, dl, ds, int(cli.k), int(cli.seed), device, layouts=layouts)
    traj_key = ds.label["exp"].astype(str) + "/" + ds.label["direction"].astype(str)
    key_ids = {key: j for j, key in enumerate(sorted(set(traj_key)))}
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
                if layout_cond and cli.layout_source != "gt":
                    if condition_mask is None or condition_mask.shape[1] < 2:
                        raise ValueError("layout_cond model but the dataset mask has no layout channel")
                    condition_mask = condition_mask.clone()
                    if layouts is None:
                        condition_mask[:, 1] = 0.0  # the dropped value; never the dataset's GT layout
                    else:
                        from src.data import layout_io

                        cov = np.stack([np.asarray(layouts[f], dtype=np.float32)[k % len(layouts[f])]
                                        for f in files[row : row + real_data.shape[0]]])
                        condition_mask[:, 1] = torch.from_numpy(
                            layout_io.coverage_to_sdf128(cov, size=condition_mask.shape[-1])).to(condition_mask)
                if app_dim > 0:
                    hidden_state = hidden_state.clone()
                    hidden_state[:, -1] = torch.from_numpy(_app_code_column(
                        cli.app_code, traj_key.iloc[row : row + real_data.shape[0]].tolist(), app_n_seq, k,
                        cli.seed, key_ids))
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
                    hs = hidden_state[:, :-1] if app_dim > 0 else hidden_state  # keep the v11 layout of cond
                    cond.append(hs.reshape(hs.shape[0], -1).float().cpu().numpy())
                    DI.append(di.view(-1).float().cpu().numpy())
                row += len(u8)

    return fake, real, cond, DI


def main():
    cli = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = os.path.abspath(cli.cfm_result_dir)
    splits = [x.strip() for x in cli.split.split(",") if x.strip()]
    bad = [x for x in splits if x not in ("test", "val", "train")]
    if not splits or bad:
        raise ValueError(f"--split must be test|val|train or a comma list of them, got {cli.split!r}")
    variants = [_parse_variant(v) for v in cli.variant] or [("", [])]
    if len(splits) > 1 and "{split}" not in cli.out:
        raise ValueError("several splits: --out must contain {split}")
    if len(variants) > 1 and "{name}" not in cli.out:
        raise ValueError("several variants: --out must contain {name}")
    if len({n for n, _ in variants}) != len(variants):
        raise ValueError("--variant names must be unique")

    args0 = _load_args_from_result_dir(result_dir)
    extractor_args = _load_args_from_result_dir(args0.extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(
        **vars(extractor_args)
    ).to(device)
    extractor_ckpt = _resolve_required_match(os.path.join(args0.extractor_path, "**/best.ckpt"), "extractor")
    extractor.load_state_dict(torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"])
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False

    for split in splits:
        outs = {name: cli.out.format(split=split, name=name) for name, _ in variants}
        todo = [(n, pairs) for n, pairs in variants if not (cli.skip_existing and os.path.isfile(outs[n]))]
        if not todo:
            print(f"[gen] {split}: every output exists, skipped", flush=True)
            continue
        args = _load_args_from_result_dir(result_dir)
        args.result_dir = result_dir
        os.makedirs(os.path.join(result_dir, "data_check"), exist_ok=True)  # the dataset draws hysteresis plots there
        args.batch_size = int(cli.batch_size)
        args.train_virtual_repeat = 1  # train mode repeats the split many times; we want each image once
        if int(cli.cfm_ode_steps) > 0:
            args.cfm_ode_steps = int(cli.cfm_ode_steps)
        ds, dl = _build_split(cli, args, extractor, extractor_args, split, device)
        cfm, step = _build_model(cli, args, device)
        layouts = _load_layouts(cli.layouts.format(split=split), split) if cli.layouts else None
        layout_cond = bool(getattr(cfm, "layout_cond", False))
        for name, pairs in todo:
            print(f"[gen] === split {split}{f', variant {name}' if name else ''} ===", flush=True)
            saved = _apply_overrides(cfm, pairs)
            t0 = time.time()
            fake, real, cond, DI = _generate(cli, cfm, ds, dl, args, layouts, device)
            app_dim = int(getattr(args, "app_code_dim", 0))
            meta = {
                "cfm_checkpoint": os.path.abspath(cli.cfm_checkpoint),
                "global_step": int(step),
                "split": split,
                "k": int(cli.k),
                "seed": int(cli.seed),
                "ode_steps": int(getattr(cfm, "cfm_ode_steps", 0)),
                "overrides": list(cli.set) + list(pairs),
                "variant": name,
                "app_code": cli.app_code if app_dim > 0 else "",
                "layout_input": ("" if not layout_cond else "gt_oracle" if cli.layout_source == "gt"
                                 else os.path.abspath(cli.layouts.format(split=split)) if cli.layouts else "zero"),
                "shared_noise": bool(cli.shared_noise),
                "sampler": "refiner_sequence" if bool(getattr(cfm, "use_refiner_residual", False)) else "single_frame",
                "refiner_implicit": bool(getattr(cfm, "refiner_implicit", False)),
                "refiner_shared_lowfreq_noise": bool(getattr(cfm, "refiner_shared_lowfreq_noise", False)),
                "refiner_joint_mode": (str(getattr(cfm, "refiner_joint_mode", ""))
                                       if bool(getattr(cfm, "refiner_neighbors", False)) else ""),
            }
            out = outs[name]
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            save_samples_npz(
                out, fake, np.concatenate(real), np.concatenate(cond), np.concatenate(DI),
                ds.label["filename"].tolist(), meta,
            )
            print(f"[gen] wrote {out}  fake={fake.shape}  ({time.time() - t0:.0f} s)", flush=True)
            _restore(cfm, saved)


if __name__ == "__main__":
    main()
