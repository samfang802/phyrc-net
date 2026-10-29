#!/usr/bin/env python3
"""Dump the GRU state g (1024-d) of every labelled frame -> npz (filename, g, split).

Same extractor / dataset path as scripts/eval/generate_eval_samples.py (g = ds.hidden_state[exp][timestep]).
Used by the C2 layout generator (docs/redesign/phase1_c2.md) as an alternative physical condition.
"""
import argparse, importlib, os, sys
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_required_match


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfm_result_dir", default="results/DamagePatternGeneration/CFMADM/2026-10-25_14-20-09_layout_scratch")
    p.add_argument("--out", default="Data/DamagePattern/derived/gru_g.npz")
    p.add_argument("--tmp_dir", required=True)
    cli = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = _load_args_from_result_dir(os.path.abspath(cli.cfm_result_dir))
    args.result_dir = cli.tmp_dir
    os.makedirs(os.path.join(cli.tmp_dir, "data_check"), exist_ok=True)
    eargs = _load_args_from_result_dir(args.extractor_path)
    ext = getattr(importlib.import_module("src.model.HystereticPrediction"), eargs.architecture)(**vars(eargs)).to(device)
    ck = _resolve_required_match(os.path.join(args.extractor_path, "**/best.ckpt"), "extractor")
    ext.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["state_dict"])
    ext.eval()
    tf = build_damage_image_transform_with_loss_mask(image_size=args.image_size, use_augmentation=False, output_scale=True,
        normalize_mean=(0.5,) * 3, normalize_std=(0.5,) * 3, use_grayscale_input=args.use_grayscale_input,
        use_clahe=args.use_clahe, clahe_clip_limit=args.clahe_clip_limit, clahe_tile_grid_size=args.clahe_tile_grid_size)
    names, gs, splits = [], [], []
    for mode in ("train", "val", "test"):
        kw = dict(vars(args)); kw["use_loss_mask"] = True; kw["train_virtual_repeat"] = 1
        ds = PatternLoopDataset(root="./Data", mode=mode, transform=tf, extractor=ext, extractor_args=eargs, **kw)
        for _, r in ds.label.iterrows():
            g = ds.hidden_state[r["exp"]][int(r["timestep"]), :]
            gs.append(np.asarray(g, dtype=np.float32).reshape(-1)); names.append(r["filename"]); splits.append(mode)
        print(mode, len(ds.label), flush=True)
    os.makedirs(os.path.dirname(cli.out), exist_ok=True)
    np.savez_compressed(cli.out, filename=np.asarray(names), g=np.stack(gs), split=np.asarray(splits))
    print("wrote", cli.out, np.stack(gs).shape)


if __name__ == "__main__":
    main()
