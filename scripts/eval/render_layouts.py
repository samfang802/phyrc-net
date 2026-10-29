#!/usr/bin/env python3
"""Render texture for generated / given spall layouts with the layout-conditioned renderer.

Input  layout npz (src/data/layout_io.py): filename (N,), layout (N,K,S,S) coverage 0-1, method, split.
Output eval-suite samples npz (src/metrics/samples_io.py): fake (N,K,128,128) uint8, real, cond, DI, filename, meta,
rows in the split's label.csv order (the frames of the eval suite). Put it in an eval suite config as
  {"name": "...", "npz": {"test": "<this file>"}}.

Per (frame, k): layout[frame, k] → coverage_to_sdf128 → condition-mask channel 1; every other condition (g, DI,
direction, channel 0) is the dataset's, as in generate_eval_samples.py. Sample k uses torch.manual_seed(seed + k)
before the batch loop, so two layout files rendered with the same --seed/--batch_size share start noise.

  python scripts/eval/render_layouts.py --layouts L.npz --out samples.npz [--split test] [--k 4]
         [--result_dir <layout_scratch run>] [--checkpoint <ckpt>]
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

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from src.data import layout_io  # noqa: E402

_RUN = "results/DamagePatternGeneration/CFMADM/2026-10-25_14-20-09_layout_scratch"
_CKPT = _RUN + "/Opensees Based damage Patterns Generation/96lmhyv1/checkpoints/last.ckpt"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layouts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="", help="test | val; default = the npz's split")
    p.add_argument("--k", type=int, default=0, help="0 = all K of the layout file")
    p.add_argument("--result_dir", default=_RUN)
    p.add_argument("--checkpoint", default=_CKPT)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trunc_px", type=float, default=8.0)
    cli = p.parse_args()
    dev = torch.device("cuda")

    lay = layout_io.load_layout_npz(cli.layouts)
    split = cli.split or lay["split"]
    assert split in ("test", "val"), f"split? got {split!r}"
    K = cli.k or lay["layout"].shape[1]
    assert K <= lay["layout"].shape[1]
    lidx = {f: i for i, f in enumerate(lay["filename"])}

    gen = _load("scripts/eval/generate_eval_samples.py", "_gen")
    args = gen._load_args_from_result_dir(os.path.abspath(cli.result_dir))
    args.result_dir = os.path.abspath(cli.result_dir)
    os.makedirs(os.path.join(args.result_dir, "data_check"), exist_ok=True)
    args.batch_size = cli.batch_size
    args.train_virtual_repeat = 1
    assert bool(getattr(args, "layout_cond", False)), "run has no layout channel"
    ex_args = gen._load_args_from_result_dir(args.extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), ex_args.architecture)(**vars(ex_args)).to(dev)
    ck = gen._resolve_required_match(os.path.join(args.extractor_path, "**/best.ckpt"), "extractor")
    extractor.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["state_dict"])
    extractor.eval()
    transform = gen.build_damage_image_transform_with_loss_mask(
        image_size=args.image_size, use_augmentation=False, output_scale=True, normalize_mean=(0.5,) * 3,
        normalize_std=(0.5,) * 3, use_grayscale_input=args.use_grayscale_input, use_clahe=args.use_clahe,
        clahe_clip_limit=args.clahe_clip_limit, clahe_tile_grid_size=args.clahe_tile_grid_size)
    kw = dict(vars(args))
    kw["use_loss_mask"] = True
    kw["use_damage_mask_condition"] = True
    kw["use_object_loss_mask"] = False
    ds = gen.PatternLoopDataset(root="./Data", mode=split, transform=transform, extractor=extractor,
                                extractor_args=ex_args, **kw)
    dl = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=1)
    gen.set_dataset_args(args, dl)
    cfm = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(regressor=None, **vars(args)).to(dev)
    gen._eval_r2._load_cfm_checkpoint(cfm, os.path.abspath(cli.checkpoint), dev)
    cfm.eval()

    files = [str(f) for f in ds.label["filename"]]
    missing = [f for f in files if f not in lidx]
    assert not missing, f"{len(missing)} eval frames missing from the layout file, e.g. {missing[:3]}"
    n = len(files)
    # SDF for every (frame, k), once: (N,K,128,128) float32 (~130 MB at K=8)
    cov = lay["layout"][[lidx[f] for f in files], :K]
    sdf = layout_io.coverage_to_sdf128(cov, cli.trunc_px)
    print(f"[render] {lay['method']} split={split} n={n} K={K} layout_S={cov.shape[-1]} "
          f"mean mask area {(sdf > 0).mean():.3f}", flush=True)

    fake = np.zeros((n, K, 128, 128), np.uint8)
    real, cond, DI = [], [], []
    from src.metrics.dino_mmd import to_uint8_gray
    for k in range(K):
        torch.manual_seed(cli.seed + k)
        row = 0
        for batch in dl:
            real_data, hidden_state, direction, _e, _t, di, info, _vm, cmask, *_ = cfm._unpack_generation_batch(batch)
            b = real_data.shape[0]
            cmask = cmask.clone().to(dev)
            cmask[:, 1] = torch.from_numpy(sdf[row:row + b, k]).to(dev)
            with torch.no_grad():
                img = gen._eval_r2._generate_batch(cfm, info.to(dev), hidden_state.to(dev), direction.to(dev),
                                                   di.to(dev), damage_mask=cmask)
            fake[row:row + b, k] = to_uint8_gray(img)
            if k == 0:
                real.append(to_uint8_gray(real_data))
                cond.append(hidden_state.reshape(b, -1).float().cpu().numpy())
                DI.append(di.view(-1).float().cpu().numpy())
            row += b
        print(f"[render] k={k} done", flush=True)
    meta = {"renderer_checkpoint": os.path.abspath(cli.checkpoint), "layouts": os.path.abspath(cli.layouts),
            "layout_method": lay["method"], "split": split, "k": K, "seed": cli.seed, "trunc_px": cli.trunc_px,
            "sampler": "single_frame_layout"}
    from src.metrics.samples_io import save_samples_npz
    save_samples_npz(cli.out, fake, np.concatenate(real), np.concatenate(cond), np.concatenate(DI), files, meta)
    print(f"[render] wrote {cli.out} fake={fake.shape}", flush=True)


if __name__ == "__main__":
    main()
