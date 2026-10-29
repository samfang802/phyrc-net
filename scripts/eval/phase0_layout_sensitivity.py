#!/usr/bin/env python3
"""Phase-0 layout-oracle diagnostic: does the generator read the layout channel?

Same conditions (g, DI, direction) and the same start noise for every variant; only the layout input changes:
  true      the frame's own AnyLabel SDF
  zero      all-zero layout (the "dropped" value seen in training)
  maxdrift  SDF of the highest-drift frame of the same column (exp + face)
  cross     SDF of a frame of the other pillar (test: CTR1 -> R307; val: R307 -> CTR1) with the same
            drift level (prefer same face)
Measures the spall area of every generated image with the phase-0 segmenter. If the variants give the
same areas / near-identical pixels, the model ignores the layout.

  python scripts/eval/phase0_layout_sensitivity.py --result_dir <run> --checkpoint <ckpt> --split test --out x.json
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load("scripts/eval/generate_eval_samples.py", "_gen")
seg = _load("scripts/eval/phase0_layout_oracle_spallseg.py", "_seg")
from src.metrics.dino_mmd import to_uint8_gray  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--segmenter", default="results/phase0_layout_oracle/spallseg/best.pt")
    p.add_argument("--layout_cache", default="Data/DamagePattern/anylabel_layout_sdf128.npz")
    p.add_argument("--out", required=True)
    cli = p.parse_args()
    dev = torch.device("cuda")

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
    ds = gen.PatternLoopDataset(root="./Data", mode=cli.split, transform=transform, extractor=extractor,
                                extractor_args=ex_args, **kw)
    dl = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=1)
    gen.set_dataset_args(args, dl)
    cfm = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(regressor=None, **vars(args)).to(dev)
    gen._eval_r2._load_cfm_checkpoint(cfm, os.path.abspath(cli.checkpoint), dev)
    cfm.eval()

    lay = np.load(cli.layout_cache)
    sdf_all = np.asarray(lay["sdf"], dtype=np.float32)
    mask_all = np.asarray(lay["mask"]).astype(bool)
    names = [str(f) for f in lay["filename"]]
    idx = {f: i for i, f in enumerate(names)}
    parse = lambda f: os.path.basename(f)[:-4].split("_")  # exp, face, drift, idx
    files = [str(f) for f in ds.label["filename"]]
    n = len(files)

    def pick_maxdrift(f):
        e, face = parse(f)[:2]
        c = [g for g in names if parse(g)[0] == e and parse(g)[1] == face]
        return max(c, key=lambda g: (int(parse(g)[2]), parse(g)[3] == parse(f)[3]))

    def pick_cross(f):
        e, face, drift, k = parse(f)
        other = "R307" if e == "CTR1" else "CTR1"
        c = [g for g in names if parse(g)[0] == other and parse(g)[2] == drift]
        if not c:
            return None
        return sorted(c, key=lambda g: (parse(g)[1] != face, parse(g)[3] != k, g))[0]

    src = {
        "true": files,
        "zero": [None] * n,
        "maxdrift": [pick_maxdrift(f) for f in files],
        "cross": [pick_cross(f) for f in files],
    }
    variants = list(src)
    fake = {v: np.zeros((n, cli.k, 128, 128), np.uint8) for v in variants}
    for v in variants:
        sel = src[v]
        for k in range(cli.k):
            torch.manual_seed(cli.seed + k)
            row = 0
            for batch in dl:
                real_data, hidden_state, direction, _e, _t, di, info, _vm, cmask, *_ = cfm._unpack_generation_batch(batch)
                b = real_data.shape[0]
                cmask = cmask.clone().to(dev)
                for j in range(b):
                    g = sel[row + j]
                    cmask[j, 1] = 0 if g is None else torch.from_numpy(sdf_all[idx[g]]).to(dev)
                with torch.no_grad():
                    img = gen._eval_r2._generate_batch(cfm, info.to(dev), hidden_state.to(dev), direction.to(dev), di.to(dev), damage_mask=cmask)
                fake[v][row:row + b, k] = to_uint8_gray(img)
                row += b
            print(f"[sens] {v} k={k} done", flush=True)

    net, thr = seg.load_segmenter(cli.segmenter, dev)
    masks = {v: seg.predict_masks(net, fake[v].reshape(-1, 128, 128), dev, thr).reshape(n, cli.k, 128, 128) for v in variants}
    gt = np.stack([mask_all[idx[f]] for f in files])
    di = ds.label["DI"].to_numpy(dtype=float)
    hi = di >= np.quantile(di, 2 / 3)
    drift = np.array([parse(f)[2] for f in files])
    area = {v: masks[v].reshape(n, cli.k, -1).mean(-1).mean(1) for v in variants}
    gt_area = gt.reshape(n, -1).mean(1)

    def iou(a, b):
        inter = (a & b).reshape(len(a), -1).sum(1)
        uni = (a | b).reshape(len(a), -1).sum(1)
        return np.where(uni > 0, inter / np.maximum(uni, 1), np.nan)

    out = {"split": cli.split, "n": int(n), "k": cli.k, "checkpoint": cli.checkpoint, "variants": {}}
    for v in variants:
        ok = np.ones(n, bool) if v == "zero" else np.array([g is not None for g in src[v]])
        d = {}
        for tag, m in (("all", np.ones(n, bool)), ("highDI", hi)):
            m = m & ok
            pix = np.abs(fake[v].astype(np.float32) - fake["true"].astype(np.float32)).mean((1, 2, 3))
            d[tag] = {
                "n": int(m.sum()),
                "gt_area": float(gt_area[m].mean()),
                "gen_area": float(area[v][m].mean()),
                "gen_area_minus_true_variant": float((area[v] - area["true"])[m].mean()),
                "mean_abs_pixel_diff_vs_true": float(pix[m].mean()),
                "iou_with_own_gt_layout": float(np.nanmean(iou(masks[v][:, 0], gt)[m])),
            }
            if v in ("maxdrift", "cross"):
                tgt = np.stack([mask_all[idx[g]] if g else np.zeros((128, 128), bool) for g in src[v]])
                d[tag]["iou_with_fed_layout"] = float(np.nanmean(iou(masks[v][:, 0], tgt)[m]))
                d[tag]["fed_layout_area"] = float(tgt.reshape(n, -1).mean(1)[m].mean())
        d["by_drift"] = {dr: {"gt_area": float(gt_area[drift == dr].mean()), "gen_area": float(area[v][drift == dr].mean())}
                         for dr in sorted(set(drift))}
        out["variants"][v] = d
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    json.dump(out, open(cli.out, "w"), indent=1)
    np.savez_compressed(cli.out.replace(".json", ".npz"), filename=np.array(files), **{f"fake_{v}": fake[v][:, :2] for v in variants})
    print(json.dumps({v: out["variants"][v]["highDI"] for v in variants}, indent=1))


if __name__ == "__main__":
    main()
