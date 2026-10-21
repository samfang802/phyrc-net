#!/usr/bin/env python3
"""Remove interference from generated images with the same CFM inpainter as the reference set.

1. The interference segmenter (train_interference_detector.py) finds fixtures on every generated
   image; clean_mask keeps components ≥ min_area px that touch the bottom or a side border.
2. Flagged images are inpainted with the CFM that produced Data/DamagePattern_inpaint_selfcondnoDI
   (sample_with_ode_inpaint, OT-path replacement, hole dilated 5 px inside the sampler), using the
   hysteresis condition of the same test frame. Unflagged images pass through unchanged.

Writes an npz in the generate_eval_samples.py layout (fake replaced by the inpainted images) plus
  interference_mask  uint8 (N, K, H, W)
  interference_flag  bool  (N, K)
and, with --png_dir, one [generated | detected mask | inpainted] strip per flagged image and an
overview grid.

Example:
  python scripts/eval/inpaint_generated_samples.py --samples cgan_raw_k1.npz \
      --detector results/InterferenceDetector/v1/best.pt --out cgan_inpainted.npz --png_dir out/cgan_png
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.metrics import interference as itf
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match

DEFAULT_INPAINT_DIR = "results/DamagePatternGeneration/CFMADM/selfcondnoDI"
DEFAULT_INPAINT_CKPT = DEFAULT_INPAINT_DIR + "/Opensees Based damage Patterns Generation/vq9dirur/checkpoints/last.ckpt"


def _parse_args():
    p = argparse.ArgumentParser(description="Detect and inpaint interference on generated images")
    p.add_argument("--samples", required=True, help="npz from generate_eval_samples.py / save_samples_npz")
    p.add_argument("--detector", required=True, help="best.pt from train_interference_detector.py")
    p.add_argument("--out", required=True)
    p.add_argument("--png_dir", default="")
    p.add_argument("--split", default="test")
    p.add_argument("--cfm_checkpoint", default=DEFAULT_INPAINT_CKPT, help="Same inpainter as the reference set")
    p.add_argument("--cfm_result_dir", default=DEFAULT_INPAINT_DIR)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--min_area", type=int, default=150)
    p.add_argument("--border_px", type=int, default=3)
    p.add_argument("--dilate_px", type=int, default=2, help="Extra dilation before the sampler's own 5 px")
    p.add_argument(
        "--add_gt_mask",
        action="store_true",
        help="Also inpaint the frame's GT SAM region on every image, so every model carries the same "
        "inpainter footprint as the inpainted reference (otherwise only flagged images do)",
    )
    p.add_argument("--mask_dir", default="Data/DamagePattern/mask")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42, help="export_inpaint_dataset.py default")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _build_inpainter(cli, device):
    args = _load_args_from_result_dir(os.path.abspath(cli.cfm_result_dir))
    args.result_dir = os.path.abspath(cli.cfm_result_dir)
    os.makedirs(os.path.join(args.result_dir, "data_check"), exist_ok=True)  # the dataset draws hysteresis plots there
    args.train_virtual_repeat = 1
    args.condition_perturb = False
    args.use_wandb = False
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    payload = torch.load(cli.cfm_checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    for k in [k for k in state if "u2_vss" in k or k.startswith("fid.")]:
        del state[k]
    w = state.get("velocity_net.scalar_mlp.0.weight")
    if w is not None and int(w.shape[1]) == 3 and bool(getattr(args, "zero_di_condition", False)):
        args.keep_zero_di_slot = True  # same special case as eval_cfm_generated_r2_siamese.py

    extractor_args = _load_args_from_result_dir(args.extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(
        **vars(extractor_args)
    ).to(device)
    ck = _resolve_required_match(os.path.join(args.extractor_path, "**/best.ckpt"), "extractor")
    extractor.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["state_dict"])
    extractor.eval()

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
    kw = dict(vars(args))
    kw["use_loss_mask"] = True
    kw["use_damage_mask_condition"] = args.architecture == "CFMADM" and bool(args.use_damage_mask_condition)
    if kw["use_damage_mask_condition"]:
        kw["use_object_loss_mask"] = False
    ds = PatternLoopDataset(root="./Data", mode=cli.split, transform=transform, extractor=extractor,
                            extractor_args=extractor_args, **kw)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=_resolve_num_workers(1))
    set_dataset_args(args, dl)

    model = getattr(importlib.import_module("src.model.ImageGeneration"), args.architecture)(
        regressor=None, **vars(args)
    )
    status = model.load_state_dict(state, strict=False)
    print(f"[inpaint] CFM load missing={len(status.missing_keys)} unexpected={len(status.unexpected_keys)} "
          f"step={payload.get('global_step')}", flush=True)
    model = model.to(device).eval()

    # Conditions of every frame, in label order (= generate_eval_samples.py row order).
    cs, gs = [], []
    with torch.no_grad():
        for batch in dl:
            _real, hidden, direction, _exp, _t, DI, info, *_ = model._unpack_generation_batch(batch)
            c, g = model._unpack_conditions(info.to(device), hidden.to(device), direction.to(device), DI.to(device))
            cs.append(c.cpu())
            gs.append(None if g is None else g.cpu())
    c_all = torch.cat(cs)
    g_all = None if gs[0] is None else torch.cat(gs)
    return model, args, ds.label["filename"].tolist(), c_all, g_all


def _strip(gen, mask, out):
    g3 = np.stack([gen] * 3, -1).astype(np.float32)
    ov = g3.copy()
    ov[mask > 0] = 0.5 * ov[mask > 0] + 0.5 * np.array([255, 0, 0])
    return np.concatenate([g3, ov, np.stack([out] * 3, -1)], 1).astype(np.uint8)


def main():
    cli = _parse_args()
    device = cli.device
    z = np.load(cli.samples, allow_pickle=False)
    fake = z["fake"]
    G, K = fake.shape[:2]
    meta = json.loads(str(z["meta"])) if "meta" in z else {}

    detector, _ = itf.load_segmenter(cli.detector, device)
    prob = itf.predict_prob(detector, fake.reshape(-1, *fake.shape[2:]), device=device)
    mask = itf.clean_mask(prob, cli.thr, cli.min_area, cli.border_px).reshape(fake.shape)
    flag = mask.reshape(G, K, -1).any(-1)
    print(f"[inpaint] flagged {int(flag.sum())}/{G * K} images ({flag.mean():.3f})", flush=True)
    hole_all = mask
    if cli.add_gt_mask:
        gt = itf.load_sam_masks([str(n) for n in z["filename"]], cli.mask_dir, size=fake.shape[-1])
        hole_all = np.maximum(mask, gt[:, None])
        print("[inpaint] --add_gt_mask: every image is inpainted at its GT fixture region too", flush=True)
    todo = hole_all.reshape(G, K, -1).any(-1)
    del detector
    torch.cuda.empty_cache()

    out = fake.copy()
    if todo.any():
        model, args, names, c_all, g_all = _build_inpainter(cli, device)
        if list(names) != [str(n) for n in z["filename"]]:
            raise ValueError("npz rows are not in the dataset's label order for this split")
        steps = int(args.cfm_ode_steps)
        k_dil = 2 * int(cli.dilate_px) + 1
        for k in range(K):
            idx = np.flatnonzero(todo[:, k])
            torch.manual_seed(cli.seed + k)
            for s in tqdm(range(0, len(idx), cli.batch_size), desc=f"inpaint k={k}"):
                b = idx[s : s + cli.batch_size]
                x = torch.from_numpy(fake[b, k]).to(device).float().div(127.5).sub(1.0)[:, None].repeat(1, 3, 1, 1)
                hole = torch.from_numpy(hole_all[b, k]).to(device).float()[:, None]
                if cli.dilate_px > 0:
                    hole = F.max_pool2d(hole, kernel_size=k_dil, stride=1, padding=int(cli.dilate_px))
                with torch.no_grad():
                    z1 = model._encode_to_target_space(x)
                    keep = F.interpolate(1.0 - hole, size=z1.shape[-2:], mode="nearest")
                    zo = model.sample_with_ode_inpaint(
                        c=c_all[b].to(device), z1_gt=z1, inpaint_mask=keep, z0=None, num_steps=steps,
                        gru_feat=None if g_all is None else g_all[b].to(device),
                    )
                    dec = model._decode_from_target_space(zo)
                out[b, k] = torch.round(torch.clamp(dec[:, 0].float() * 0.5 + 0.5, 0, 1) * 255).to(torch.uint8).cpu().numpy()

    meta = {
        **meta,
        "interference_removed": True,
        "detector": os.path.abspath(cli.detector),
        "inpainter": os.path.abspath(cli.cfm_checkpoint),
        "thr": cli.thr, "min_area": cli.min_area, "border_px": cli.border_px, "dilate_px": cli.dilate_px,
        "frac_flagged": float(flag.mean()),
        "add_gt_mask": bool(cli.add_gt_mask),
    }
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    np.savez_compressed(
        cli.out, fake=out, real=z["real"], cond=z["cond"], DI=z["DI"], filename=z["filename"],
        meta=np.asarray(json.dumps(meta)), interference_mask=mask, interference_flag=flag,
    )
    print(f"[inpaint] wrote {cli.out}", flush=True)

    if cli.png_dir:
        os.makedirs(cli.png_dir, exist_ok=True)
        strips = []
        for i, k in zip(*np.nonzero(flag)):
            st = _strip(fake[i, k], mask[i, k], out[i, k])
            stem = os.path.splitext(str(z["filename"][i]))[0]
            Image.fromarray(st).save(os.path.join(cli.png_dir, f"{stem}_k{k}.png"))
            strips.append(st)
        if strips:
            pick = np.linspace(0, len(strips) - 1, min(12, len(strips))).astype(int)
            half = (len(pick) + 1) // 2
            cols = [np.concatenate([strips[j] for j in pick[:half]], 0), np.concatenate([strips[j] for j in pick[half:]], 0)]
            if cols[1].shape[0] < cols[0].shape[0]:
                cols[1] = np.pad(cols[1], ((0, cols[0].shape[0] - cols[1].shape[0]), (0, 0), (0, 0)), constant_values=255)
            Image.fromarray(np.concatenate(cols, 1)).save(os.path.join(cli.png_dir, "overview.png"))
        print(f"[inpaint] wrote {len(strips)} strips + overview.png to {cli.png_dir}", flush=True)


if __name__ == "__main__":
    main()
