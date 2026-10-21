#!/usr/bin/env python3
"""Train the interference segmenter (src/metrics/interference.py) on the SAM masks.

Positives: original images of the train split with Data/DamagePattern/mask.
Hard negatives (default on): the same frames from the inpainted set with an all-zero mask, so the
detector learns that an inpainted region is not interference. That keeps false positives low on
generated images, which never contain the exact fixture pixels.

Model selection on val IoU; val/test report IoU, Dice, image-level detection, and the share of
inpainted frames that still get flagged (should be ~0).

Example:
  python scripts/eval/train_interference_detector.py --out_dir results/InterferenceDetector/run1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics import dino_mmd as dm
from src.metrics import interference as itf


def _parse_args():
    p = argparse.ArgumentParser(description="Train the interference segmenter")
    p.add_argument("--out_dir", default="")
    p.add_argument("--data_path", default="Data/DamagePattern/id_list_val_R307.json")
    p.add_argument("--image_dir", default="Data/DamagePattern/images")
    p.add_argument("--inpaint_dir", default="Data/DamagePattern_inpaint_selfcondnoDI/images")
    p.add_argument("--mask_dir", default="Data/DamagePattern/mask")
    p.add_argument("--no_inpainted_negatives", action="store_true")
    p.add_argument("--dino_model_id", default=dm.DEFAULT_DINO)
    p.add_argument("--input_size", type=int, default=448)
    p.add_argument("--n_train_blocks", type=int, default=2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_backbone", type=float, default=2e-5)
    p.add_argument("--pos_weight", type=float, default=4.0)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--min_area", type=int, default=150)
    p.add_argument("--border_px", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _augment(x, m, gen):
    """x, m: (B,1,H,W). Flip, small affine (reflect padding), brightness/contrast, noise."""
    B = x.shape[0]
    dev = x.device
    flip = torch.rand(B, generator=gen, device="cpu") < 0.5
    x = torch.where(flip.to(dev)[:, None, None, None], x.flip(-1), x)
    m = torch.where(flip.to(dev)[:, None, None, None], m.flip(-1), m)
    s = 1.0 + (torch.rand(B, generator=gen) - 0.5) * 0.2
    t = (torch.rand(B, 2, generator=gen) - 0.5) * 0.1
    theta = torch.zeros(B, 2, 3)
    theta[:, 0, 0] = s
    theta[:, 1, 1] = s
    theta[:, :, 2] = t
    grid = F.affine_grid(theta.to(dev), list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    m = (F.grid_sample(m, grid, mode="nearest", padding_mode="zeros", align_corners=False) > 0.5).float()
    c = 1.0 + (torch.rand(B, 1, 1, 1, generator=gen) - 0.5) * 0.3
    b = (torch.rand(B, 1, 1, 1, generator=gen) - 0.5) * 0.3
    mu = x.mean(dim=(2, 3), keepdim=True)
    x = (x - mu) * c.to(dev) + mu + b.to(dev)
    x = x + torch.randn(x.shape, generator=gen).to(dev) * (torch.rand(B, 1, 1, 1, generator=gen).to(dev) * 0.03)
    return x.clamp(0, 1), m


def _loss(logit, m, pos_weight):
    bce = F.binary_cross_entropy_with_logits(logit, m, pos_weight=torch.tensor(pos_weight, device=m.device))
    p = torch.sigmoid(logit)
    inter = (p * m).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + m.sum(dim=(1, 2, 3))
    has = (m.sum(dim=(1, 2, 3)) > 0).float()
    # Dice only on images that contain interference; negatives are handled by BCE.
    dice = (1 - (2 * inter + 1) / (den + 1)) * has
    return bce + dice.sum() / has.sum().clamp_min(1.0)


def _evaluate(model, data, cli):
    out = {}
    prob = itf.predict_prob(model, data["orig"], device=cli.device)
    out.update(itf.mask_stats(itf.clean_mask(prob, cli.thr, cli.min_area, cli.border_px), data["mask"]))
    if "inpainted" in data:
        pi = itf.predict_prob(model, data["inpainted"], device=cli.device)
        out["frac_inpainted_flagged"] = float(itf.clean_mask(pi, cli.thr, cli.min_area, cli.border_px).reshape(len(pi), -1).any(1).mean())
    return out


def _preview(model, data, path, cli, n=8):
    from PIL import Image

    idx = np.linspace(0, len(data["orig"]) - 1, n).astype(int)
    u8 = data["orig"][idx]
    pred = itf.clean_mask(itf.predict_prob(model, u8, device=cli.device), cli.thr, cli.min_area, cli.border_px)
    rows = []
    for i, k in enumerate(idx):
        img = u8[i]
        ov = np.stack([img, img, img], -1).astype(np.float32)
        ov[pred[i] > 0] = 0.5 * ov[pred[i] > 0] + 0.5 * np.array([255, 0, 0])
        gt = np.stack([img, img, img], -1).astype(np.float32)
        gt[data["mask"][k] > 0] = 0.5 * gt[data["mask"][k] > 0] + 0.5 * np.array([0, 255, 0])
        rows.append(np.concatenate([gt, ov], 1).astype(np.uint8))
    Image.fromarray(np.concatenate(rows, 0)).save(path)


def main():
    cli = _parse_args()
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    out_dir = cli.out_dir or os.path.join(
        "results", "InterferenceDetector", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(cli.data_path) as f:
        ids = json.load(f)

    data = {}
    for split in ("train", "val", "test"):
        names = ids[split]
        d = {
            "names": names,
            "orig": dm.load_real_uint8(names, cli.image_dir),
            "mask": itf.load_sam_masks(names, cli.mask_dir),
        }
        if os.path.isdir(cli.inpaint_dir):
            d["inpainted"] = dm.load_real_uint8(names, cli.inpaint_dir, use_clahe=False)
        data[split] = d
        print(f"[itf] {split}: n={len(names)}  frames with interference={d['mask'].reshape(len(names), -1).any(1).mean():.2f}", flush=True)

    X = [data["train"]["orig"]]
    Y = [data["train"]["mask"]]
    if not cli.no_inpainted_negatives and "inpainted" in data["train"]:
        X.append(data["train"]["inpainted"])
        Y.append(np.zeros_like(data["train"]["mask"]))
    X = torch.from_numpy(np.concatenate(X)).float().div(255.0)[:, None]
    Y = torch.from_numpy(np.concatenate(Y)).float()[:, None]

    model = itf.InterferenceSegmenter(
        cli.dino_model_id, input_size=cli.input_size, n_train_blocks=cli.n_train_blocks
    ).to(cli.device)
    opt = torch.optim.AdamW(model.trainable_groups(cli.lr_head, cli.lr_backbone), weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(X) / cli.batch_size)
    total = steps_per_epoch * cli.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
    )
    gen = torch.Generator().manual_seed(cli.seed)

    best, history = -1.0, []
    for epoch in range(cli.epochs):
        model.train()
        perm = torch.randperm(len(X), generator=gen)
        run = 0.0
        for i in range(0, len(X), cli.batch_size):
            b = perm[i : i + cli.batch_size]
            x, m = _augment(X[b].to(cli.device), Y[b].to(cli.device), gen)
            loss = _loss(model(x), m, cli.pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss.detach()) * len(b)
        model.eval()
        val = _evaluate(model, data["val"], cli)
        history.append({"epoch": epoch, "train_loss": run / len(X), **val})
        print(
            f"[itf] epoch {epoch:02d} loss={run / len(X):.4f}  val IoU={val['iou']:.3f} dice={val['dice']:.3f} "
            f"img P/R={val['image_precision']:.2f}/{val['image_recall']:.2f}  "
            f"inpainted flagged={val.get('frac_inpainted_flagged', float('nan')):.2f}",
            flush=True,
        )
        if val["iou"] > best:
            best = val["iou"]
            itf.save_segmenter(model, os.path.join(out_dir, "best.pt"), {"epoch": epoch, "val": val, "thr": cli.thr, "min_area": cli.min_area, "border_px": cli.border_px})

    model, extra = itf.load_segmenter(os.path.join(out_dir, "best.pt"), cli.device)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(cli),
        "best_epoch": extra["epoch"],
        "val": _evaluate(model, data["val"], cli),
        "test": _evaluate(model, data["test"], cli),
        "history": history,
    }
    _preview(model, data["test"], os.path.join(out_dir, "preview_test.png"), cli)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[itf] best epoch {extra['epoch']}  val={report['val']}", flush=True)
    print(f"[itf] test={report['test']}", flush=True)
    print(f"[itf] wrote {out_dir}/best.pt, metrics.json, preview_test.png", flush=True)


if __name__ == "__main__":
    main()
