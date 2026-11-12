#!/usr/bin/env python3
"""Train patch-weighted Siamese with independent heads + Focal-R ranking loss."""

from __future__ import annotations

import datetime
import os
import random

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

from argument.argument_resnet18_siamese import check_args, parse_args, save_args
from src.dataset_pairwise import DamageIndexPairwiseDataset, collate_pairwise
from src.model.PatchWeightedSiameseLightning import PatchWeightedSiameseLightning


def _resolve_num_workers(requested_workers):
    cpu_count = os.cpu_count() or 1
    return max(0, min(int(requested_workers), cpu_count))


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_args()
    args.use_mask = False
    check_args(args, current_time)
    save_args(args)
    _set_seed(int(args.seed))

    num_workers = _resolve_num_workers(args.num_workers)
    modes = ["train", "val"] if bool(args.include_val_in_train) else ["train"]
    train_ds = DamageIndexPairwiseDataset(
        root="./Data",
        mode="train",
        modes=modes,
        image_size=args.image_size,
        data_path=args.data_path,
        pattern_dirname=args.pattern_dirname,
        use_mask=False,
        use_augmentation=args.img_aug,
        seed=int(args.seed),
        beta=args.beta,
        r_spall_csv=args.r_spall_csv,
        r_bar_method=args.r_bar_method,
        poly_degree=args.poly_degree,
        ma_window=args.ma_window,
        target_mode=args.target_mode,
        return_rs=False,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_pairwise,
    )
    print(
        f"[train_patch_weighted_siamese] modes={modes} train_pairs={len(train_ds)} "
        f"pattern={args.pattern_dirname} target={args.target_mode} "
        f"lambda_rank={args.lambda_rank} loss_scale={args.loss_scale} "
        f"include_val={bool(args.include_val_in_train)}",
        flush=True,
    )

    model = PatchWeightedSiameseLightning(**vars(args))
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[train_patch_weighted_siamese] trainable params={n_param/1e6:.3f}M",
        flush=True,
    )

    ckpt_dir = os.path.join(args.result_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    callback_list = [
        pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            save_top_k=0,
            save_last=True,
            filename="epoch",
            enable_version_counter=False,
            every_n_epochs=1,
        ),
    ]

    logger = pl.loggers.CSVLogger(save_dir=args.result_dir, name="csv_logs")
    if args.use_wandb:
        logger = pl.loggers.WandbLogger(
            project="Opensees Based damage Patterns Generation",
            group="Damage Quantification",
            save_dir=args.result_dir,
            name=current_time,
        )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        log_every_n_steps=2,
        max_epochs=args.epoch + 1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=callback_list,
        default_root_dir=args.result_dir,
    )
    trainer.fit(model, train_dataloaders=train_dl)
    print(f"[train_patch_weighted_siamese] Done. results → {args.result_dir}", flush=True)


if __name__ == "__main__":
    main()
