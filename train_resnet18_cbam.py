#!/usr/bin/env python3
"""Train ResNet18_CBAM DI regressor with PyTorch Lightning."""

from __future__ import annotations

import datetime
import os
import importlib

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from argument.argument_resnet18_cbam import check_args, parse_args, save_args
from src.dataset import DamageIndexGrayMaskDataset


def _resolve_num_workers(requested_workers):
    cpu_count = os.cpu_count() or 1
    return max(0, min(int(requested_workers), cpu_count))


def main():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_args()
    check_args(args, current_time)
    save_args(args)

    num_workers = _resolve_num_workers(args.num_workers)
    train_ds = DamageIndexGrayMaskDataset(
        root="./Data",
        mode="train",
        image_size=args.image_size,
        data_path=args.data_path,
        pattern_dirname=args.pattern_dirname,
        use_mask=args.use_mask,
        use_augmentation=args.img_aug,
        target_mode=args.target_mode,
        return_rs=bool(args.multi_task),
        beta=args.beta,
        r_spall_csv=args.r_spall_csv,
        r_bar_method=args.r_bar_method,
        poly_degree=args.poly_degree,
        ma_window=args.ma_window,
    )
    val_ds = DamageIndexGrayMaskDataset(
        root="./Data",
        mode="val",
        image_size=args.image_size,
        data_path=args.data_path,
        pattern_dirname=args.pattern_dirname,
        use_mask=args.use_mask,
        use_augmentation=False,
        target_mode=args.target_mode,
        return_rs=bool(args.multi_task),
        beta=args.beta,
        r_spall_csv=args.r_spall_csv,
        r_bar_method=args.r_bar_method,
        poly_degree=args.poly_degree,
        ma_window=args.ma_window,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    print(
        f"[train_resnet18_cbam] train={len(train_ds)} val={len(val_ds)} "
        f"use_mask={args.use_mask} pattern={args.pattern_dirname} "
        f"target={args.target_mode} multi_task={args.multi_task}",
        flush=True,
    )

    model = getattr(importlib.import_module("src.model.ResNet18_CBAM_Regressor"), "ResNet18CBAMLightning")(**vars(args))
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[train_resnet18_cbam] trainable params={n_param/1e6:.3f}M "
        f"in_channels={model.net.in_channels} multi_task={model.net.multi_task}",
        flush=True,
    )

    ckpt_dir = os.path.join(args.result_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    callback_list = [
        pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="validation/r2",
            mode="max",
            save_top_k=1,
            save_last=True,
            filename="best",
            enable_version_counter=False,
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
        num_sanity_val_steps=min(2, len(val_dl)),
        callbacks=callback_list,
        default_root_dir=args.result_dir,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    print(f"[train_resnet18_cbam] Done. results → {args.result_dir}", flush=True)


if __name__ == "__main__":
    main()
