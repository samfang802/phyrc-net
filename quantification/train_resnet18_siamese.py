#!/usr/bin/env python3
"""Train Siamese ResNet18-CBAM with Dynamic Margin Ranking Loss."""

from __future__ import annotations

import datetime
import os
import random

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

from argument.argument_resnet18_siamese import check_args, parse_args, save_args
from src.dataset import DamageIndexGrayMaskDataset
from src.dataset_pairwise import DamageIndexPairwiseDataset, collate_pairwise
from src.model.ResNet18_Siamese_Lightning import ResNet18SiameseLightning


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
    check_args(args, current_time)
    save_args(args)
    _set_seed(int(args.seed))

    num_workers = _resolve_num_workers(args.num_workers)

    return_rs = bool(args.multi_task) or (float(args.rs0_weight) != 1.0)
    modes = ["train", "val"] if bool(args.include_val_in_train) else ["train"]
    train_ds = DamageIndexPairwiseDataset(
        root="./Data",
        mode="train",
        modes=modes,
        image_size=args.image_size,
        data_path=args.data_path,
        pattern_dirname=args.pattern_dirname,
        use_mask=args.use_mask,
        use_rgb=bool(getattr(args, "use_rgb", False)),
        use_clahe=bool(getattr(args, "use_clahe", False)),
        use_augmentation=args.img_aug,
        seed=int(args.seed),
        beta=args.beta,
        r_spall_csv=args.r_spall_csv,
        r_bar_method=args.r_bar_method,
        poly_degree=args.poly_degree,
        ma_window=args.ma_window,
        target_mode=args.target_mode,
        return_rs=return_rs,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_pairwise,
    )

    val_dl = None
    if not bool(args.include_val_in_train):
        val_ds = DamageIndexGrayMaskDataset(
            root="./Data",
            mode="val",
            image_size=args.image_size,
            data_path=args.data_path,
            pattern_dirname=args.pattern_dirname,
            use_mask=args.use_mask,
            use_rgb=bool(getattr(args, "use_rgb", False)),
            use_clahe=bool(getattr(args, "use_clahe", False)),
            use_augmentation=False,
            beta=args.beta,
            r_spall_csv=args.r_spall_csv,
            r_bar_method=args.r_bar_method,
            poly_degree=args.poly_degree,
            ma_window=args.ma_window,
            target_mode=args.target_mode,
            return_rs=return_rs,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        print(
            f"[train_resnet18_siamese] val={len(val_ds)} (single-image R² for DI"
            f"{' and R_spall' if return_rs else ''})",
            flush=True,
        )

    print(
        f"[train_resnet18_siamese] modes={modes} train_pairs={len(train_ds)} "
        f"use_mask={args.use_mask} use_rgb={getattr(args, 'use_rgb', False)} "
        f"use_clahe={getattr(args, 'use_clahe', False)} pattern={args.pattern_dirname} "
        f"target={args.target_mode} multi_task={args.multi_task} "
        f"di_low≤{args.di_low_thresh}×{args.di_low_weight} rs0×{args.rs0_weight} "
        f"unc={args.uncertainty_weighting} rs_warmup={args.rs_warmup_epochs} "
        f"lambda_rank={args.lambda_rank} no_val={bool(args.include_val_in_train)}",
        flush=True,
    )

    model = ResNet18SiameseLightning(**vars(args))
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[train_resnet18_siamese] trainable params={n_param/1e6:.3f}M "
        f"in_channels={model.net.in_channels} multi_task={model.multi_task}",
        flush=True,
    )

    ckpt_dir = os.path.join(args.result_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if val_dl is None:
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
    else:
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
        limit_val_batches=0 if val_dl is None else 1.0,
        num_sanity_val_steps=0 if val_dl is None else min(2, len(val_dl)),
        callbacks=callback_list,
        default_root_dir=args.result_dir,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    # Lightning 2.6 only refreshes last.ckpt when top-k actually writes.
    # Val R² often peaks early, so force-write the in-memory final weights.
    last_path = os.path.join(ckpt_dir, "last.ckpt")
    trainer.save_checkpoint(last_path)
    print(
        f"[train_resnet18_siamese] wrote final {last_path} "
        f"epoch={int(trainer.current_epoch)} step={int(trainer.global_step)}",
        flush=True,
    )
    print(f"[train_resnet18_siamese] Done. results → {args.result_dir}", flush=True)


if __name__ == "__main__":
    main()
