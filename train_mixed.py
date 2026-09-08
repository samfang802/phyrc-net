import os, torch, datetime, importlib, json, glob, argparse
import numpy as np
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask
from argument.argument_mixed import parse_args, check_args, set_dataset_args, save_args

def _resolve_num_workers(requested_workers):
    cpu_count = os.cpu_count() or 1
    return max(0, min(requested_workers, cpu_count))

def _resolve_required_match(path_pattern, description):
    matches = glob.glob(path_pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"Could not find {description} with pattern: {path_pattern}")
    return matches[0]

def _load_args_from_result_dir(result_dir):
    args_path = _resolve_required_match(os.path.join(result_dir, "**/args.json"), "args.json")
    return argparse.Namespace(**json.load(open(args_path, "r")))


def _save_augmented_grid_preview(dataset, result_dir, num_images=100, grid_size=10):
    total = len(dataset)
    if total == 0:
        print("[train_mixed] Skip augmentation preview: empty dataset.", flush=True)
        return

    sample_count = min(num_images, total)
    indices = torch.randperm(total)[:sample_count].tolist()
    images = []
    for idx in indices:
        sample = dataset[idx]
        if isinstance(sample, dict):
            image = sample.get("image_target", sample.get("image"))
        else:
            image, *_ = sample
        if torch.is_tensor(image):
            img = image.detach().cpu()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = (img * 0.5 + 0.5).clamp(0.0, 1.0)
                images.append(img.permute(1, 2, 0).numpy())

    if not images:
        print("[train_mixed] Skip augmentation preview: no valid images sampled.", flush=True)
        return

    rows = grid_size
    cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i])

    fig.suptitle("Train Augmentation Preview (Random 100)", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(result_dir, "data_check", "train_aug_preview_10x10.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train_mixed] Saved augmentation preview: {out_path}", flush=True)



class PeriodicPtCheckpoint(pl.Callback):
    def __init__(self, every_n_epochs=100):
        super().__init__()
        self.every_n_epochs = max(1, int(every_n_epochs))

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_num = trainer.current_epoch + 1
        if epoch_num % self.every_n_epochs != 0:
            return
        save_path = os.path.join(pl_module.result_dir, f"checkpoint_epoch_{epoch_num}.pt")
        payload = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "state_dict": pl_module.state_dict(),
            "hyper_parameters": dict(getattr(pl_module, "hparams", {})),
        }
        torch.save(payload, save_path)
        print(f"[train_mixed] Saved periodic checkpoint: {save_path}", flush=True)


class PeriodicStepCheckpoint(pl.Callback):
    def __init__(self, every_n_steps=2000, start_step=10000):
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))
        self.start_step = max(0, int(start_step))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step < self.start_step or step % self.every_n_steps != 0:
            return
        save_path = os.path.join(pl_module.result_dir, f"checkpoint_step_{step:06d}.pt")
        payload = {
            "epoch": trainer.current_epoch,
            "global_step": step,
            "state_dict": pl_module.state_dict(),
            "hyper_parameters": dict(getattr(pl_module, "hparams", {})),
        }
        torch.save(payload, save_path)
        print(f"[train_mixed] Saved periodic step checkpoint: {save_path}", flush=True)


class ResetEarlyStoppingPatience(pl.Callback):
    """Reset EarlyStopping wait counter after Lightning restores callback state from ckpt."""

    def __init__(self, reset_best_score=False):
        super().__init__()
        self.reset_best_score = bool(reset_best_score)
        self._did_reset = False

    def _reset(self, trainer):
        if self._did_reset:
            return
        for callback in trainer.callbacks:
            if not isinstance(callback, pl.callbacks.EarlyStopping):
                continue
            old_wait = int(getattr(callback, "wait_count", 0))
            callback.wait_count = 0
            msg = (
                f"[train_mixed] Reset EarlyStopping patience: wait_count {old_wait} -> 0 "
                f"(monitor={callback.monitor}, patience={callback.patience})"
            )
            if self.reset_best_score:
                init = float("inf") if callback.mode == "min" else float("-inf")
                best_score = getattr(callback, "best_score", init)
                if torch.is_tensor(best_score):
                    callback.best_score = torch.tensor(init, device=best_score.device, dtype=best_score.dtype)
                else:
                    callback.best_score = init
                msg += f", best_score reset to {init}"
            print(msg, flush=True)
            self._did_reset = True

    def on_fit_start(self, trainer, pl_module):
        self._reset(trainer)

    def on_train_start(self, trainer, pl_module):
        # Safety: run again in case callback state is restored after on_fit_start.
        self._reset(trainer)

if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("[train_mixed] Starting...", flush=True)
    args = parse_args()
    print(
        f"[train_mixed] architecture={args.architecture} epoch={args.epoch} "
        f"batch_size={args.batch_size} num_workers={args.num_workers} "
        f"use_sd3_vae_latent={getattr(args, 'use_sd3_vae_latent', False)} "
        f"use_mafp={getattr(args, 'use_mafp', False)} "
        f"use_patch_diffusion={getattr(args, 'use_patch_diffusion', False)}",
        flush=True,
    )
    
    
    args.data_type = {}
    args.data_version = 1
    callback_list = [pl.callbacks.ModelCheckpoint(save_last=True, filename="last")]
    if args.checkpoint_every_n_epochs > 0:
        callback_list.insert(0, PeriodicPtCheckpoint(every_n_epochs=args.checkpoint_every_n_epochs))
    if args.checkpoint_every_n_steps > 0:
        start_step = int(getattr(args, "checkpoint_start_step", 10000))
        callback_list.insert(
            0,
            PeriodicStepCheckpoint(
                every_n_steps=args.checkpoint_every_n_steps,
                start_step=start_step,
            ),
        )
        print(
            f"[train_mixed] Step ckpt: every {args.checkpoint_every_n_steps} steps "
            f"after step>={start_step}",
            flush=True,
        )
    if args.use_early_stopping and args.early_stop_patience > 0:
        callback_list.append(
            pl.callbacks.EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
            )
        )
        if args.resume.strip() and args.reset_early_stop_patience:
            callback_list.append(
                ResetEarlyStoppingPatience(reset_best_score=args.reset_early_stop_best_score)
            )
            print(
                "[train_mixed] Will reset EarlyStopping patience on resume "
                f"(reset_best_score={args.reset_early_stop_best_score})",
                flush=True,
            )
    elif args.use_early_stopping and args.early_stop_patience <= 0:
        print("[train_mixed] use_early_stopping=True but early_stop_patience<=0; EarlyStopping disabled.", flush=True)
    else:
        print("[train_mixed] EarlyStopping disabled (use_early_stopping=False).", flush=True)
    callback_list.append(
        pl.callbacks.ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            filename="best_val_loss_{val/loss:.4f}",
        )
    )

    # check arguments
    check_args(args, current_time)
    
    train_transform = build_damage_image_transform_with_loss_mask(
        image_size=args.image_size,
        use_augmentation=args.img_aug,
        output_scale=True,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
        use_grayscale_input=args.use_grayscale_input,
        use_clahe=args.use_clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid_size=args.clahe_tile_grid_size,
        aug_prob=args.aug_prob,
        aug_blur_prob=args.aug_blur_prob,
        aug_brightness=args.aug_brightness,
        aug_contrast=args.aug_contrast,
        aug_saturation=args.aug_saturation,
        aug_hue=args.aug_hue,
        aug_hflip_prob=args.aug_hflip_prob,
        aug_blur_kernel=args.aug_blur_kernel,
        aug_blur_sigma_min=args.aug_blur_sigma_min,
        aug_blur_sigma_max=args.aug_blur_sigma_max,
        aug_erase_prob=args.aug_erase_prob,
        aug_erase_scale_min=args.aug_erase_scale_min,
        aug_erase_scale_max=args.aug_erase_scale_max,
        aug_erase_ratio_min=args.aug_erase_ratio_min,
        aug_erase_ratio_max=args.aug_erase_ratio_max,
        aug_erase_random_value=args.aug_erase_random_value,
    )
    val_transform = build_damage_image_transform_with_loss_mask(
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
    print(
        f"[train_mixed] img_aug={args.img_aug} use_grayscale_input={args.use_grayscale_input} "
        f"use_clahe={args.use_clahe} clahe_clip_limit={args.clahe_clip_limit} "
        f"clahe_tile_grid_size={args.clahe_tile_grid_size} "
        f"aug_brightness={args.aug_brightness} aug_contrast={args.aug_contrast} "
        f"aug_erase_prob={args.aug_erase_prob}",
        flush=True,
    )
    
    # Load extractor
    if not args.extractor_path:
        raise ValueError("Please provide --extractor_path to a trained hysteresis result directory.")
    print(f"[train_mixed] Loading extractor from: {args.extractor_path}", flush=True)
    extractor_path = args.extractor_path
    extractor_args = _load_args_from_result_dir(extractor_path)
    extractor = importlib.import_module(f"src.model.HystereticPrediction").__dict__[extractor_args.architecture](**vars(extractor_args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)
    extractor_ckpt = _resolve_required_match(os.path.join(extractor_path, "**/best.ckpt"), "extractor checkpoint")
    state_dict = torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"]
    status = extractor.load_state_dict(state_dict)
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False
    print("Extractor Loading Status: ", status, flush=True)
    
    
    num_workers = _resolve_num_workers(args.num_workers)
    # Build once so we don't pass the same keyword twice via **vars(args).
    dataset_kwargs = dict(vars(args))
    dataset_kwargs["use_loss_mask"] = True
    dataset_kwargs["use_damage_mask_condition"] = (
        args.architecture in ("CFMADM", "LaplacianBandCFM")
        and bool(args.use_damage_mask_condition)
    )
    if dataset_kwargs["use_damage_mask_condition"]:
        if bool(dataset_kwargs.get("use_object_loss_mask", True)):
            print(
                "[train_mixed] use_damage_mask_condition=True -> forcing use_object_loss_mask=False "
                "(mask is condition only; CFM loss is unmasked).",
                flush=True,
            )
        dataset_kwargs["use_object_loss_mask"] = False
        args.use_object_loss_mask = False
    print(
        f"[train_mixed] use_damage_mask_condition={dataset_kwargs['use_damage_mask_condition']} "
        f"use_object_loss_mask={dataset_kwargs.get('use_object_loss_mask', True)}",
        flush=True,
    )
    print("[train_mixed] Building train dataset (hidden-state extraction may take several minutes)...", flush=True)
    train_ds = PatternLoopDataset(
        root="./Data",
        mode="train",
        transform=train_transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **dataset_kwargs,
    )
    print("[train_mixed] Building val dataset (hidden-state extraction may take several minutes)...", flush=True)
    val_ds = PatternLoopDataset(
        root="./Data",
        mode="val",
        transform=val_transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **dataset_kwargs,
    )
    print(f"[train_mixed] train={len(train_ds)} val={len(val_ds)}", flush=True)
    _save_augmented_grid_preview(train_ds, args.result_dir, num_images=100, grid_size=10)

    train_batch_size = int(args.batch_size)
    train_dataloader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    # Load regressor (only required when DI auxiliary loss is enabled)
    regressor = None
    if args.add_DI_loss:
        if not args.regressor_path:
            raise ValueError("Please provide --regressor_path when --add_DI_loss is enabled.")
        regressor_path = args.regressor_path
        regressor_args = _load_args_from_result_dir(regressor_path)
        regressor = importlib.import_module(f"src.model.DamageQuantification").__dict__["DamageQuantification"](**vars(regressor_args))
        regressor_ckpt = _resolve_required_match(os.path.join(regressor_path, "**/best_*.ckpt"), "regressor checkpoint")
        state_dict = torch.load(regressor_ckpt, map_location=device, weights_only=False)["state_dict"]
        status = regressor.load_state_dict(state_dict)
        regressor = regressor.to(device)
        regressor.eval()
        for p in regressor.parameters():
            p.requires_grad = False
        print("Regressor Loading Status: ", status)
    elif args.regressor_path:
        print("Skipping regressor load: --add_DI_loss is disabled; DI/direction come from label.csv.")
    
    
    
    # set dataset argument
    set_dataset_args(args, train_dataloader)
    # save_args
    save_args(args)

    print(f"[train_mixed] Building model: {args.architecture}", flush=True)
    model = importlib.import_module(f"src.model.ImageGeneration").__dict__[args.architecture](regressor=regressor,**vars(args))
    print(
        f"[train_mixed] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1024/1024:.4f} MB",
        flush=True,
    )

    logger = pl.loggers.CSVLogger(save_dir=args.result_dir, name="csv_logs")
    if args.use_wandb:
        logger = pl.loggers.WandbLogger(
            project = "Opensees Based damage Patterns Generation",
            group = "Damage Pattern Generation",
            save_dir = args.result_dir, 
            name = current_time,
        )
    
    # training
    gpu = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator = gpu, 
        devices = 1,
        logger = logger,
        log_every_n_steps=2,
        check_val_every_n_epoch=None,
        val_check_interval=args.val_check_interval,
        max_steps=args.max_steps,
        max_epochs=-1,
        num_sanity_val_steps=min(2, len(val_dataloader)),
        callbacks = callback_list
    )


    ckpt_path = args.resume.strip() if getattr(args, "resume", "") else None
    if ckpt_path:
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
        print(f"[train_mixed] Resuming from checkpoint: {ckpt_path}", flush=True)
    print("[train_mixed] Starting Lightning trainer.fit(...)", flush=True)
    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=ckpt_path)
