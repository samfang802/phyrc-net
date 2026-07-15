import os, torch, datetime, importlib, json, glob, argparse
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import PatternLoopDataset
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



if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("[train_mixed] Starting...", flush=True)
    args = parse_args()
    print(
        f"[train_mixed] architecture={args.architecture} epoch={args.epoch} "
        f"batch_size={args.batch_size} num_workers={args.num_workers} "
        f"use_sd3_vae_latent={getattr(args, 'use_sd3_vae_latent', False)}",
        flush=True,
    )
    
    
    args.data_type = {}
    args.data_version = 1
    callback_list = [
        pl.callbacks.ModelCheckpoint(monitor="epoch", every_n_epochs=int((args.epoch + 1)/2), filename="{epoch:05d}"),
        pl.callbacks.ModelCheckpoint(filename="{epoch:05d}"),
        pl.callbacks.ModelCheckpoint(monitor="val/fid", mode="min", save_top_k=1, filename="best_fid_{val/fid:.2f}"),
        pl.callbacks.ModelCheckpoint(monitor="val/ssim", mode="max", save_top_k=1, filename="best_ssim_{val/ssim:.4f}")
    ]

    # check arguments
    check_args(args, current_time)
    
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
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
    print("[train_mixed] Building train dataset (hidden-state extraction may take several minutes)...", flush=True)
    train_ds = PatternLoopDataset(root="./Data", mode="train", transform=transform, extractor=extractor, extractor_args=extractor_args, **vars(args))
    print("[train_mixed] Building val dataset (hidden-state extraction may take several minutes)...", flush=True)
    val_ds = PatternLoopDataset(root="./Data", mode="val", transform=transform, extractor=extractor, extractor_args=extractor_args, **vars(args))
    print(f"[train_mixed] train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
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
        check_val_every_n_epoch=args.val_every_n_epoch,
        max_epochs = args.epoch + 1,
        num_sanity_val_steps=min(2, len(val_dataloader)),
        callbacks = callback_list
    )


    print("[train_mixed] Starting Lightning trainer.fit(...)", flush=True)
    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
