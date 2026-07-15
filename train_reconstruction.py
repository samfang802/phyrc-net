import os, torch, datetime, importlib
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from src.dataset import DamagePatternDataset
from src.utils import build_damage_image_transform_with_loss_mask
from argument.argument_reconstruction import parse_args, check_args, save_args

def _resolve_num_workers(requested_workers):
    cpu_count = os.cpu_count() or 1
    return max(0, min(requested_workers, cpu_count))


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    
    callback_list = [
        pl.callbacks.ModelCheckpoint(monitor="epoch", every_n_epochs=int((args.epoch + 1)/2), filename="{epoch:05d}"),
        pl.callbacks.ModelCheckpoint(filename="{epoch:05d}"),
        pl.callbacks.ModelCheckpoint(monitor="val/rec_loss", mode="min", save_top_k=1, filename="best_{val/rec_loss:.4f}"),
    ]
    
    
    # ddconfig
    match args.architecture:
        case "VQGAN":
            args.ddconfig = {
                "double_z": False,
                "z_channels": 4,
                "resolution": args.resolution,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": (1, 2, 2, 4),
                "num_res_blocks": 2,
                "attn_resolutions": [32],
                "dropout": 0.0
            }
        case "VAE":
            args.ddconfig = {
                "double_z": True,
                "z_channels": 4,
                "resolution": args.resolution,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": (1, 2, 2, 4),
                "num_res_blocks": 2,
                "attn_resolutions": [32],
                "dropout": 0.0
            }
    
    
    # lossconfig
    args.lossconfig = {
        "disc_conditional": False,
        "disc_in_channels": 3,
        "disc_start": args.disc_start,
        "disc_weight": 0.75,
    }
    
    if args.architecture == "VQGAN":
        args.lossconfig["codebook_weight"] = 1.0



    # check arguments
    check_args(args, current_time)
    
    train_transform = build_damage_image_transform_with_loss_mask(
        image_size=args.resolution,
        use_augmentation=args.img_aug,
        output_scale=False,
        aug_prob=args.aug_prob,
        aug_blur_prob=args.aug_blur_prob,
        aug_brightness=args.aug_brightness,
        aug_contrast=args.aug_contrast,
        aug_saturation=args.aug_saturation,
        aug_hue=args.aug_hue,
        aug_translate=args.aug_translate,
        aug_scale_min=args.aug_scale_min,
        aug_scale_max=args.aug_scale_max,
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
        image_size=args.resolution,
        use_augmentation=False,
        output_scale=False,
    )

    ds = DamagePatternDataset(root="./Data", transform=val_transform)
    # Split the dataset into train and validation
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(ds), generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_ds = torch.utils.data.Subset(DamagePatternDataset(root="./Data", transform=train_transform), train_indices)
    val_ds = torch.utils.data.Subset(DamagePatternDataset(root="./Data", transform=val_transform), val_indices)
    num_workers = _resolve_num_workers(args.num_workers)
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    
    # save_args
    save_args(args)

    model = importlib.import_module(f"src.model.ImageReconstruction").__dict__[args.architecture](**vars(args))
    print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1024/1024:.4f} MB")
    print(model)

    logger = pl.loggers.CSVLogger(save_dir=args.result_dir, name="csv_logs")
    if args.use_wandb:
        logger = pl.loggers.WandbLogger(
            project = "Opensees Based damage Patterns Generation",
            group = "Damage Pattern Reconstruction",
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
        max_epochs = args.epoch + 1,
        num_sanity_val_steps=len(val_dataloader),
        callbacks = callback_list
    )


    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
