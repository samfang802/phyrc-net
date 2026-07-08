import os, torch, datetime, importlib
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from src.dataset import DamageIndexDataset
from src.utils import build_damage_image_transform
from argument.argument_quantification import parse_args, check_args, set_dataset_args, save_args

def _resolve_num_workers(requested_workers):
    cpu_count = os.cpu_count() or 1
    return max(0, min(requested_workers, cpu_count))


if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_args()
    
    
    callback_list = [
        pl.callbacks.ModelCheckpoint(monitor="epoch", every_n_epochs=int((args.epoch + 1)/2), filename="{epoch:05d}"),
        pl.callbacks.ModelCheckpoint(filename="{epoch:05d}"),
        pl.callbacks.ModelCheckpoint(monitor="validation/r2", mode="max", save_top_k=1, filename="best"),
    ]

    # check arguments
    check_args(args, current_time)
    
    args.column_info = [
        "id",
        "concreteStrength",
        "transverse_yield",
        "transverse_strength",
        "longitudinal_corner_yield",
        "longitudinal_corner_strength",
        "longitudinal_intermediate_yield",
        "longitudinal_intermediate_strength",
        "lMeasured",
        "axialLoad",
        "fcc",
        "epscc",
        "fcu",
        "epscu",
        "Vn",
        "Vst",
        "reinforcementRatio",
        "volTransReinfRatio"
    ]
    
    train_transform = build_damage_image_transform(
        image_size=224,
        use_augmentation=args.img_aug,
        output_scale=True,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
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
    val_transform = build_damage_image_transform(
        image_size=224,
        use_augmentation=False,
        output_scale=True,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
    )
    
    
    num_workers = _resolve_num_workers(args.num_workers)
    train_ds = DamageIndexDataset(root="./Data", mode="train", transform=train_transform, **vars(args))
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_ds = DamageIndexDataset(root="./Data", mode="val", transform=val_transform, **vars(args))
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    

    
    # set dataset argument
    set_dataset_args(args, train_dataloader)
    # save_args
    save_args(args)

    model = importlib.import_module(f"src.model.DamageQuantification").__dict__["DamageQuantification"](**vars(args))
    print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1024/1024:.4f} MB")
    print(model)

    logger = pl.loggers.CSVLogger(save_dir=args.result_dir, name="csv_logs")
    if args.use_wandb:
        logger = pl.loggers.WandbLogger(
            project = "Opensees Based damage Patterns Generation",
            group = "Damage Quantification",
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
