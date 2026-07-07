import os, torch, wandb, datetime, importlib
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from src.dataset import DamageIndexDataset
from argument.argument_quantification import parse_args, check_args, set_dataset_args, save_args


if __name__ == "__main__":
    # wandb.login()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_args()
    
    
    callback_list = [
        pl.callbacks.ModelCheckpoint(monitor="epoch", every_n_epochs=int((args.epoch + 1)/2), filename="{epoch:05d}"),
        pl.callbacks.ModelCheckpoint(filename="{epoch:05d}"),
        pl.callbacks.ModelCheckpoint(monitor="validation/r2", mode="max", save_top_k=1, filename="best_{validation/r2:.4f}"),
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
    
    transforms = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    
    train_ds = DamageIndexDataset(root="./Data", mode="train", transform=transforms, **vars(args))
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count())
    val_ds = DamageIndexDataset(root="./Data", mode="val", transform=transforms, **vars(args))
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())
    

    
    # set dataset argument
    set_dataset_args(args, train_dataloader)
    # save_args
    save_args(args)

    model = importlib.import_module(f"src.model.DamageQuantification").__dict__["DamageQuantification"](**vars(args))
    print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1024/1024:.4f} MB")
    print(model)

    wandb_logger = pl.loggers.WandbLogger(
        project = "Opensees Based damage Patterns Generation",
        group = "Damage Quantification",
        save_dir = args.result_dir, 
        name = current_time,
    )
    
    # training
    gpu = "gpu" if torch.cuda.is_available else "cpu"
    trainer = pl.Trainer(
        accelerator = gpu, 
        devices = 1,
        logger = wandb_logger,
        log_every_n_steps=2,
        max_epochs = args.epoch + 1,
        num_sanity_val_steps=len(val_dataloader),
        callbacks = callback_list
    )


    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
