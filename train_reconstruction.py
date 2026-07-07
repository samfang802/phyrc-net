import os, torch, wandb, datetime, importlib
import lightning.pytorch as pl
from argparse import Namespace
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from src.dataset import DamagePatternDataset
from argument.argument_reconstruction import parse_args, check_args, save_args


if __name__ == "__main__":
    torch.manual_seed(8465468)
    wandb.login()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_args()
    
    
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
    
    transforms = v2.Compose([
        v2.Resize((256, 256)),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=False),
    ])
    
    
    ds = DamagePatternDataset(root="./Data", transform=transforms)
    # Split the dataset into train and validation
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size]) 
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count())
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())

    
    # save_args
    save_args(args)

    model = importlib.import_module(f"src.model.ImageReconstruction").__dict__[args.architecture](**vars(args))
    print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1024/1024:.4f} MB")
    print(model)

    wandb_logger = pl.loggers.WandbLogger(
        project = "Opensees Based damage Patterns Generation",
        group = "Damage Pattern Reconstruction",
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
