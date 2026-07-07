import os, torch, wandb, datetime, importlib, json, glob, argparse
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import PatternLoopDataset
from argument.argument_mixed import parse_args, check_args, set_dataset_args, save_args



if __name__ == "__main__":
    # wandb.login()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_args()
    
    
    args.data_type = {}
    args.data_version = 1
    callback_list = [
        pl.callbacks.ModelCheckpoint(monitor="epoch", every_n_epochs=int((args.epoch + 1)/2), filename="{epoch:05d}"),
        pl.callbacks.ModelCheckpoint(filename="{epoch:05d}"),
        pl.callbacks.ModelCheckpoint(monitor="val/fid", mode="min", save_top_k=1, filename="{validation/fid:.2f}"),
        pl.callbacks.ModelCheckpoint(monitor="val/ssim", mode="max", save_top_k=1, filename="{validation/ssim:.4f}")
    ]

    # check arguments
    check_args(args, current_time)
    
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    # Load extractor
    extractor_path = args.extractor_path
    extractor_args = json.load(open(glob.glob(os.path.join(extractor_path, "**/args.json"), recursive=True)[0], "r"))
    extractor_args = argparse.Namespace(**extractor_args)
    extractor = importlib.import_module(f"src.model.HystereticPrediction").__dict__[extractor_args.architecture](**vars(extractor_args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)
    state_dict = torch.load(glob.glob(os.path.join(extractor_path, "**/best.ckpt"), recursive=True)[0])["state_dict"]
    status = extractor.load_state_dict(state_dict)
    print("Extractor Loading Status: ", status)
    
    
    train_ds = PatternLoopDataset(root="./Data", mode="train", transform=transform, extractor=extractor, extractor_args=extractor_args, **vars(args))
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count())
    val_ds = PatternLoopDataset(root="./Data", mode="val", transform=transform, extractor=extractor, extractor_args=extractor_args, **vars(args))
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())
    
    # Load regressor
    regressor_path = "./results/DamageQuantification/vit_b_16/2024-06-12_14-01-03"
    regressor_args = json.load(open(glob.glob(os.path.join(regressor_path, "**/args.json"), recursive=True)[0], "r"))
    regressor_args = argparse.Namespace(**regressor_args)
    regressor = importlib.import_module(f"src.model.DamageQuantification").__dict__["DamageQuantification"](**vars(regressor_args))
    state_dict = torch.load(glob.glob(os.path.join(regressor_path, "**/r2*.ckpt"), recursive=True)[0])["state_dict"]
    status = regressor.load_state_dict(state_dict)
    print("Regressor Loading Status: ", status)
    
    
    
    # set dataset argument
    set_dataset_args(args, train_dataloader)
    # save_args
    save_args(args)

    model = importlib.import_module(f"src.model.ImageGeneration").__dict__[args.architecture](regressor=regressor,**vars(args))
    print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1024/1024:.4f} MB")
    print(model)

    wandb_logger = pl.loggers.WandbLogger(
        project = "Opensees Based damage Patterns Generation",
        group = "Damage Pattern Generation",
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
        check_val_every_n_epoch=100, 
        max_epochs = args.epoch + 1,
        num_sanity_val_steps=len(val_dataloader),
        callbacks = callback_list
    )


    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
