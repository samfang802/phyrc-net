import os, torch, datetime, importlib
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from src.dataset import HystereticDataset
from argument.argument_hysteresis import parse_args, check_args, set_dataset_args, save_args

def _resolve_num_workers(requested_workers):
    cpu_count = os.cpu_count() or 1
    return max(0, min(requested_workers, cpu_count))


if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_args()
    
    
    args.data_type = {}
    args.data_version = 1
    callback_list = [
        pl.callbacks.ModelCheckpoint(monitor="epoch", every_n_epochs=int((args.epoch + 1)/2), filename="{epoch:05d}"),
        pl.callbacks.ModelCheckpoint(monitor="epoch", every_n_epochs=1, filename="last"),
        pl.callbacks.ModelCheckpoint(filename="{epoch:05d}"),
        pl.callbacks.ModelCheckpoint(monitor="validation/loss", mode="min", save_top_k=1, filename="best"),
    ]

    # check arguments
    check_args(args, current_time)
    
    
    num_workers = _resolve_num_workers(args.num_workers)
    train_ds = HystereticDataset(root="./Data/Hysteresis", mode="train", **vars(args))
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, collate_fn=train_ds.collate_fn)
    val_ds = HystereticDataset(root="./Data/Hysteresis", mode="val", **vars(args))
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, collate_fn=val_ds.collate_fn)
    

    
    # set dataset argument
    set_dataset_args(args, train_dataloader)
    # save_args
    save_args(args)

    model = getattr(importlib.import_module(f"src.model.HystereticPrediction"), args.architecture)(**vars(args))
    print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1024/1024:.4f} MB")
    print(model)

    logger = pl.loggers.CSVLogger(save_dir=args.result_dir, name="csv_logs")
    if args.use_wandb:
        logger = pl.loggers.WandbLogger(
            project = "Opensees Based damage Patterns Generation",
            group = "Hysteresis Loop Prediction",
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
