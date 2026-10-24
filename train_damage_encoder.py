"""Train the damage-state encoder (docs/DAMAGE_STATE_ENCODER.md).

Split: id_list_big.json train / val with the six imaged specimens removed (they never enter training
or checkpoint selection). One run, early stopping on val loss.

  python train_damage_encoder.py --result_dir results/DamageStateEncoder
"""

import argparse
import datetime
import json
import os

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.damage_state import DamageStateDataset, specimen_split
from src.model.damage_state_encoder import DamageStateEncoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", default="results/DamageStateEncoder")
    p.add_argument("--max_epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--mirror_prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input_mode", choices=("tokens", "points"), default="points",
                   help="points: the GRU extractor's own per-point input (column info, drift, simulated loop)")
    p.add_argument("--static_info", type=int, default=1, help="1: s also reads the 17 column-info features")
    p.add_argument("--force_weight", type=float, default=1.0, help="points mode: per-point experimental-force loss")
    p.add_argument("--seq_info", choices=("raw", "dimless"), default="raw",
                   help="points mode: per-point specimen features, 17 column info (raw) or 7 dimensionless descriptors")
    p.add_argument("--force_target", choices=("signed", "abs"), default="signed", help="points mode: per-point force target")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pl.seed_everything(args.seed)
    out = os.path.join(args.result_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out, exist_ok=True)
    tr_ids, va_ids = specimen_split()
    train_ds = DamageStateDataset(tr_ids, seq_info=args.seq_info)
    val_ds = DamageStateDataset(va_ids, stats=train_ds.stats, seq_info=args.seq_info)
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump({k: v.tolist() for k, v in train_ds.stats.items()}, f)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump({**vars(args), "train_ids": tr_ids, "val_ids": va_ids}, f, indent=2)
    print(f"[damage_encoder] train {len(train_ds)} specimens, val {len(val_ds)}; out {out}", flush=True)

    dl = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
                                        collate_fn=DamageStateDataset.collate)
    model = DamageStateEncoder(hidden=args.hidden, layers=args.layers, dropout=args.dropout, lr=args.lr,
                               weight_decay=args.weight_decay, mirror_prob=args.mirror_prob, input_mode=args.input_mode,
                               static_info=bool(args.static_info), force_weight=args.force_weight,
                               seq_info=args.seq_info, force_target=args.force_target)
    ckpt = pl.callbacks.ModelCheckpoint(dirpath=out, monitor="val/loss", mode="min", save_top_k=1, filename="best")
    stop = pl.callbacks.EarlyStopping(monitor="val/loss", mode="min", patience=args.patience)
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, max_epochs=args.max_epochs,
                         callbacks=[ckpt, stop], logger=pl.loggers.CSVLogger(out, name="csv"), enable_progress_bar=False,
                         log_every_n_steps=1)
    trainer.fit(model, dl(train_ds, True), dl(val_ds, False))
    best = DamageStateEncoder.load_from_checkpoint(ckpt.best_model_path)
    res = trainer.validate(best, dl(val_ds, False), verbose=False)[0]
    with open(os.path.join(out, "val_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in res.items()}, f, indent=2)
    print(f"[damage_encoder] best {ckpt.best_model_path} epoch-stop {trainer.current_epoch}", flush=True)
    print(json.dumps({k: round(float(v), 4) for k, v in res.items()}, indent=1), flush=True)
