import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train Siamese ResNet18-CBAM with Dynamic Margin Ranking Loss "
            "(same-specimen pairwise DI evolution)"
        )
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--reduction", type=int, default=16, help="CBAM channel reduction")
    parser.add_argument("--img_aug", action="store_true", help="Synced horizontal flip on pairs")
    parser.add_argument(
        "--data_path",
        type=str,
        default="id_list_val_R307.json",
        help="Split json under Data/<pattern_dirname>/",
    )
    parser.add_argument(
        "--pattern_dirname",
        type=str,
        default="DamagePattern_inpaint_selfcondnoDI",
        help="Inpainted DamagePattern-compatible folder under Data/",
    )
    parser.add_argument(
        "--use_mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Gray-only by default for Siamese inpaint runs. Pass --use_mask for 2ch.",
    )
    parser.add_argument(
        "--use_rgb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="3-channel RGB ImageNet-norm input (no grayscale convert). Incompatible with --use_mask.",
    )
    parser.add_argument(
        "--use_clahe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional CLAHE. Off by default (DIresnet18 / original-photo runs).",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--figure_last_n_epochs",
        type=int,
        default=1,
        help="Only save/log scatter figures in the last N epochs (1 = end of training only)",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default="./results/DamageQuantification/resnet18_siamese_inpaint_nomask",
    )
    # DropBlock (optional; nomask baseline used 0)
    parser.add_argument("--dropblock_prob", type=float, default=0.0)
    parser.add_argument("--dropblock_size_l3", type=int, default=3)
    parser.add_argument("--dropblock_size_l4", type=int, default=2)
    # Joint loss (Focal-R absolute + Focal-R pairwise gap)
    parser.add_argument(
        "--lambda_rank",
        type=float,
        default=0.5,
        help="Weight of pairwise gap loss L_pair",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal-R (1+|e|)^γ for L_abs and L_pair",
    )
    parser.add_argument(
        "--focal_beta",
        type=float,
        default=1.0,
        help="Smooth-L1 beta inside Focal-R (applied after loss_scale)",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=100.0,
        help="Scale Pred/GT by this factor before Focal-R (DI→≈0–100 range)",
    )
    # Deprecated alias kept so old job scripts / args.json still work.
    parser.add_argument(
        "--smooth_l1_beta",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int, default=42)
    # Primary target: original DI (recommended for multi-task) or residual vDI.
    parser.add_argument(
        "--target_mode",
        type=str,
        default="di",
        choices=["di", "vdi"],
        help="Main regression / ranking target: original DI or residual vDI",
    )
    parser.add_argument(
        "--multi_task",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Dual heads: DI (main) + R_spall (aux). Default off (DI only).",
    )
    parser.add_argument(
        "--rs_loss_weight",
        type=float,
        default=1.0,
        help="α in Loss = Loss_DI + α · L_Rs when uncertainty weighting is off",
    )
    parser.add_argument(
        "--rs_loss_scale",
        type=float,
        default=None,
        help="Scale Pred/GT Rs before MSE (default: same as --loss_scale)",
    )
    parser.add_argument(
        "--rs_pos_weight",
        type=float,
        default=4.0,
        help="Extra weight on samples with R_spall>0 inside L_Rs",
    )
    parser.add_argument(
        "--uncertainty_weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Kendall uncertainty weighting between DI and Rs losses (default on)",
    )
    parser.add_argument(
        "--rs_warmup_epochs",
        type=int,
        default=5,
        help="Train DI-only for this many epochs before enabling L_Rs",
    )
    parser.add_argument(
        "--include_val_in_train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge val split into train and disable validation loop (default on)",
    )
    parser.add_argument(
        "--di_low_thresh",
        type=float,
        default=0.3,
        help="DI ≤ this threshold gets di_low_weight on DI loss",
    )
    parser.add_argument(
        "--di_low_weight",
        type=float,
        default=1.0,
        help="Multiplier on DI loss for low-DI samples (1.0 = off)",
    )
    parser.add_argument(
        "--rs0_weight",
        type=float,
        default=1.0,
        help="Multiplier on DI loss when R_spall≈0 (1.0 = off)",
    )
    parser.add_argument(
        "--rs0_eps",
        type=float,
        default=1e-6,
        help="Treat |R_spall|≤eps as Rs=0 for weighting",
    )
    # vDI label helpers (used when --target_mode vdi, and always loads R_spall for aux).
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Scale on Δ=R̄-R_spall when building vDI (ignored for target_mode=di)",
    )
    parser.add_argument(
        "--r_spall_csv",
        type=str,
        default=None,
        help="CSV with columns filename,R_spall (default: <pattern>/r_spall.csv or eadi predictions)",
    )
    parser.add_argument(
        "--r_bar_method",
        type=str,
        default="poly",
        choices=["poly", "ma"],
        help="How to fit expected spall trend R̄(vDI): poly | ma",
    )
    parser.add_argument(
        "--poly_degree",
        type=int,
        default=2,
        help="Polynomial degree for R̄(vDI) when --r_bar_method=poly",
    )
    parser.add_argument(
        "--ma_window",
        type=int,
        default=51,
        help="Moving-average window (odd) for R̄(vDI) when --r_bar_method=ma",
    )
    # Deprecated flags kept so old job scripts / args.json still parse.
    parser.add_argument("--alpha", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--di_max", type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def check_args(args, current_time):
    args.result_dir = os.path.join(args.result_root, current_time)
    os.makedirs(args.result_dir, exist_ok=True)
    return args


def save_args(args):
    with open(os.path.join(args.result_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
