import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet18_CBAM Lightning DI regressor (gray + SAM mask)"
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--reduction", type=int, default=16, help="CBAM channel reduction")
    parser.add_argument("--img_aug", action="store_true", help="Train-time horizontal flip")
    parser.add_argument(
        "--data_path",
        type=str,
        default="id_list_val_R307.json",
        help="Split json filename under Data/<pattern_dirname>/",
    )
    parser.add_argument(
        "--pattern_dirname",
        type=str,
        default="DamagePattern",
        help="Subfolder under Data/ (e.g. DamagePattern_inpaint_selfcondnoDI)",
    )
    parser.add_argument(
        "--use_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Concat SAM mask as 2nd channel (default on). Use --no-use_mask for gray-only.",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--figure_last_n_epochs",
        type=int,
        default=100,
        help="Only save/log scatter figures in the last N epochs; earlier epochs log numbers only",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default="./results/DamageQuantification/resnet18_cbam",
    )
    # DropBlock / Manifold Mixup / Focal-R
    parser.add_argument("--dropblock_prob", type=float, default=0.1)
    parser.add_argument("--dropblock_size_l3", type=int, default=3, help="DropBlock size on layer3")
    parser.add_argument("--dropblock_size_l4", type=int, default=2, help="DropBlock size on layer4")
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Beta(α,α) for Manifold Mixup")
    parser.add_argument("--mixup_prob", type=float, default=1.0, help="Per-batch mixup probability")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal-R (1+|e|)^γ")
    parser.add_argument("--focal_beta", type=float, default=1.0, help="Smooth L1 beta")
    parser.add_argument(
        "--target_mode",
        type=str,
        default="di",
        choices=["di", "vdi", "rs"],
        help="Regression target: OpenSees DI, vDI, or R_spall",
    )
    parser.add_argument(
        "--nonneg_head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ReLU on the scalar head (use for R_spall)",
    )
    parser.add_argument(
        "--multi_task",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Dual heads: predict DI and R_spall (both logged as R²)",
    )
    parser.add_argument(
        "--rs_loss_weight",
        type=float,
        default=1.0,
        help="L = L_DI + rs_loss_weight * L_Rs (Focal-R on both)",
    )
    # vDI label: Δ = R̄(vDI) - R_spall; vDI = clip(vDI + β·Δ, 0, 1)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Scale on Δ=R̄(vDI)-R_spall for vDI = clip(vDI + β·Δ, 0, 1)",
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
