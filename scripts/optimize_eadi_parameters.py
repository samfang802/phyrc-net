#!/usr/bin/env python3
"""Data-driven tuning of EADI fusion weights (alpha, beta).

Target model
------------
    EADI = sqrt(alpha * R_crack + beta * R_spall)

with alpha >= 0, beta >= 0. Optimize MSE (or Smooth-L1) vs GT DI using
``scipy.optimize.minimize`` (L-BFGS-B).

Examples
--------
Synthetic smoke test::

    python scripts/optimize_eadi_parameters.py --demo

Real inpaint images + label.csv DI::

    python scripts/optimize_eadi_parameters.py --real \\
        --image-dir Data/DamagePattern_inpaint_selfcondnoDI/images \\
        --label-csv Data/DamagePattern/label.csv \\
        --scatter results/eadi_opt/scatter.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

ArrayLike = Union[Sequence[float], np.ndarray]


def predict_EADI(R_crack: np.ndarray, R_spall: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Vectorized EADI = sqrt(alpha * R_crack + beta * R_spall)."""
    inside = alpha * R_crack + beta * R_spall
    return np.sqrt(np.maximum(inside, 0.0))


def _smooth_l1(pred: np.ndarray, target: np.ndarray, beta_huber: float = 1.0) -> float:
    """Mean Smooth-L1 / Huber loss (PyTorch-style beta)."""
    diff = np.abs(pred - target)
    b = float(beta_huber)
    quadratic = np.minimum(diff, b)
    linear = diff - quadratic
    return float(0.5 * (quadratic**2).mean() / b + linear.mean())


def optimize_EADI_parameters(
    R_crack_list: ArrayLike,
    R_spall_list: ArrayLike,
    GT_DI_list: ArrayLike,
    *,
    loss: str = "mse",
    x0: Tuple[float, float] = (1.0, 2.0),
    method: str = "L-BFGS-B",
    scatter_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Fit ``alpha``, ``beta`` so predicted EADI matches GT DI.

    Parameters
    ----------
    R_crack_list, R_spall_list, GT_DI_list
        Per-image ratios / DI (same length). GT DI may exceed 1.
    loss
        ``\"mse\"`` or ``\"smooth_l1\"``.
    x0
        Initial guess ``(alpha, beta)``. Default ``(1.0, 2.0)`` (heavier spall).
    method
        ``scipy.optimize.minimize`` method supporting bounds (``L-BFGS-B`` /
        ``TNC`` / ``SLSQP`` …).
    scatter_path
        If set, save GT-vs-EADI scatter with y=x reference.
    verbose
        Print best params and metrics.

    Returns
    -------
    dict with alpha, beta, mse, smooth_l1, pearson_r, r2, success, message, n.
    """
    r_c = np.asarray(R_crack_list, dtype=np.float64).reshape(-1)
    r_s = np.asarray(R_spall_list, dtype=np.float64).reshape(-1)
    gt = np.asarray(GT_DI_list, dtype=np.float64).reshape(-1)

    if not (len(r_c) == len(r_s) == len(gt)):
        raise ValueError(
            f"Length mismatch: R_crack={len(r_c)}, R_spall={len(r_s)}, GT_DI={len(gt)}"
        )
    if len(gt) == 0:
        raise ValueError("Empty feature lists")

    loss_name = str(loss).strip().lower()
    if loss_name not in ("mse", "smooth_l1", "huber"):
        raise ValueError(f"Unknown loss={loss!r}; use 'mse' or 'smooth_l1'")

    def objective(params: np.ndarray) -> float:
        alpha, beta = float(params[0]), float(params[1])
        pred = predict_EADI(r_c, r_s, alpha, beta)
        if loss_name == "mse":
            return float(np.mean((pred - gt) ** 2))
        return _smooth_l1(pred, gt)

    # Non-negativity keeps the radicand non-negative for non-negative R_*.
    bounds = [(0.0, None), (0.0, None)]
    result = minimize(
        objective,
        x0=np.asarray(x0, dtype=np.float64),
        method=method,
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    alpha_opt = float(max(0.0, result.x[0]))
    beta_opt = float(max(0.0, result.x[1]))
    pred = predict_EADI(r_c, r_s, alpha_opt, beta_opt)

    mse = float(mean_squared_error(gt, pred))
    s_l1 = _smooth_l1(pred, gt)
    # pearsonr needs variance; guard constant vectors.
    if np.std(gt) < 1e-12 or np.std(pred) < 1e-12:
        pearson = float("nan")
    else:
        pearson = float(pearsonr(gt, pred).statistic)
    r2 = float(r2_score(gt, pred))

    out = {
        "alpha": alpha_opt,
        "beta": beta_opt,
        "mse": mse,
        "smooth_l1": s_l1,
        "pearson_r": pearson,
        "r2": r2,
        "success": bool(result.success),
        "message": str(result.message),
        "n": int(len(gt)),
        "loss": loss_name,
        "method": method,
    }

    if verbose:
        print("=" * 60)
        print("[optimize_EADI] best parameters")
        print(f"  alpha = {alpha_opt:.6f}")
        print(f"  beta  = {beta_opt:.6f}")
        print(f"  n     = {out['n']}   loss={loss_name}   opt={method}")
        print(f"  MSE   = {mse:.6f}")
        print(f"  SmoothL1 = {s_l1:.6f}")
        print(f"  Pearson r = {pearson:.6f}")
        print(f"  R^2   = {r2:.6f}")
        print(f"  success={result.success}  msg={result.message}")
        print("=" * 60)

    if scatter_path:
        _save_scatter(gt, pred, scatter_path, out)

    return out


def _save_scatter(gt: np.ndarray, pred: np.ndarray, path: str, metrics: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    lo = float(min(gt.min(), pred.min(), 0.0))
    hi = float(max(gt.max(), pred.max(), 1.0))
    pad = 0.05 * (hi - lo + 1e-6)
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(gt, pred, s=18, alpha=0.65, edgecolors="none", c="#1f4e79")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("GT DI")
    ax.set_ylabel("Predicted EADI")
    ax.set_title(
        f"EADI fit  α={metrics['alpha']:.3f}, β={metrics['beta']:.3f}\n"
        f"R²={metrics['r2']:.3f}, r={metrics['pearson_r']:.3f}, MSE={metrics['mse']:.4f}"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[optimize_EADI] scatter → {path}")


def _synthetic_demo(seed: int = 0) -> Dict[str, float]:
    """Random features + known (α, β) → recover weights (smoke test)."""
    rng = np.random.default_rng(seed)
    n = 200
    # Spall-heavy synthetic schedule (R_spall grows; crack partially independent).
    r_spall = rng.uniform(0.0, 0.65, size=n)
    r_crack = np.clip(0.08 * (1.0 - r_spall) + rng.normal(0, 0.02, size=n), 0.0, None)
    true_alpha, true_beta = 1.5, 3.0
    gt = predict_EADI(r_crack, r_spall, true_alpha, true_beta)
    gt = gt + rng.normal(0.0, 0.03, size=n)  # observation noise

    print(
        f"[demo] synthetic n={n} true_alpha={true_alpha} true_beta={true_beta} "
        f"(noise σ=0.03)"
    )
    return optimize_EADI_parameters(
        r_crack,
        r_spall,
        gt,
        loss="mse",
        x0=(1.0, 2.0),
        scatter_path="results/eadi_opt/demo_scatter.png",
    )


def _collect_real_features(
    image_dir: str,
    label_csv: str,
    *,
    mu_grid: float,
    min_spall_area: float,
    max_images: int = 0,
    stratified: bool = False,
    n_bins: int = 10,
    extractor_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Run DamageFeatureExtractor; return R_crack, R_spall, GT DI."""
    import pandas as pd
    from scripts.extract_damage_features import DamageFeatureExtractor
    from tqdm import tqdm

    df = pd.read_csv(label_csv)
    if "filename" not in df.columns or "DI" not in df.columns:
        raise ValueError(f"label.csv needs filename,DI columns; got {list(df.columns)}")

    kw = dict(extractor_kwargs or {})
    kw.setdefault("mu_grid", mu_grid)
    kw.setdefault("min_spall_area", min_spall_area)
    kw.setdefault("alpha", 1.0)
    kw.setdefault("beta", 2.0)
    ex = DamageFeatureExtractor(**kw)

    # Keep only rows whose image exists.
    df = df.copy()
    df["path"] = df["filename"].map(lambda f: os.path.join(image_dir, f))
    df = df[df["path"].map(os.path.isfile)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No images found under {image_dir} matching {label_csv}")

    if stratified and max_images > 0 and len(df) > max_images:
        # Sample evenly across DI quantiles for representative tuning.
        df = df.sort_values("DI").reset_index(drop=True)
        bins = np.array_split(np.arange(len(df)), n_bins)
        per = max(1, max_images // n_bins)
        pick = []
        for b in bins:
            if len(b) == 0:
                continue
            take = min(per, len(b))
            choose = np.linspace(0, len(b) - 1, take).astype(int)
            pick.extend(b[choose].tolist())
        df = df.iloc[sorted(set(pick))].reset_index(drop=True)
    elif max_images > 0:
        df = df.iloc[:max_images].reset_index(drop=True)

    r_c, r_s, gt, names = [], [], [], []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="extract features"):
        filename = getattr(row, "filename")
        di = float(getattr(row, "DI"))
        path = getattr(row, "path")
        out = ex.evaluate(path)
        r_c.append(out["R_crack"])
        r_s.append(out["R_spall"])
        gt.append(di)
        names.append(filename)

    return (
        np.asarray(r_c, dtype=np.float64),
        np.asarray(r_s, dtype=np.float64),
        np.asarray(gt, dtype=np.float64),
        names,
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Optimize EADI alpha/beta vs GT DI")
    p.add_argument("--demo", action="store_true", help="Run synthetic recovery demo")
    p.add_argument("--real", action="store_true", help="Extract features from images + label.csv")
    p.add_argument(
        "--image-dir",
        type=str,
        default="Data/DamagePattern/images",
    )
    p.add_argument("--label-csv", type=str, default="Data/DamagePattern/label.csv")
    p.add_argument("--mu-grid", type=float, default=0.11)
    p.add_argument("--min-spall-area", type=float, default=200.0)
    p.add_argument("--loss", type=str, default="mse", choices=("mse", "smooth_l1"))
    p.add_argument("--max-images", type=int, default=0, help="0 = all labeled images that exist")
    p.add_argument("--stratified", action="store_true", help="DI-stratified subsample when max-images>0")
    p.add_argument("--scatter", type=str, default="results/eadi_opt/scatter.png")
    p.add_argument("--save-json", type=str, default="results/eadi_opt/best_params.json")
    p.add_argument("--alpha0", type=float, default=1.0)
    p.add_argument("--beta0", type=float, default=2.0)
    p.add_argument("--fill-below-top", action="store_true", default=True)
    p.add_argument("--no-fill-below-top", action="store_true")
    p.add_argument("--use-object-mask", action="store_true", default=True)
    p.add_argument("--no-object-mask", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    if not args.demo and not args.real:
        args.demo = True

    if args.demo:
        metrics = _synthetic_demo()
    else:
        ex_kw = {
            "fill_below_spall_top": (not args.no_fill_below_top),
            "use_object_mask": (not args.no_object_mask),
            "use_clahe": True,
        }
        r_c, r_s, gt, names = _collect_real_features(
            args.image_dir,
            args.label_csv,
            mu_grid=args.mu_grid,
            min_spall_area=args.min_spall_area,
            max_images=args.max_images,
            stratified=args.stratified,
            extractor_kwargs=ex_kw,
        )
        print(
            f"[real] n={len(gt)}  R_crack[{r_c.min():.3f},{r_c.max():.3f}]  "
            f"R_spall[{r_s.min():.3f},{r_s.max():.3f}]  GT_DI[{gt.min():.3f},{gt.max():.3f}]"
        )
        metrics = optimize_EADI_parameters(
            r_c,
            r_s,
            gt,
            loss=args.loss,
            x0=(args.alpha0, args.beta0),
            scatter_path=args.scatter,
        )
        metrics["n_files"] = len(names)
        metrics["mu_grid"] = float(args.mu_grid)
        metrics["min_spall_area"] = float(args.min_spall_area)
        metrics["image_dir"] = args.image_dir
        metrics["fill_below_spall_top"] = not args.no_fill_below_top

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[optimize_EADI] saved → {args.save_json}")


if __name__ == "__main__":
    main()
