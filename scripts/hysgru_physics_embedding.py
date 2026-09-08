#!/usr/bin/env python3
"""UMAP / t-SNE of HysGRU 1024-d features, color-coded by physical attributes.

Supports train / val / test, or ``all`` = train+val+test (full image set,
6 specimens: 4 train + 1 val + 1 test).

Colorings
---------
- Drift(%) at the sample timestep (from Experiment CSV)
- Cumulative dissipated energy proxy up to timestep (∑ |F Δu|)
- Aspect ratio = lMeasured / depth
- failureType (Flexure / Flexure-Shear / Shear)
- DI (reference)
- Specimen id / split
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import tempfile
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

from argument.argument_mixed import set_dataset_args
from src.dataset import PatternLoopDataset
from src.utils import build_damage_image_transform_with_loss_mask
from train_mixed import _load_args_from_result_dir, _resolve_num_workers, _resolve_required_match

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def _parse_args():
    p = argparse.ArgumentParser(description="HysGRU embedding colored by physics attributes")
    p.add_argument(
        "--result_dir",
        type=str,
        default="results/DamagePatternGeneration/CFMADM/2026-07-26_21-24-55",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "val", "test", "all"),
        help="Dataset split; 'all' = train+val+test (full 6-specimen image set)",
    )
    p.add_argument("--method", type=str, default="both", choices=("umap", "tsne", "both"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--output_dir", type=str, default="")
    return p.parse_args()


def _prepare_all_split_data_path(data_path: str) -> str:
    """Write a temp id_list JSON with an 'all' key = train∪val∪test."""
    src = os.path.join("./Data/DamagePattern", data_path)
    with open(src, "r", encoding="utf-8") as f:
        splits = json.load(f)
    combined = list(splits.get("train", [])) + list(splits.get("val", [])) + list(splits.get("test", []))
    # Preserve original keys so other code paths stay valid if reused.
    out = dict(splits)
    out["all"] = combined
    fd, tmp_path = tempfile.mkstemp(prefix="id_list_all_", suffix=".json", dir="./Data/DamagePattern")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(
        f"[embed] Full image set: n={len(combined)} "
        f"(train={len(splits.get('train', []))} "
        f"val={len(splits.get('val', []))} "
        f"test={len(splits.get('test', []))})",
        flush=True,
    )
    return os.path.basename(tmp_path)


def _build_dataset(args, extractor, extractor_args, mode):
    transform = build_damage_image_transform_with_loss_mask(
        image_size=args.image_size,
        use_augmentation=False,
        output_scale=True,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
        use_grayscale_input=args.use_grayscale_input,
        use_clahe=args.use_clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid_size=args.clahe_tile_grid_size,
    )
    kwargs = dict(vars(args))
    kwargs["use_loss_mask"] = False
    kwargs["train_virtual_repeat"] = 1
    kwargs["use_damage_mask_condition"] = False
    kwargs["use_object_loss_mask"] = False
    tmp_data_path = None
    if mode == "all":
        tmp_data_path = _prepare_all_split_data_path(kwargs["data_path"])
        kwargs["data_path"] = tmp_data_path
    ds = PatternLoopDataset(
        root="./Data",
        mode=mode,
        transform=transform,
        extractor=extractor,
        extractor_args=extractor_args,
        **kwargs,
    )
    # Attach for cleanup; PatternLoopDataset does not own the temp file.
    ds._tmp_id_list_path = (
        os.path.join("./Data/DamagePattern", tmp_data_path) if tmp_data_path else None
    )
    return ds


def _cum_abs_work(disp_mm: np.ndarray, force_kn: np.ndarray) -> np.ndarray:
    """Cumulative absolute work proxy [kN·mm] as ∫|F du| along the path."""
    d_disp = np.diff(disp_mm, prepend=disp_mm[0])
    d_disp[0] = 0.0
    incr = np.abs(force_kn * d_disp)
    return np.cumsum(incr)


def _load_hysteresis_physics(exp_ids):
    """Cache per-exp arrays: drift[%], cum_energy, and specimen metadata."""
    info = pd.read_csv("./Data/Hysteresis/info_w_mander_big.csv")
    info = info.set_index("id")
    cache = {}
    for exp in exp_ids:
        exp = str(exp)
        path = os.path.join("./Data/Hysteresis/Experiment", f"{exp}.csv")
        if not os.path.isfile(path):
            path = os.path.join("./Data/Hysteresis/Simulate", f"{exp}.csv")
        df = pd.read_csv(path)
        drift = df["Drift(%)"].to_numpy(dtype=np.float64)
        disp = df["Displacement(mm)"].to_numpy(dtype=np.float64)
        force = df["Force(kN)"].to_numpy(dtype=np.float64)
        cum_e = _cum_abs_work(disp, force)
        row = info.loc[exp] if exp in info.index else None
        if row is None:
            aspect = np.nan
            failure = "Unknown"
            width = depth = length = np.nan
            axial = np.nan
        else:
            width = float(row["width"])
            depth = float(row["depth"])
            length = float(row["lMeasured"])
            axial = float(row["axialLoad"])
            # Aspect ratio ≈ shear span ratio L / section depth
            aspect = length / max(depth, 1e-8)
            failure = str(row["failureType"])
        cache[exp] = {
            "drift": drift,
            "cum_energy": cum_e,
            "aspect_ratio": aspect,
            "failureType": failure,
            "width": width,
            "depth": depth,
            "lMeasured": length,
            "axialLoad": axial,
        }
    return cache


def _filename_to_split(data_path: str) -> dict:
    """Map filename → train/val/test using the original id_list (not temp 'all')."""
    src = os.path.join("./Data/DamagePattern", data_path)
    # If data_path is a temp all-list, fall back to id_list_no_leak.json.
    if not os.path.isfile(src) or "id_list_all_" in os.path.basename(src):
        src = "./Data/DamagePattern/id_list_no_leak.json"
    with open(src, "r", encoding="utf-8") as f:
        splits = json.load(f)
    mapping = {}
    for split_name in ("train", "val", "test"):
        for fn in splits.get(split_name, []):
            mapping[str(fn)] = split_name
    return mapping


@torch.no_grad()
def _collect_features(dataset, phys_cache, split_map=None):
    records = []
    n = int(getattr(dataset, "base_size", len(dataset)))
    labels = dataset.label.reset_index(drop=True)
    for i in range(n):
        sample = dataset[i]
        _img, hidden, _direction, exp, timestep, DI, _info = sample[:7]
        e = exp
        if torch.is_tensor(e):
            e = e.item() if e.ndim == 0 else str(e)
        e = str(e)
        t = int(timestep.item() if torch.is_tensor(timestep) else timestep)
        di = float(DI.reshape(-1)[0].item() if torch.is_tensor(DI) else DI)
        g = hidden.detach().cpu().float().reshape(-1).numpy()
        phys = phys_cache[e]
        t_clip = min(max(t, 0), len(phys["drift"]) - 1)
        fn = str(labels.iloc[i]["filename"]) if i < len(labels) else ""
        split_name = (split_map or {}).get(fn, str(getattr(dataset, "mode", "unknown")))
        records.append(
            {
                "exp": e,
                "timestep": t,
                "di": di,
                "hidden": g,
                "filename": fn,
                "split": split_name,
                "drift_pct": float(phys["drift"][t_clip]),
                "cum_energy": float(phys["cum_energy"][t_clip]),
                "aspect_ratio": float(phys["aspect_ratio"]),
                "failureType": phys["failureType"],
                "axialLoad": float(phys["axialLoad"]),
            }
        )
    return records


def _embed(X, method, seed, perplexity, n_neighbors, min_dist):
    method = method.lower()
    if method == "umap":
        if not HAS_UMAP:
            raise ImportError("umap-learn is not installed")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=seed,
        )
        return reducer.fit_transform(X)
    # t-SNE
    # sklearn >=1.2 uses max_iter; older uses n_iter
    kwargs = dict(
        n_components=2,
        perplexity=min(perplexity, max(5.0, (len(X) - 1) / 3.0)),
        metric="euclidean",
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    try:
        tsne = TSNE(max_iter=1000, **kwargs)
    except TypeError:
        tsne = TSNE(n_iter=1000, **kwargs)
    return tsne.fit_transform(X)


def _plot_embeddings(xy, records, method, out_path):
    drift = np.array([r["drift_pct"] for r in records], dtype=np.float64)
    energy = np.array([r["cum_energy"] for r in records], dtype=np.float64)
    aspect = np.array([r["aspect_ratio"] for r in records], dtype=np.float64)
    di = np.array([r["di"] for r in records], dtype=np.float64)
    fail = np.array([r["failureType"] for r in records])

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5))
    axes = axes.ravel()

    # 1) Drift
    sc0 = axes[0].scatter(xy[:, 0], xy[:, 1], c=drift, s=12, cmap="viridis", alpha=0.85)
    axes[0].set_title("Color: Drift ratio [%]")
    plt.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2) Cumulative energy
    sc1 = axes[1].scatter(xy[:, 0], xy[:, 1], c=energy, s=12, cmap="magma", alpha=0.85)
    axes[1].set_title(r"Color: Cum. $|F\Delta u|$ energy [kN·mm]")
    plt.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3) Aspect ratio
    sc2 = axes[2].scatter(xy[:, 0], xy[:, 1], c=aspect, s=12, cmap="coolwarm", alpha=0.85)
    axes[2].set_title(r"Color: Aspect ratio $L/d$")
    plt.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

    cmap = plt.get_cmap("tab10")
    splits = np.array([r.get("split", "unknown") for r in records])
    uniq_splits = sorted(set(splits.tolist()))
    # Prefer split coloring when multiple splits are present (full image set).
    if len(uniq_splits) > 1:
        split_colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
        for i, cat in enumerate(uniq_splits):
            m = splits == cat
            axes[3].scatter(
                xy[m, 0],
                xy[m, 1],
                s=14,
                c=[split_colors.get(cat, cmap(i % 10))],
                label=f"{cat} (n={int(m.sum())})",
                alpha=0.85,
            )
        axes[3].set_title("Color: Data split")
        axes[3].legend(markerscale=1.4, fontsize=8, frameon=True)
    else:
        cats = sorted(set(fail.tolist()))
        for i, cat in enumerate(cats):
            m = fail == cat
            axes[3].scatter(
                xy[m, 0],
                xy[m, 1],
                s=14,
                c=[cmap(i % 10)],
                label=cat,
                alpha=0.85,
            )
        axes[3].set_title("Color: Failure mode")
        axes[3].legend(markerscale=1.4, fontsize=8, frameon=True)

    # 5) DI reference
    sc4 = axes[4].scatter(xy[:, 0], xy[:, 1], c=di, s=12, cmap="plasma", alpha=0.85)
    axes[4].set_title("Color: DI (reference)")
    plt.colorbar(sc4, ax=axes[4], fraction=0.046, pad=0.04)

    # 6) Specimen id (categorical light)
    exps = np.array([r["exp"] for r in records])
    uniq = sorted(set(exps.tolist()))
    for i, e in enumerate(uniq):
        m = exps == e
        axes[5].scatter(xy[m, 0], xy[m, 1], s=12, c=[cmap(i % 10)], label=e, alpha=0.8)
    axes[5].set_title("Color: Specimen ID")
    axes[5].legend(markerscale=1.2, fontsize=7, ncol=2, frameon=True)

    for ax in axes:
        ax.set_xlabel(f"{method.upper()}-1")
        ax.set_ylabel(f"{method.upper()}-2")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"HysGRU 1024-d → {method.upper()} 2D · N={len(records)} ({os.path.basename(out_path)})",
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[embed] Saved: {out_path}", flush=True)


def main():
    cli = _parse_args()
    result_dir = os.path.abspath(cli.result_dir)
    args = _load_args_from_result_dir(result_dir)
    args.result_dir = result_dir
    args.batch_size = cli.batch_size
    out_dir = cli.output_dir.strip() or os.path.join(result_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[embed] device={device} split={cli.split} method={cli.method} umap={HAS_UMAP}", flush=True)

    extractor_args = _load_args_from_result_dir(args.extractor_path)
    extractor = importlib.import_module("src.model.HystereticPrediction").__dict__[
        extractor_args.architecture
    ](**vars(extractor_args)).to(device)
    extractor_ckpt = _resolve_required_match(
        os.path.join(args.extractor_path, "**/best.ckpt"), "extractor checkpoint"
    )
    extractor.load_state_dict(
        torch.load(extractor_ckpt, map_location=device, weights_only=False)["state_dict"]
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False

    print(f"[embed] Building {cli.split} dataset...", flush=True)
    original_data_path = str(args.data_path)
    ds = _build_dataset(args, extractor, extractor_args, cli.split)
    from torch.utils.data import DataLoader

    dl = DataLoader(
        ds,
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=_resolve_num_workers(cli.num_workers),
    )
    set_dataset_args(args, dl)

    exp_ids = sorted(set(ds.label["exp"].astype(str).tolist()))
    print(
        f"[embed] Specimens ({len(exp_ids)}): {exp_ids}",
        flush=True,
    )
    phys_cache = _load_hysteresis_physics(exp_ids)

    print("[embed] Collecting 1024-d features...", flush=True)
    split_map = _filename_to_split(original_data_path)
    records = _collect_features(ds, phys_cache, split_map=split_map)
    X = np.stack([r["hidden"] for r in records], axis=0)
    from collections import Counter

    split_counts = dict(Counter(r["split"] for r in records))
    print(f"[embed] X shape={X.shape} split_counts={split_counts}", flush=True)
    tmp_id = getattr(ds, "_tmp_id_list_path", None)
    if tmp_id and os.path.isfile(tmp_id):
        try:
            os.remove(tmp_id)
        except OSError:
            pass

    methods = []
    if cli.method in ("umap", "both"):
        if HAS_UMAP:
            methods.append("umap")
        else:
            print("[embed] UMAP unavailable; skip (install umap-learn).", flush=True)
    if cli.method in ("tsne", "both"):
        methods.append("tsne")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split": cli.split,
        "n_samples": len(records),
        "feature_dim": int(X.shape[1]),
        "methods": methods,
        "specimens": exp_ids,
        "split_counts": split_counts,
        "colorings": [
            "drift_pct",
            "cum_energy (|F Δu|)",
            "aspect_ratio L/d",
            "split (if multi-split) / failureType",
            "DI",
            "specimen id",
        ],
        "outputs": [],
    }

    for method in methods:
        print(f"[embed] Fitting {method.upper()}...", flush=True)
        xy = _embed(
            X,
            method,
            cli.seed,
            cli.perplexity,
            cli.n_neighbors,
            cli.min_dist,
        )
        out_png = os.path.join(out_dir, f"hysgru_{method}_{cli.split}_physics_color.png")
        _plot_embeddings(xy, records, method, out_png)
        np.savez_compressed(
            os.path.join(out_dir, f"hysgru_{method}_{cli.split}_embedding.npz"),
            xy=xy,
            hidden=X,
            di=np.array([r["di"] for r in records]),
            drift_pct=np.array([r["drift_pct"] for r in records]),
            cum_energy=np.array([r["cum_energy"] for r in records]),
            aspect_ratio=np.array([r["aspect_ratio"] for r in records]),
            failureType=np.array([r["failureType"] for r in records]),
            exp=np.array([r["exp"] for r in records]),
            timestep=np.array([r["timestep"] for r in records]),
            split=np.array([r["split"] for r in records]),
        )
        summary["outputs"].append(out_png)

    summary_path = os.path.join(out_dir, f"hysgru_embedding_{cli.split}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[embed] Summary: {summary_path}", flush=True)
    print("[embed] Done.", flush=True)


if __name__ == "__main__":
    main()
