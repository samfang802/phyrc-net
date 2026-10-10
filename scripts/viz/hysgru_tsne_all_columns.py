#!/usr/bin/env python3
"""t-SNE of HysGRU 1024-d features over all 160 hysteresis columns.

Role labels (requested):
  - upstream_train  : id_list_big.json train (128)
  - upstream_val    : id_list_big.json val excluding CTR1
  - downstream_test : CTR1 (downstream damage-pattern test column;
                      note: CTR1 sits in HysGRU val in id_list_big.json)

Timesteps are evenly subsampled per column so N stays t-SNE-friendly.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from collections import Counter
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.hysteresis import HystereticDataset, _to_hidden_state_matrix
from train_mixed import _load_args_from_result_dir, _resolve_required_match


def _parse_args():
    p = argparse.ArgumentParser(description="t-SNE of all 160 HysGRU column trajectories")
    p.add_argument(
        "--result_dir",
        type=str,
        default="results/DamagePatternGeneration/CFMADM/2026-07-26_21-24-55",
        help="CFM result dir (used only for default output_dir / logging)",
    )
    p.add_argument("--extractor_path", type=str, default="./phase1_extractor")
    p.add_argument(
        "--id_list",
        type=str,
        default="./Data/Hysteresis/id_list_big.json",
        help="Upstream HysGRU train/val split",
    )
    p.add_argument(
        "--max_steps_per_exp",
        type=int,
        default=40,
        help="Evenly subsample at most this many timesteps per column",
    )
    p.add_argument("--perplexity", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument(
        "--downstream_test_exps",
        type=str,
        default="CTR1",
        help="Comma-separated specimen ids marked as downstream test",
    )
    return p.parse_args()


def _cum_abs_work(disp_mm: np.ndarray, force_kn: np.ndarray) -> np.ndarray:
    d_disp = np.diff(disp_mm, prepend=disp_mm[0])
    d_disp[0] = 0.0
    return np.cumsum(np.abs(force_kn * d_disp))


def _role_for_exp(exp: str, train_set: set, val_set: set, downstream: set) -> str:
    if exp in downstream:
        return "downstream_test"
    if exp in train_set:
        return "upstream_train"
    if exp in val_set:
        return "upstream_val"
    return "unknown"


def _even_indices(n: int, k: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    k = max(1, min(int(k), n))
    if k == n:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, num=k).round().astype(np.int64))


@torch.no_grad()
def _extract_all_hidden(extractor, extractor_args, exp_list, device):
    ds = HystereticDataset(
        root="./Data/Hysteresis",
        specified=list(exp_list),
        **vars(extractor_args),
    )
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=ds.collate_fn,
    )
    hidden = {}
    for batch in tqdm(dl, desc="HysGRU forward"):
        simulated_loop, _test_force, _olen, _maxf, exp = batch
        simulated_loop = simulated_loop.to(device)
        output, _ = extractor.L(simulated_loop)
        hidden[str(exp[0])] = _to_hidden_state_matrix(output).float().numpy()
    return hidden


def _load_physics(exp_ids):
    info = pd.read_csv("./Data/Hysteresis/info_w_mander_big.csv").set_index("id")
    cache = {}
    for exp in exp_ids:
        exp = str(exp)
        path = os.path.join("./Data/Hysteresis/Experiment", f"{exp}.csv")
        df = pd.read_csv(path)
        drift = df["Drift(%)"].to_numpy(dtype=np.float64)
        disp = df["Displacement(mm)"].to_numpy(dtype=np.float64)
        force = df["Force(kN)"].to_numpy(dtype=np.float64)
        cum_e = _cum_abs_work(disp, force)
        if exp in info.index:
            row = info.loc[exp]
            aspect = float(row["lMeasured"]) / max(float(row["depth"]), 1e-8)
            failure = str(row["failureType"])
            axial = float(row["axialLoad"])
        else:
            aspect, failure, axial = np.nan, "Unknown", np.nan
        cache[exp] = {
            "drift": drift,
            "cum_energy": cum_e,
            "aspect_ratio": aspect,
            "failureType": failure,
            "axialLoad": axial,
        }
    return cache


def _fit_tsne(X, perplexity, seed):
    kwargs = dict(
        n_components=2,
        perplexity=min(float(perplexity), max(5.0, (len(X) - 1) / 3.0)),
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


def _plot(xy, records, out_path):
    drift = np.array([r["drift_pct"] for r in records], dtype=np.float64)
    energy = np.array([r["cum_energy"] for r in records], dtype=np.float64)
    aspect = np.array([r["aspect_ratio"] for r in records], dtype=np.float64)
    fail = np.array([r["failureType"] for r in records])
    roles = np.array([r["role"] for r in records])
    exps = np.array([r["exp"] for r in records])

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.8))
    axes = axes.ravel()

    sc0 = axes[0].scatter(xy[:, 0], xy[:, 1], c=drift, s=8, cmap="viridis", alpha=0.75)
    axes[0].set_title("Color: Drift ratio [%]")
    plt.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)

    sc1 = axes[1].scatter(xy[:, 0], xy[:, 1], c=energy, s=8, cmap="magma", alpha=0.75)
    axes[1].set_title(r"Color: Cum. $|F\Delta u|$ energy [kN·mm]")
    plt.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)

    sc2 = axes[2].scatter(xy[:, 0], xy[:, 1], c=aspect, s=8, cmap="coolwarm", alpha=0.75)
    axes[2].set_title(r"Color: Aspect ratio $L/d$")
    plt.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

    # Role panel (main requested annotation)
    role_style = {
        "upstream_train": ("#1f77b4", 7, 0.55),
        "upstream_val": ("#ff7f0e", 10, 0.75),
        "downstream_test": ("#2ca02c", 22, 0.95),
        "unknown": ("#7f7f7f", 8, 0.5),
    }
    # Draw train first, then val, then CTR1 on top
    for role in ("upstream_train", "upstream_val", "downstream_test", "unknown"):
        m = roles == role
        if not m.any():
            continue
        c, s, a = role_style[role]
        n = int(m.sum())
        n_exp = len(set(exps[m].tolist()))
        axes[3].scatter(
            xy[m, 0],
            xy[m, 1],
            s=s,
            c=c,
            label=f"{role} (pts={n}, cols={n_exp})",
            alpha=a,
            edgecolors="k" if role == "downstream_test" else "none",
            linewidths=0.35 if role == "downstream_test" else 0.0,
            zorder=3 if role == "downstream_test" else 1,
        )
    axes[3].set_title("Color: Upstream train/val + downstream test (CTR1)")
    axes[3].legend(markerscale=1.3, fontsize=7.5, frameon=True, loc="best")

    cats = sorted(set(fail.tolist()))
    cmap = plt.get_cmap("tab10")
    for i, cat in enumerate(cats):
        m = fail == cat
        axes[4].scatter(xy[m, 0], xy[m, 1], s=8, c=[cmap(i % 10)], label=cat, alpha=0.7)
    axes[4].set_title("Color: Failure mode")
    axes[4].legend(markerscale=1.3, fontsize=7.5, frameon=True)

    # Highlight named columns of interest on top of light grey cloud
    axes[5].scatter(xy[:, 0], xy[:, 1], s=4, c="#d0d0d0", alpha=0.35, zorder=0)
    highlight = ["R307", "R315", "R330", "R1015", "R615", "CTR1", "COC"]
    cmap2 = plt.get_cmap("tab10")
    for i, e in enumerate(highlight):
        m = exps == e
        if not m.any():
            continue
        axes[5].scatter(
            xy[m, 0],
            xy[m, 1],
            s=14 if e == "CTR1" else 9,
            c=[cmap2(i % 10)],
            label=e,
            alpha=0.9,
            zorder=2 if e != "CTR1" else 3,
            edgecolors="k" if e == "CTR1" else "none",
            linewidths=0.3 if e == "CTR1" else 0.0,
        )
    axes[5].set_title("Highlight: damage-set columns (+COC)")
    axes[5].legend(markerscale=1.2, fontsize=7, ncol=2, frameon=True)

    for ax in axes:
        ax.set_xlabel("TSNE-1")
        ax.set_ylabel("TSNE-2")
        ax.grid(True, alpha=0.2)

    n_cols = len(set(exps.tolist()))
    fig.suptitle(
        f"HysGRU 1024-d → t-SNE · {n_cols} columns · N={len(records)}  "
        f"(upstream train/val + downstream test CTR1)",
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[tsne160] Saved: {out_path}", flush=True)


def main():
    cli = _parse_args()
    result_dir = os.path.abspath(cli.result_dir)
    out_dir = cli.output_dir.strip() or os.path.join(result_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    with open(cli.id_list, "r", encoding="utf-8") as f:
        id_splits = json.load(f)
    train_set = set(map(str, id_splits["train"]))
    val_set = set(map(str, id_splits["val"]))
    downstream = {s.strip() for s in cli.downstream_test_exps.split(",") if s.strip()}

    exp_dir = "./Data/Hysteresis/Experiment"
    all_exps = sorted(os.path.splitext(f)[0] for f in os.listdir(exp_dir) if f.endswith(".csv"))
    if len(all_exps) != 160:
        print(f"[tsne160] Warning: expected 160 Experiment CSVs, found {len(all_exps)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[tsne160] device={device} columns={len(all_exps)} "
        f"max_steps/exp={cli.max_steps_per_exp} downstream={sorted(downstream)}",
        flush=True,
    )
    print(
        f"[tsne160] upstream train={len(train_set)} val={len(val_set)} "
        f"(CTR1 in HysGRU val={('CTR1' in val_set)})",
        flush=True,
    )

    extractor_args = _load_args_from_result_dir(cli.extractor_path)
    extractor = getattr(importlib.import_module("src.model.HystereticPrediction"), extractor_args.architecture)(**vars(extractor_args)).to(device)
    ckpt = _resolve_required_match(
        os.path.join(cli.extractor_path, "**/best.ckpt"), "extractor checkpoint"
    )
    extractor.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False

    print("[tsne160] Extracting hidden states for all columns...", flush=True)
    hidden_map = _extract_all_hidden(extractor, extractor_args, all_exps, device)
    phys = _load_physics(all_exps)

    records = []
    for exp in all_exps:
        H = hidden_map[exp]  # (T, 1024)
        t_len = H.shape[0]
        idxs = _even_indices(t_len, cli.max_steps_per_exp)
        role = _role_for_exp(exp, train_set, val_set, downstream)
        p = phys[exp]
        for t in idxs.tolist():
            t_clip = min(max(t, 0), len(p["drift"]) - 1)
            records.append(
                {
                    "exp": exp,
                    "timestep": int(t),
                    "role": role,
                    "hidden": H[t],
                    "drift_pct": float(p["drift"][t_clip]),
                    "cum_energy": float(p["cum_energy"][t_clip]),
                    "aspect_ratio": float(p["aspect_ratio"]),
                    "failureType": p["failureType"],
                    "axialLoad": float(p["axialLoad"]),
                }
            )

    X = np.stack([r["hidden"] for r in records], axis=0).astype(np.float32)
    role_counts = dict(Counter(r["role"] for r in records))
    col_role = {}
    for r in records:
        col_role[r["exp"]] = r["role"]
    col_role_counts = dict(Counter(col_role.values()))
    print(
        f"[tsne160] X={X.shape} role_pts={role_counts} role_cols={col_role_counts}",
        flush=True,
    )

    print("[tsne160] Fitting t-SNE...", flush=True)
    xy = _fit_tsne(X, cli.perplexity, cli.seed)
    out_png = os.path.join(out_dir, "hysgru_tsne_all160_upstream_downstream_color.png")
    _plot(xy, records, out_png)

    np.savez_compressed(
        os.path.join(out_dir, "hysgru_tsne_all160_embedding.npz"),
        xy=xy,
        hidden=X,
        exp=np.array([r["exp"] for r in records]),
        timestep=np.array([r["timestep"] for r in records]),
        role=np.array([r["role"] for r in records]),
        drift_pct=np.array([r["drift_pct"] for r in records]),
        cum_energy=np.array([r["cum_energy"] for r in records]),
        aspect_ratio=np.array([r["aspect_ratio"] for r in records]),
        failureType=np.array([r["failureType"] for r in records]),
    )
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_columns": len(all_exps),
        "n_points": len(records),
        "max_steps_per_exp": cli.max_steps_per_exp,
        "upstream_train_cols": len(train_set),
        "upstream_val_cols": len(val_set - downstream),
        "downstream_test_cols": sorted(downstream),
        "note": (
            "CTR1 is in HysGRU id_list_big val, but labeled here as downstream_test "
            "to match the damage-pattern generation test split."
        ),
        "role_point_counts": role_counts,
        "role_column_counts": col_role_counts,
        "output_png": out_png,
    }
    summary_path = os.path.join(out_dir, "hysgru_tsne_all160_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[tsne160] Summary: {summary_path}", flush=True)
    print("[tsne160] Done.", flush=True)


if __name__ == "__main__":
    main()
