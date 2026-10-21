#!/usr/bin/env python3
"""What kind of damage the generators draw: interpretable pixel statistics (src/metrics/damage_texture.py).

The DINOv3 metrics (eval_dinov3_kid.py) say how far a model is from GT, not why. This script puts
numbers on the three visual failure modes:

  texture   per DI tertile, model vs GT: damage area, share of the damage that is white blobs
            (white_share, white_blobs, blob_size), share that is pitch-black (black_share), share that is
            gray haze (fog_share), and the fine-texture energy inside the damage (E_damage).
            `diff` = model − GT paired per condition. SE = leave-one-block-out jackknife (block = one
            face at one drift level). Reference rows, matched by the test DI tertile edges: the training
            columns (the style the generators were taught) and the other real column (--other_split).
            Shares need a damaged region, so only the high tertile is meaningful for them. The other
            column's low-DI "white" is the bright patch the inpainter leaves at the fixture, not spall.
  cracks    crack maps on the low/mid-DI conditions (no spall yet): tolerance F1 of each fake with its
            own GT (`f1_gt`) and the best F1 over every train real (`f1_train_max`). A model that draws a
            training column's crack layout scores high on the second and low on the first. The same two
            numbers for the GT itself and for the other real column are the baseline: how much a real,
            unseen column overlaps a training one by chance. `copy_excess` = model f1_train_max − GT's.
            `nn_specimen` = which training column the best match comes from. `before_loading` = crack
            area at the lowest drift level of each face, where the real column has almost none.
  white_vs_dino (--dino)
            within one condition, the K samples differ in how many white blobs they have; regress each
            sample's DINOv3 distance from the model's own lowest-drift centroid of that face (the
            quantity the progression metric sums) on white area, controlling for damage area and fog.
            A negative coefficient means white blobs make the image look less damaged to DINOv3.
            `explained_gap` = coef × (model white − GT white) on the high-DI conditions, next to the
            actual distance gap GT − model.

Example:
  python scripts/eval/eval_damage_texture.py --samples gn=gn.npz base=base.npz cgan=cgan.npz \
      --image_dir Data/DamagePattern_inpaint_selfcondnoDI/images --no_clahe --dino --out_dir out/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics import damage_texture as dt
from src.metrics import dino_mmd as dm

TEX_KEYS = ("damage_area", "white_share", "black_share", "white_blobs", "blob_size", "fog_share", "E_damage")


def _parse_args():
    p = argparse.ArgumentParser(description="Interpretable damage-texture metrics")
    p.add_argument("--samples", nargs="+", required=True, metavar="NAME=NPZ")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--data_path", default="Data/DamagePattern/id_list_val_R307.json")
    p.add_argument("--image_dir", default="Data/DamagePattern_inpaint_selfcondnoDI/images")
    p.add_argument("--no_clahe", action="store_true", help="Image files are already CLAHE'd (the inpainted set)")
    p.add_argument("--other_split", default="val")
    p.add_argument("--intact_max_code", type=int, default=250, help="Train frames with drift code ≤ this calibrate the thresholds")
    p.add_argument("--crack_tol", type=int, default=3)
    p.add_argument("--dino", action="store_true", help="Also run the white-blob vs DINOv3 distance analysis")
    p.add_argument("--dino_model_id", default=dm.DEFAULT_DINO)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _jk(x, blocks):
    ok = np.isfinite(x)
    if ok.sum() < 3 or len(np.unique(blocks[ok])) < 3:
        return {"value": float(np.nanmean(x)) if ok.any() else None, "se": None, "n": int(ok.sum())}
    r, _ = dm.mean_jackknife(x[ok], blocks[ok])
    return {"value": round(r["value"], 5), "se": round(r["se"], 5), "n": int(ok.sum())}


def _cond_mean(stats, G, K):
    """Per-condition mean over the K samples (nan-aware: shares are undefined without damage)."""
    with np.errstate(all="ignore"):
        return {k: np.nanmean(v.reshape(G, K), axis=1) for k, v in stats.items()}


def _texture(models, gt_stats, ref_rows, tertile, blocks):
    """ref_rows: {name: (stats, tertile)} for real columns other than GT, matched by the test DI edges."""
    out = {"gt": {}, **{r: {} for r in ref_rows}}
    names = ["low", "mid", "high"]
    for t, tn in enumerate(names):
        m = tertile == t
        out["gt"][tn] = {k: _jk(gt_stats[k][m], blocks[m]) for k in TEX_KEYS}
        for r, (st, tert) in ref_rows.items():
            mo = tert == t
            out[r][tn] = {k: {"value": round(float(np.nanmean(st[k][mo])), 5) if mo.any() else None, "n": int(mo.sum())} for k in TEX_KEYS}
    for name, st in models.items():
        out[name] = {}
        for t, tn in enumerate(names):
            m = tertile == t
            out[name][tn] = {
                k: {"model": _jk(st[k][m], blocks[m]), "diff": _jk(st[k][m] - gt_stats[k][m], blocks[m])} for k in TEX_KEYS
            }
    return out


def _cracks(models, gt_u8, train_u8, train_spec, other_u8, keep, blocks, tol):
    train_c = dt.crack_map(train_u8)

    def score(maps_by_k, own):
        # maps_by_k: list over K of (n,H,W) maps for the kept conditions.
        f_gt, f_tr, spec = [], [], []
        for c in maps_by_k:
            f_gt.append(np.diag(dt.crack_f1(c, own, tol)))
            F = dt.crack_f1(c, train_c, tol)
            f_tr.append(F.max(1))
            spec.append(train_spec[F.argmax(1)])
        f_gt, f_tr = np.mean(f_gt, 0), np.mean(f_tr, 0)
        sp = pd.Series(np.concatenate(spec)).value_counts(normalize=True).round(3).to_dict()
        return f_gt, f_tr, sp

    gt_c = dt.crack_map(gt_u8[keep])
    b = blocks[keep]
    _, g_tr, g_sp = score([gt_c], gt_c)
    per_cond = {"gt": g_tr}
    out = {"n_conditions": int(keep.sum()), "gt": {"f1_train_max": _jk(g_tr, b), "nn_specimen": g_sp,
                                                    "crack_area": _jk(gt_c.mean((1, 2)), b)}}
    if other_u8 is not None:
        oc = dt.crack_map(other_u8)
        F = dt.crack_f1(oc, train_c, tol)
        out["other_real"] = {"f1_train_max": round(float(F.max(1).mean()), 5),
                             "nn_specimen": pd.Series(train_spec[F.argmax(1)]).value_counts(normalize=True).round(3).to_dict(),
                             "crack_area": round(float(oc.mean()), 5), "n": int(len(oc))}
    for name, fake in models.items():
        maps = [dt.crack_map(fake[keep, k]) for k in range(fake.shape[1])]
        f_gt, f_tr, sp = score(maps, gt_c)
        per_cond[name] = f_tr
        out[name] = {
            "f1_gt": _jk(f_gt, b),
            "f1_train_max": _jk(f_tr, b),
            "copy_excess": _jk(f_tr - g_tr, b),
            "nn_specimen": sp,
            "crack_area": _jk(np.mean([m.mean((1, 2)) for m in maps], 0), b),
            "crack_area_diff": _jk(np.mean([m.mean((1, 2)) for m in maps], 0) - gt_c.mean((1, 2)), b),
        }
    return out, per_cond, train_c


# --------------------------------------------------------------------------- composite score

# (name, label, source). Shares are only defined on the high-DI tertile; cracks are judged before spalling.
COMPONENTS = (
    ("damage_area", "Damage area (high DI)", "high"),
    ("fog_share", "Gray-haze share (high DI)", "high"),
    ("E_damage", "Fine texture in damage (high DI)", "high"),
    ("white_share", "White-blob share (high DI)", "high"),
    ("early_cracks", "Cracks before loading", "first"),
    ("copy", "Overlap with training cracks (low/mid DI)", "lowmid"),
)


def _column_values(train_stats, train_tert, train_spec, train_names, train_c, low_edge, tol):
    """Each training column's value of every component, measured like the test column's."""
    groups, levels, _ = dm.drift_levels(train_names)
    di_low = np.asarray(low_edge)
    out = {}
    for j in np.unique(train_spec):
        mj = train_spec == j
        row = {k: float(np.nanmean(train_stats[k][mj & (train_tert == 2)])) if np.any(mj & (train_tert == 2)) else np.nan
               for k, _, src in COMPONENTS if src == "high"}
        first = np.zeros(len(train_names), bool)
        for g in np.unique(groups[mj]):
            m = groups == g
            first |= m & (levels == levels[m].min())
        row["early_cracks"] = float(train_c[first].mean())
        # Leave-one-column-out: overlap of this column's early frames with the other training columns.
        sel = mj & di_low
        row["copy"] = float(dt.crack_f1(train_c[sel], train_c[~mj], tol).max(1).mean()) if sel.any() and (~mj).any() else np.nan
        out[str(j)] = row
    return out


def _composite(per_cond, columns, blocks, names):
    """Signed z per component = (entity − GT) / spread of the training columns; score = RMS of the z's.

    per_cond: {component: {entity: (G,) array, nan outside the component's conditions}}, entity 'gt' included.
    SE = leave-one-test-block-out jackknife (the training-column spread stays fixed).
    """
    keys = [k for k, _, _ in COMPONENTS]
    col = np.array([[columns[c][k] for k in keys] for c in columns])
    scale = np.maximum(np.nanstd(col, axis=0, ddof=1), 1e-9)

    def zvec(entity, keep):
        with np.errstate(all="ignore"):
            gt = np.array([np.nanmean(per_cond[k]["gt"][keep]) for k in keys])
            val = np.array([np.nanmean(per_cond[k][entity][keep]) for k in keys])
        return (val - gt) / scale

    def rms(z):
        return float(np.sqrt(np.nanmean(z**2)))

    labels = np.unique(blocks)
    full = np.ones(len(blocks), bool)
    out = {"scale_from_training_columns": dict(zip(keys, scale.round(6).tolist())), "components": keys, "models": {}, "columns": {}}
    loo = {}
    for e in names:
        z = zvec(e, full)
        zl = np.array([zvec(e, blocks != b) for b in labels])
        sl = np.array([rms(v) for v in zl])
        n = len(labels)
        jse = lambda a: float(np.sqrt((n - 1) / n * np.nansum((a - np.nanmean(a, 0)) ** 2, 0))) if a.ndim == 1 else np.sqrt((n - 1) / n * np.nansum((a - np.nanmean(a, 0)) ** 2, 0))
        out["models"][e] = {"score": {"value": round(rms(z), 4), "se": round(jse(sl), 4)},
                            "z": dict(zip(keys, z.round(3).tolist())), "z_se": dict(zip(keys, jse(zl).round(3).tolist()))}
        loo[e] = sl
    # Real training columns scored the same way against the test column: what "a real column" reaches.
    with np.errstate(all="ignore"):
        gt = np.array([np.nanmean(per_cond[k]["gt"]) for k in keys])
    for c in columns:
        z = (np.array([columns[c][k] for k in keys]) - gt) / scale
        out["columns"][c] = {"score": round(rms(z), 4), "z": dict(zip(keys, z.round(3).tolist()))}
    out["real_column_median_score"] = round(float(np.median([v["score"] for v in out["columns"].values()])), 4)
    out["comparisons"] = {}
    n = len(labels)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            d = loo[a] - loo[b]
            se = float(np.sqrt((n - 1) / n * np.sum((d - d.mean()) ** 2)))
            diff = out["models"][a]["score"]["value"] - out["models"][b]["score"]["value"]
            out["comparisons"][f"{a}_vs_{b}"] = {"diff": round(diff, 4), "se": round(se, 4), "z": round(diff / max(se, 1e-12), 2)}
    return out


def _plot_composite(comp, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    markers = ["o", "s", "D"]
    names = list(comp["models"])
    keys = comp["components"]
    labels = [lab for _, lab, _ in COMPONENTS]
    cols = comp["columns"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw=dict(width_ratios=[2.3, 1], wspace=0.45), facecolor=surf)
    for a in (ax, bx):
        a.set_facecolor(surf)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.spines["left"].set_color(grid)
        a.spines["bottom"].set_color(grid)
        a.tick_params(colors=ink2, labelsize=9)
    y = np.arange(len(keys))[::-1]
    # Spread of the real training columns around the test column.
    for yi, k in zip(y, keys):
        zs = [cols[c]["z"][k] for c in cols if np.isfinite(cols[c]["z"][k])]
        if zs:
            ax.plot([min(zs), max(zs)], [yi, yi], color=grid, lw=9, solid_capstyle="round", zorder=1)
    ax.axvline(0, color=ink2, lw=1, zorder=2)
    off = np.linspace(-0.22, 0.22, len(names))
    for i, n_ in enumerate(names):
        z = np.array([comp["models"][n_]["z"][k] for k in keys])
        ax.scatter(z, y + off[i], s=46, color=colors[i % 3], marker=markers[i % 3], edgecolor=surf, linewidth=1.5, zorder=3, label=n_)
    ax.set_yticks(y, labels, color=ink)
    ax.set_xlabel("Deviation from GT (training-column SDs)  ← too little · too much →", color=ink2, fontsize=9)
    ax.grid(axis="x", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.plot([], [], color=grid, lw=9, label="real training columns")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=ink)
    ax.set_title("Components", color=ink, fontsize=11, loc="left")
    fig.suptitle("Damage texture deviation from GT", color=ink, fontsize=13, x=0.02, ha="left", y=1.03)
    # Overall score.
    sc = [comp["models"][n_]["score"]["value"] for n_ in names]
    se = [comp["models"][n_]["score"]["se"] for n_ in names]
    yb = np.arange(len(names))[::-1]
    bx.barh(yb, sc, xerr=se, height=0.5, color=[colors[i % 3] for i in range(len(names))], error_kw=dict(ecolor=ink2, lw=1, capsize=3))
    ref = comp["real_column_median_score"]
    bx.axvline(ref, color=ink2, lw=1, ls="--")
    xmax = max(max(sc) + max(se), ref) * 1.3
    bx.text(ref + 0.02 * xmax, -0.55, f"real column (median) {ref:.2f}", color=ink2, fontsize=8, va="center")
    for yi, v, e in zip(yb, sc, se):
        bx.text(v + e + 0.03 * xmax, yi, f"{v:.2f}", va="center", color=ink, fontsize=9)
    bx.set_yticks(yb, names, color=ink)
    bx.set_xlabel("RMS z  (lower = closer to GT)", color=ink2, fontsize=9)
    bx.set_xlim(0, xmax)
    bx.set_ylim(-0.8, len(names) - 0.5)
    bx.grid(axis="x", color=grid, lw=0.8)
    bx.set_axisbelow(True)
    bx.set_title("Overall score", color=ink, fontsize=11, loc="left")
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=surf)
    plt.close(fig)


def _white_vs_dino(name, fake, stats, gt_feat, gt_stats, filenames, blocks, high, fx):
    G, K = fake.shape[:2]
    f = fx(fake.reshape(-1, *fake.shape[2:]))[0].reshape(G, K, -1)
    groups, levels, _ = dm.drift_levels(filenames)
    dist = np.zeros((G, K))
    gt_dist = np.zeros(G)
    for g in np.unique(groups):
        m = groups == g
        base = m & (levels == levels[m].min())
        dist[m] = np.linalg.norm(f[m] - f[base].reshape(-1, f.shape[-1]).mean(0), axis=-1)
        gt_dist[m] = np.linalg.norm(gt_feat[m] - gt_feat[base].mean(0), axis=-1)
    X = np.stack([stats["white"], stats["damage_area"], stats["fog"]], -1).reshape(G, K, 3)
    y = dist

    def fit(sel):
        # Within-condition regression: demean every variable per condition, then OLS on the pooled residuals.
        Xs = (X[sel] - X[sel].mean(1, keepdims=True)).reshape(-1, 3)
        ys = (y[sel] - y[sel].mean(1, keepdims=True)).reshape(-1)
        return np.linalg.lstsq(Xs, ys, rcond=None)[0]

    sel = high.copy()
    coef = fit(sel)
    labels = np.unique(blocks[sel])
    loo = np.array([fit(sel & (blocks != b_)) for b_ in labels])
    n = len(labels)
    se = np.sqrt((n - 1) / n * ((loo - loo.mean(0)) ** 2).sum(0))
    white_m = stats["white"].reshape(G, K).mean(1)[high]
    white_gt = gt_stats["white"][high]
    return {
        "conditions": "high DI tertile",
        "coef_white": {"value": round(float(coef[0]), 4), "se": round(float(se[0]), 4), "z": round(float(coef[0] / max(se[0], 1e-12)), 2)},
        "coef_damage_area": {"value": round(float(coef[1]), 4), "se": round(float(se[1]), 4)},
        "coef_fog": {"value": round(float(coef[2]), 4), "se": round(float(se[2]), 4)},
        "mean_dist_model": round(float(dist[high].mean()), 4),
        "mean_dist_gt": round(float(gt_dist[high].mean()), 4),
        "gap_gt_minus_model": round(float(gt_dist[high].mean() - dist[high].mean()), 4),
        "white_model": round(float(white_m.mean()), 5),
        "white_gt": round(float(white_gt.mean()), 5),
        "explained_gap": round(float(-coef[0] * (white_m.mean() - white_gt.mean())), 4),
        "within_condition_sd_white": round(float(stats["white"].reshape(G, K)[high].std(1).mean()), 5),
    }


def main():
    cli = _parse_args()
    os.makedirs(cli.out_dir, exist_ok=True)
    with open(cli.data_path) as f:
        ids = json.load(f)
    lab = pd.read_csv("Data/DamagePattern/label.csv").set_index("filename")
    clahe = not cli.no_clahe

    fakes, filenames = {}, None
    for spec in cli.samples:
        name, path = spec.split("=", 1)
        z = np.load(path, allow_pickle=False)
        fn = [str(n) for n in z["filename"]]
        if filenames is None:
            filenames = fn
        elif fn != filenames:
            raise ValueError(f"{name}: npz rows differ from the first npz")
        fakes[name] = z["fake"]
    filenames = np.asarray(filenames)
    G = len(filenames)
    di = lab.loc[filenames, "DI"].to_numpy(np.float64)
    edges = np.quantile(di, [1 / 3, 2 / 3])
    tertile = np.digitize(di, edges)
    blocks = np.asarray([dm.frame_block(n) for n in filenames])
    gt_u8 = dm.load_real_uint8(list(filenames), cli.image_dir, use_clahe=clahe)

    train = list(ids["train"])
    codes = np.array([int(n.split("_")[2]) for n in train])
    train_u8 = dm.load_real_uint8(train, cli.image_dir, use_clahe=clahe)
    thr = dt.calibrate(train_u8[codes <= cli.intact_max_code])
    print(f"[tex] thresholds from {thr['n_intact']} intact train frames: {thr}", flush=True)

    gt_stats = dt.texture_stats(gt_u8, thr)
    # Training columns = the style the generators were taught; the other column = an unseen real one.
    ref_rows = {"train_real": (dt.texture_stats(train_u8, thr), np.digitize(lab.loc[train, "DI"].to_numpy(np.float64), edges))}
    other_u8 = None
    if cli.other_split:
        other = list(ids[cli.other_split])
        other_u8 = dm.load_real_uint8(other, cli.image_dir, use_clahe=clahe)
        ref_rows["other_real"] = (dt.texture_stats(other_u8, thr), np.digitize(lab.loc[other, "DI"].to_numpy(np.float64), edges))

    model_stats, model_raw = {}, {}
    for name, fake in fakes.items():
        K = fake.shape[1]
        raw = dt.texture_stats(fake.reshape(-1, *fake.shape[2:]), thr)
        model_raw[name] = raw
        model_stats[name] = _cond_mean(raw, G, K)
        print(f"[tex] {name}: K={K}", flush=True)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(cli),
        "thresholds": thr,
        "di_tertile_edges": edges.tolist(),
        "texture": _texture(model_stats, gt_stats, ref_rows, tertile, blocks),
    }

    keep = tertile < 2
    train_spec = np.asarray([n.split("_")[0] for n in train])
    other_low = None
    if other_u8 is not None:
        other_low = other_u8[lab.loc[list(ids[cli.other_split]), "DI"].to_numpy() < edges[1]]
    report["cracks"], copy_cond, train_c = _cracks(fakes, gt_u8, train_u8, train_spec, other_low, keep, blocks, cli.crack_tol)
    groups, levels, _ = dm.drift_levels(filenames)
    first = np.zeros(G, bool)
    for g in np.unique(groups):
        m = groups == g
        first |= m & (levels == levels[m].min())
    early = {"gt": dt.crack_map(gt_u8[first]).mean((1, 2))}
    for name, f in fakes.items():
        early[name] = np.mean([dt.crack_map(f[first, k]).mean((1, 2)) for k in range(f.shape[1])], 0)
    report["cracks"]["before_loading"] = {"n_conditions": int(first.sum()), **{e: round(float(v.mean()), 5) for e, v in early.items()}}

    # Composite: every component as a (G,) per-condition array, nan outside the conditions it is defined on.
    def spread(mask, values):
        a = np.full(G, np.nan)
        a[mask] = values
        return a

    per_cond = {}
    for k, _, src in COMPONENTS:
        if src == "high":
            per_cond[k] = {"gt": np.where(tertile == 2, gt_stats[k], np.nan),
                           **{n_: np.where(tertile == 2, model_stats[n_][k], np.nan) for n_ in fakes}}
    per_cond["early_cracks"] = {e: spread(first, v) for e, v in early.items()}
    per_cond["copy"] = {e: spread(keep, v) for e, v in copy_cond.items()}
    train_di = lab.loc[train, "DI"].to_numpy(np.float64)
    columns = _column_values(ref_rows["train_real"][0], ref_rows["train_real"][1], train_spec, train, train_c,
                             train_di < edges[1], cli.crack_tol)
    report["composite"] = _composite(per_cond, columns, blocks, list(fakes))
    _plot_composite(report["composite"], os.path.join(cli.out_dir, "damage_texture_score.png"))

    if cli.dino:
        fx = dm.DinoFeatures(cli.dino_model_id, device=cli.device)
        gt_feat = fx(gt_u8)[0]
        report["white_vs_dino"] = {
            name: _white_vs_dino(name, fakes[name], model_raw[name], gt_feat, gt_stats, filenames, blocks, tertile == 2, fx)
            for name, fake in fakes.items()
            if fake.shape[1] > 1
        }

    path = os.path.join(cli.out_dir, "damage_texture.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    def v(d):
        return "  -  " if d["value"] is None else (f"{d['value']:.3f}" if d["se"] is None else f"{d['value']:.3f}±{d['se']:.3f}")

    tex = report["texture"]
    for tn in ("low", "mid", "high"):
        print(f"\n[tex] DI tertile {tn}")
        print("      " + " ".join(f"{k:>18s}" for k in TEX_KEYS))
        print(f"{'gt':6s}" + " ".join(f"{v(tex['gt'][tn][k]):>18s}" for k in TEX_KEYS))
        for r in ref_rows:
            print(f"{r[:6]:6s}" + " ".join(f"{tex[r][tn][k]['value'] if tex[r][tn][k]['value'] is None else round(tex[r][tn][k]['value'], 3)!s:>18s}" for k in TEX_KEYS))
        for name in fakes:
            print(f"{name:6s}" + " ".join(f"{v(tex[name][tn][k]['model']):>18s}" for k in TEX_KEYS))
    cr = report["cracks"]
    print(f"\n[crack] low+mid DI ({cr['n_conditions']} conditions)  gt f1_train_max={v(cr['gt']['f1_train_max'])} nn={cr['gt']['nn_specimen']}")
    if "other_real" in cr:
        print(f"[crack] other real column f1_train_max={cr['other_real']['f1_train_max']:.3f} nn={cr['other_real']['nn_specimen']}")
    for name in fakes:
        c = cr[name]
        print(f"[crack] {name:6s} f1_gt={v(c['f1_gt'])} f1_train_max={v(c['f1_train_max'])} copy_excess={v(c['copy_excess'])} "
              f"crack_area_diff={v(c['crack_area_diff'])} nn={c['nn_specimen']}")
    print(f"[crack] crack area at the lowest drift level (before cracking): {cr['before_loading']}")
    for name, w in report.get("white_vs_dino", {}).items():
        print(f"[white→dino] {name}: coef_white={w['coef_white']} gap GT−model={w['gap_gt_minus_model']} "
              f"explained by white={w['explained_gap']} (white model {w['white_model']} vs gt {w['white_gt']})")
    comp = report["composite"]
    print(f"\n[score] damage texture deviation (RMS z vs GT, unit = spread of the training columns; lower is better)")
    print(f"[score] real training columns: {({c: v['score'] for c, v in comp['columns'].items()})}  median {comp['real_column_median_score']}")
    for name, r in comp["models"].items():
        print(f"[score] {name:8s} {r['score']['value']:.2f} ± {r['score']['se']:.2f}   z: {r['z']}")
    for k, c in comp["comparisons"].items():
        print(f"[score] {k}: {c['diff']:+.2f} (z = {c['z']:+.1f})")
    print(f"\n[tex] wrote {path} and damage_texture_score.png")


if __name__ == "__main__":
    main()
