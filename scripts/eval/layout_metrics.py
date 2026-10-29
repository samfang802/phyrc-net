#!/usr/bin/env python3
"""Layout-only metrics (no rendering) for a layout npz (src/data/layout_io.py), vs AnyLabel truth at 128.

Layout → 128 mask exactly as the renderer input (bilinear, >= 0.5). Reported per split (test = CTR1,
val = R307; the split is read from the npz or --split), on all frames and the high-DI tertile (split DI >= 2/3
quantile, as the other evals):
  area / bias   spall fraction, signed bias vs AnyLabel (paired per frame; mean over K first); also per drift level
  iou           IoU of the layout with the truth (frames whose truth has spall; mean over K)
  n_comp        8-connected components >= min_comp px; largest = largest / total area (frames with >= min_area px)
  mono_pairs    consecutive frames of one (exp, face) in timestep order, same sample k: share of pairs whose area drops
                by > 0.01; mono_cells = mean fraction of 32x32 cells that go on -> off between consecutive frames
  diversity     mean pairwise IoU among the K samples of one condition (frames whose truth has spall)
SE = block jackknife over (exp, face, drift level).

  python scripts/eval/layout_metrics.py --layouts L.npz [--layouts M.npz ...] --out out.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import warnings
import os
import sys

import numpy as np
from scipy import ndimage

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
from src.data import layout_io  # noqa: E402
from src.metrics.dino_mmd import frame_block  # noqa: E402
from src.metrics.samples_io import split_rows  # noqa: E402

_EIGHT = np.ones((3, 3), int)


def _cc(m, min_comp):
    lab, n = ndimage.label(m, structure=_EIGHT)
    if n == 0:
        return 0, np.nan, 0
    sz = np.bincount(lab.ravel())[1:]
    return int((sz >= min_comp).sum()), float(sz.max() / sz.sum()), int(sz.sum())


def _jk(x, blocks):
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    x, b = x[ok], np.asarray(blocks)[ok]
    if len(x) == 0:
        return {"value": float("nan"), "se": float("nan"), "n": 0}
    lab = np.unique(b)
    if len(lab) < 2:
        return {"value": float(x.mean()), "se": float("nan"), "n": int(len(x))}
    loo = np.array([x[b != l].mean() for l in lab])
    n = len(lab)
    return {"value": float(x.mean()), "se": float(np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum())), "n": int(len(x))}


def compute(lay, min_comp=16, min_area=50, mono_tol=0.01):
    split = lay["split"]
    rows = split_rows(split)
    files = [str(f) for f in rows["filename"]]
    li = {f: i for i, f in enumerate(lay["filename"])}
    miss = [f for f in files if f not in li]
    if miss:
        raise ValueError(f"{len(miss)} frames of split {split} missing from the layout file, e.g. {miss[:3]}")
    cov = lay["layout"][[li[f] for f in files]]  # (N,K,S,S)
    n, K = cov.shape[:2]
    mask = layout_io.coverage_to_mask128(cov)  # (N,K,128,128)
    idx, gtm, _ = layout_io.anylabel_cache()
    gt = np.stack([gtm[idx[f]] for f in files])
    blocks = np.array([frame_block(f) for f in files])
    drift = np.array([os.path.basename(f).split("_")[2] for f in files])
    di = rows["DI"].to_numpy(float)
    hi = di >= np.quantile(di, 2 / 3)

    area = mask.reshape(n, K, -1).mean(-1)  # (N,K)
    gta = gt.reshape(n, -1).mean(1)
    inter = (mask & gt[:, None]).reshape(n, K, -1).sum(-1)
    uni = (mask | gt[:, None]).reshape(n, K, -1).sum(-1)
    has = gt.reshape(n, -1).any(1)
    iou = np.where(has[:, None] & (uni > 0), inter / np.maximum(uni, 1), np.nan).mean(1)
    st = np.array([[_cc(mask[i, k], min_comp) for k in range(K)] for i in range(n)])  # (N,K,3)
    ncomp = st[..., 0].mean(1)
    big = np.where(st[..., 2] >= min_area, st[..., 1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        largest = np.nanmean(big, 1)
    gst = np.array([_cc(gt[i], min_comp) for i in range(n)])
    g_largest = np.where(gst[:, 2] >= min_area, gst[:, 1], np.nan)
    # diversity
    div = np.full(n, np.nan)
    if K > 1:
        for i in np.where(has)[0]:
            v = []
            for a, b in itertools.combinations(range(K), 2):
                u = (mask[i, a] | mask[i, b]).sum()
                v.append(1.0 if u == 0 else (mask[i, a] & mask[i, b]).sum() / u)
            div[i] = np.mean(v)
    # monotonicity inside (exp, face) sequences, timestep order, same k
    seq = np.array(["_".join(os.path.basename(f).split("_")[:2]) for f in files])
    ts = rows["timestep"].to_numpy()
    cell = coverage_cells(mask)  # (N,K,32,32) bool
    drops, cells_off, npairs = 0, [], 0
    for s in sorted(set(seq)):
        ii = np.where(seq == s)[0]
        ii = ii[np.argsort(ts[ii], kind="stable")]
        for a, b in zip(ii[:-1], ii[1:]):
            drops += int((area[b] < area[a] - mono_tol).sum())
            npairs += K
            cells_off.append((cell[a] & ~cell[b]).reshape(K, -1).mean(-1).mean())
    mono = {"pairs": npairs, "area_drop_share": drops / max(npairs, 1), "area_drop_count": drops,
            "cells_on_to_off": float(np.mean(cells_off))}
    gtc = coverage_cells(gt)
    gcells = []
    gdrops = 0
    gp = 0
    for s in sorted(set(seq)):
        ii = np.where(seq == s)[0]
        ii = ii[np.argsort(ts[ii], kind="stable")]
        for a, b in zip(ii[:-1], ii[1:]):
            gdrops += int(gta[b] < gta[a] - mono_tol)
            gp += 1
            gcells.append((gtc[a] & ~gtc[b]).mean())
    mono_gt = {"pairs": gp, "area_drop_share": gdrops / max(gp, 1), "cells_on_to_off": float(np.mean(gcells))}

    def summ(m):
        b = blocks[m]
        return {
            "n": int(m.sum()),
            "gt_area": _jk(gta[m], b),
            "area": _jk(area.mean(1)[m], b),
            "bias": _jk(area.mean(1)[m] - gta[m], b),
            "abs_bias": _jk(np.abs(area.mean(1)[m] - gta[m]), b),
            "iou": _jk(iou[m], b),
            "n_comp": _jk(ncomp[m], b),
            "gt_n_comp": _jk(gst[m, 0], b),
            "largest": _jk(largest[m], b),
            "gt_largest": _jk(g_largest[m], b),
            "diversity_iou": _jk(div[m], b),
        }

    by_drift = {d: {"n": int((drift == d).sum()), "gt_area": float(gta[drift == d].mean()),
                    "area": float(area.mean(1)[drift == d].mean()),
                    "bias": float((area.mean(1) - gta)[drift == d].mean())} for d in sorted(set(drift))}
    return {"method": lay["method"], "split": split, "n": n, "K": K, "layout_S": int(lay["layout"].shape[-1]),
            "all": summ(np.ones(n, bool)), "highDI": summ(hi), "by_drift": by_drift,
            "monotone": mono, "monotone_gt": mono_gt}


def coverage_cells(mask, g=32):
    """bool (...,128,128) → bool (...,g,g): cell covered >= 0.5."""
    s = mask.shape[-1] // g
    sh = mask.shape[:-2] + (g, s, g, s)
    return mask.reshape(sh).mean((-3, -1)) >= 0.5


def _print(r):
    for tag in ("all", "highDI"):
        s = r[tag]
        f = lambda k: f"{s[k]['value']:.3f}"
        print(f"[{r['method']}|{r['split']}|{tag}] n={s['n']} area {f('area')} (gt {f('gt_area')}) bias {s['bias']['value']:+.3f}"
              f"±{s['bias']['se']:.3f} IoU {f('iou')} n_comp {f('n_comp')} (gt {f('gt_n_comp')}) largest {f('largest')} "
              f"(gt {f('gt_largest')}) div {f('diversity_iou')}")
    m, g = r["monotone"], r["monotone_gt"]
    print(f"    mono: area-drop pairs {m['area_drop_share']:.3f} (gt {g['area_drop_share']:.3f}); cells on->off "
          f"{m['cells_on_to_off']:.4f} (gt {g['cells_on_to_off']:.4f})")
    print("    drift  n   gt     area   bias   " + "  ".join(f"{d}:{v['gt_area']:.2f}/{v['area']:.2f}" for d, v in r["by_drift"].items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layouts", action="append", required=True)
    p.add_argument("--split", default="")
    p.add_argument("--min_comp", type=int, default=16)
    p.add_argument("--min_area", type=int, default=50)
    p.add_argument("--out", required=True)
    cli = p.parse_args()
    res = {}
    for path in cli.layouts:
        lay = layout_io.load_layout_npz(path)
        if cli.split:
            lay["split"] = cli.split
        r = compute(lay, cli.min_comp, cli.min_area)
        r["path"] = os.path.abspath(path)
        res[f"{lay['method']}|{lay['split']}"] = r
        _print(r)
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    json.dump(res, open(cli.out, "w"), indent=1)
    print("->", cli.out)


if __name__ == "__main__":
    main()
