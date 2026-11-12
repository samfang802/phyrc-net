#!/usr/bin/env python3
"""Crack continuity of generated frames vs the real ones (no training, CPU only).

Per model (eval-suite npz, every draw of every frame) and for the real frames of the first npz:
  crack_px       share of crack pixels (src.metrics.damage_texture.crack_map)
  comp_per_kpx   connected crack components per 1000 crack pixels (higher = more fragmented)
  len_med/mean   component length = 4 * sqrt(largest eigenvalue of its pixel covariance) (a line of length l -> l)
  long_share     share of crack pixels in components at least --long px long
  grid_end       share of component ends (the two extreme pixels along the principal axis) within --tol px of a
                 multiple of --cell in x or y; uniform positions give 1 - (1 - (2 tol + 1) / cell)^2
  grid_dip       crack density on the --cell grid lines (coord mod cell in {cell-1, 0}) / density mid-cell
                 (coord mod cell in cell/2-2 .. cell/2+1), x and y pooled; 1 = no grid structure
rfmv2b pastes its rollout bank into S inside 16 px cells (8 x 8 grid at 128 px): if the paste borders break the
cracks, grid_end and 1 / grid_dip rise above the real frames and the other models.

  python3 scripts/eval/crack_fragments.py --model rfmv2b=results/eval_suite/rfmv2b/val/samples/x.npz ... --out f.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics import damage_texture as dt  # noqa: E402


def components(c, cell, tol, long_px):
    """c bool (H, W) -> per-image tallies."""
    lab, n = ndi.label(c, np.ones((3, 3)))
    out = {"px": int(c.sum()), "n": int(n), "lens": [], "long_px": 0, "ends": 0, "ends_grid": 0}
    if not n:
        return out
    objs = ndi.find_objects(lab)
    for k, sl in enumerate(objs, start=1):
        ys, xs = np.nonzero(lab[sl] == k)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        pts = np.stack([xs, ys], 1).astype(np.float64)
        if len(pts) < 3:
            length = float(len(pts))
            ends = pts[[0, -1]]
        else:
            cov = np.cov(pts.T)
            w, v = np.linalg.eigh(cov)
            length = 4.0 * np.sqrt(max(w[-1], 0.0))
            proj = (pts - pts.mean(0)) @ v[:, -1]
            ends = pts[[int(proj.argmin()), int(proj.argmax())]]
        out["lens"].append(length)
        if length >= long_px:
            out["long_px"] += len(pts)
        for e in ends:
            dx = min(e[0] % cell, cell - e[0] % cell)
            dy = min(e[1] % cell, cell - e[1] % cell)
            out["ends"] += 1
            out["ends_grid"] += int(dx <= tol or dy <= tol)
    return out


def bridged(frames, cm, reach=3):
    """Crack maps with the drawn grid lines filled back where a crack reaches them (within ``reach`` px), so a
    crack crossing a grid line stays one component (crack_map cuts it there)."""
    x = np.asarray(frames, dtype=np.float32)
    g = dt._grid_mask((ndi.grey_closing(x, size=(1, 5, 5)) - x) > 12.0)
    near = np.stack([ndi.binary_dilation(c, np.ones((2 * reach + 1,) * 2)) for c in cm])
    return cm | (g & near)


def score(frames, cell, tol, long_px, bridge=False):
    """frames uint8 (N, H, W)."""
    cm = dt.crack_map(frames)
    if bridge:
        cm = bridged(frames, cm)
    tot = {"px": 0, "n": 0, "lens": [], "long_px": 0, "ends": 0, "ends_grid": 0}
    for c in cm:
        r = components(c, cell, tol, long_px)
        for k in ("px", "n", "long_px", "ends", "ends_grid"):
            tot[k] += r[k]
        tot["lens"] += r["lens"]
    H, W = cm.shape[1:]
    dens_x = cm.mean(axis=(0, 1))  # per column
    dens_y = cm.mean(axis=(0, 2))  # per row
    mod_x, mod_y = np.arange(W) % cell, np.arange(H) % cell
    line = lambda m: np.isin(m, (cell - 1, 0))
    mid = lambda m: (m >= cell // 2 - 2) & (m <= cell // 2 + 1)
    on = np.concatenate([dens_x[line(mod_x)], dens_y[line(mod_y)]]).mean()
    off = np.concatenate([dens_x[mid(mod_x)], dens_y[mid(mod_y)]]).mean()
    lens = np.asarray(tot["lens"]) if tot["lens"] else np.zeros(1)
    return {
        "frames": int(len(cm)),
        "crack_px": float(tot["px"] / cm.size),
        "comp_per_kpx": float(1000 * tot["n"] / max(tot["px"], 1)),
        "len_med": float(np.median(lens)),
        "len_mean": float(lens.mean()),
        "len_p90": float(np.percentile(lens, 90)),
        "long_share": float(tot["long_px"] / max(tot["px"], 1)),
        "grid_end": float(tot["ends_grid"] / max(tot["ends"], 1)),
        "grid_end_uniform": float(1 - (1 - (2 * tol + 1) / cell) ** 2),
        "grid_dip": float(on / max(off, 1e-12)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True, metavar="NAME=NPZ")
    p.add_argument("--cell", type=int, default=16)
    p.add_argument("--tol", type=int, default=1)
    p.add_argument("--long", type=float, default=24.0)
    p.add_argument("--bridge", action="store_true", help="refill the grid lines a crack crosses before labelling")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = {}
    for i, spec in enumerate(a.model):
        name, path = spec.split("=", 1)
        z = np.load(path, allow_pickle=True)
        if i == 0:
            res["real"] = score(z["real"], a.cell, a.tol, a.long, a.bridge)
        f = z["fake"]
        res[name] = score(f.reshape(-1, *f.shape[-2:]), a.cell, a.tol, a.long, a.bridge)
        print(name, json.dumps({k: round(v, 4) for k, v in res[name].items()}), flush=True)
    keys = ["crack_px", "comp_per_kpx", "len_med", "len_mean", "len_p90", "long_share", "grid_end", "grid_dip"]
    print(f"{'model':24s} " + " ".join(f"{k:>12}" for k in keys))
    for n, r in res.items():
        print(f"{n:24s} " + " ".join(f"{r[k]:12.4f}" for k in keys))
    print(f"grid_end if uniform: {res['real']['grid_end_uniform']:.4f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"cell": a.cell, "tol": a.tol, "long": a.long, "models": res}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
