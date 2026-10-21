"""Temporal consistency along a column face: does the generated sequence behave like a loading history?

A trajectory = the labelled frames of one face of one column (exp, direction), in timestep order.
Sample k of every frame forms trajectory k. For a single-frame model the K samples of neighbouring
frames are independent draws, so its "trajectory" is a random pairing; that is exactly what you get
when you use such a model to render a sequence, and it is the baseline a temporal model must beat.

Per consecutive pair (t, t+1) of one trajectory:
  flicker      mean |x_{t+1} − x_t| (gray levels). GT pairs carry the real change plus photo noise.
  tone_jump    |mean(x_{t+1}) − mean(x_t)|: the global brightness part of the flicker.
  struct_flicker  flicker after subtracting each frame's own mean: what moves apart from the tone.
  persistence  share of the damage pixels at t (damage_texture classes, dilated 2 px) still damaged
               at t+1. Real damage does not heal, so GT stays near 1 except for debris falling off.
  crack_keep   share of the crack pixels at t with a crack within 2 px at t+1.
  regress      1 if the damage area drops by more than 20 % (and at least 50 px), else 0.
Per trajectory:
  brightness_slope  OLS slope of the image mean vs frame index (gray levels per frame); a recurrent
                    sampler that accumulates residuals drifts here.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from src.metrics import damage_texture as dt


def trajectories(label):
    """label: DataFrame row-aligned with the npz (exp, direction, timestep). → {key: row indices in time order}."""
    out = {}
    for (exp, direction), rows in label.groupby(["exp", "direction"], sort=True):
        out[f"{exp}/{direction}"] = rows.sort_values("timestep", kind="stable").index.to_numpy()
    return out


def _damage_masks(u8, thr):
    rough, fog, white, _ = dt.pixel_classes(u8, thr)
    return rough | fog | white


def pair_stats(seq, thr, tol=2):
    """seq: (T, H, W) uint8 frames of one trajectory → dict of per-pair arrays (T-1,) and the brightness slope."""
    seq = np.asarray(seq)
    x = seq.astype(np.float32)
    dmg = _damage_masks(seq, thr)
    crk = dt.crack_map(seq)
    st = np.ones((1, 2 * tol + 1, 2 * tol + 1), bool)
    dmg_d = ndi.binary_dilation(dmg, st)
    crk_d = ndi.binary_dilation(crk, st)
    a, b = slice(0, -1), slice(1, None)
    area = dmg.reshape(len(seq), -1).sum(1).astype(np.float64)
    n_d = area[a]
    n_c = crk[a].reshape(len(seq) - 1, -1).sum(1)
    keep = (dmg[a] & dmg_d[b]).reshape(len(seq) - 1, -1).sum(1)
    ckeep = (crk[a] & crk_d[b]).reshape(len(seq) - 1, -1).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "flicker": np.abs(x[b] - x[a]).mean((1, 2)),
            "tone_jump": np.abs(x[b].mean((1, 2)) - x[a].mean((1, 2))),
            "struct_flicker": np.abs((x[b] - x[b].mean((1, 2), keepdims=True)) - (x[a] - x[a].mean((1, 2), keepdims=True))).mean((1, 2)),
            "persistence": np.where(n_d >= 50, keep / np.maximum(n_d, 1), np.nan),
            "crack_keep": np.where(n_c >= 20, ckeep / np.maximum(n_c, 1), np.nan),
            "regress": np.where(n_d >= 50, ((area[a] - area[b]) > np.maximum(0.2 * area[a], 50)).astype(float), np.nan),
        }
    m = x.mean((1, 2))
    idx = np.arange(len(m), dtype=np.float64)
    slope = float(np.polyfit(idx, m, 1)[0]) if len(m) > 2 else float("nan")
    return out, slope, m, area / dmg[0].size
