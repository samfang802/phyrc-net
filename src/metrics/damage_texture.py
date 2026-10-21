"""Interpretable pixel statistics of the damaged region: what kind of damage was drawn, not only how much.

Complements the DINOv3 metrics (docs/eval/), which cannot say *why* two images differ.
Everything runs on the 128×128 uint8 CLAHE images that the generators produce and is calibrated on
intact training frames, so no threshold is hand-tuned on the test column.

Per pixel, with a 7×7 window:
  d = local mean − surface level of the image (median of the top 30 % of rows, which stays intact
      in this data), so a globally darker/lighter image does not count as damage;
  E = local mean of |x − Gaussian(x, σ=1)|, the fine-texture energy.
Classes (thresholds = percentiles of d and E over intact training frames, `calibrate`):
  rough  d < dark (p0.5)                       exposed concrete / rebar: dark *and* detailed
         and E ≥ smooth (intact median E)
  fog    d < mild (p5) and E < smooth          darker than the surface but smoother than even intact
                                               concrete: the gray haze
  white  d > bright (p99.5)                    brighter than any intact surface: the white blobs
damage = rough ∪ fog ∪ white. Inside it, `black_share` = pixels (not windowed) darker than the
surface by more than the p0.1 of intact pixels (darker than the core of a grid line): the pitch-black
smears that make white fragments look like blotches.

Crack maps (`crack_map`): black-hat of the image (thin dark structures), with the drawn grid removed
by its row/column projection, small specks dropped. `crack_f1` compares two sets of maps with a
pixel tolerance (precision/recall of crack pixels within `tol` px of the other map).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

TOP_FRAC = 0.3
WIN = 7


def _residual(u8):
    """(N,H,W) uint8 → per-pixel intensity minus the image's surface level."""
    x = np.asarray(u8, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    ref = np.median(x[:, : max(1, int(TOP_FRAC * x.shape[1]))].reshape(len(x), -1), axis=1)[:, None, None]
    return x, x - ref


def local_maps(u8):
    """(N,H,W) uint8 → d (local mean minus the image's surface level), E (fine-texture energy)."""
    x, r = _residual(u8)
    d = ndi.uniform_filter(r, (1, WIN, WIN))
    E = ndi.uniform_filter(np.abs(x - ndi.gaussian_filter(x, (0, 1, 1))), (1, WIN, WIN))
    return d, E


def calibrate(intact_u8):
    """Thresholds from intact frames (lowest drift codes of the training columns)."""
    d, E = local_maps(intact_u8)
    return {
        "dark": float(np.percentile(d, 0.5)),
        "mild": float(np.percentile(d, 5)),
        "bright": float(np.percentile(d, 99.5)),
        "smooth": float(np.median(E)),
        "black": float(np.percentile(_residual(intact_u8)[1], 0.1)),
        "n_intact": int(len(d)),
    }


def pixel_classes(u8, thr):
    d, E = local_maps(u8)
    white = d > thr["bright"]
    rough = (d < thr["dark"]) & (E >= thr["smooth"])
    fog = (d < thr["mild"]) & (E < thr["smooth"]) & ~white
    return rough, fog, white, E


def texture_stats(u8, thr, min_blob=4, chunk=256):
    """Per image: area fractions of each class, shares inside the damage, white-blob count/size."""
    u8 = np.asarray(u8)
    keys = ("damage_area", "rough", "fog", "white", "fog_share", "white_share", "black_share", "white_blobs", "blob_size", "E_damage")
    out = {k: np.full(len(u8), np.nan) for k in keys}
    for s in range(0, len(u8), chunk):
        rough, fog, white, E = pixel_classes(u8[s : s + chunk], thr)
        black = _residual(u8[s : s + chunk])[1] < thr["black"]
        dmg = rough | fog | white
        for j in range(len(rough)):
            i = s + j
            n = dmg[j].sum()
            out["damage_area"][i] = n / dmg[j].size
            out["rough"][i], out["fog"][i], out["white"][i] = rough[j].mean(), fog[j].mean(), white[j].mean()
            lab, nl = ndi.label(white[j])
            sizes = ndi.sum(white[j], lab, range(1, nl + 1)) if nl else np.zeros(0)
            sizes = sizes[sizes >= min_blob]
            out["white_blobs"][i] = len(sizes)
            if len(sizes):
                out["blob_size"][i] = sizes.mean()
            # Shares only mean something once there is a damaged region to speak of.
            if n >= 50:
                out["fog_share"][i] = fog[j].sum() / n
                out["white_share"][i] = white[j].sum() / n
                out["black_share"][i] = (black[j] & dmg[j]).sum() / n
                out["E_damage"][i] = E[j][dmg[j]].mean()
    return out


# --------------------------------------------------------------------------- cracks


def _grid_mask(m, frac=0.55, pad=1):
    """Rows/columns whose crack-candidate coverage exceeds `frac` are drawn grid lines."""
    H, W = m.shape[-2:]
    g = np.zeros_like(m)
    for i in range(len(m)):
        cols = np.flatnonzero(m[i].mean(0) > frac)
        rows = np.flatnonzero(m[i].mean(1) > frac)
        for c in cols:
            g[i, :, max(0, c - pad) : c + pad + 1] = True
        for r in rows:
            g[i, max(0, r - pad) : r + pad + 1, :] = True
    return g


def crack_map(u8, thr=12.0, min_size=12):
    """(N,H,W) uint8 → bool crack maps (thin dark structures minus the grid and specks)."""
    x = np.asarray(u8, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    m = (ndi.grey_closing(x, size=(1, 5, 5)) - x) > thr
    c = m & ~_grid_mask(m)
    out = np.zeros_like(c)
    for i in range(len(c)):
        lab, n = ndi.label(c[i], np.ones((3, 3)))
        if n:
            s = ndi.sum(c[i], lab, range(1, n + 1))
            out[i] = np.isin(lab, 1 + np.flatnonzero(s >= min_size))
    return out


def _within(c, tol):
    return np.stack([ndi.distance_transform_edt(~ci) <= tol if ci.any() else np.zeros_like(ci) for ci in c])


def crack_f1(a, b, tol=3, mask=None):
    """Tolerance F1 between every map in a (Na,H,W) and every map in b (Nb,H,W) → (Na,Nb).

    mask: optional (H,W) bool region to compare in (e.g. the part of the column above the spall).
    Pairs where either map is empty get F1 = 0 (or 1 if both are empty).
    """
    if mask is not None:
        a, b = a & mask, b & mask
    A = a.reshape(len(a), -1).astype(np.float32)
    B = b.reshape(len(b), -1).astype(np.float32)
    Aw = _within(a, tol).reshape(len(a), -1).astype(np.float32)
    Bw = _within(b, tol).reshape(len(b), -1).astype(np.float32)
    na, nb = A.sum(1)[:, None], B.sum(1)[None, :]
    prec = (A @ Bw.T) / np.maximum(na, 1)
    rec = (Aw @ B.T) / np.maximum(nb, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    f1[(na == 0) & (nb == 0)] = 1.0
    return f1
