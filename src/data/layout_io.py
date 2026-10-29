"""Shared layout I/O for the single-frame redesign (docs/redesign/phase1_layout_harness.md).

Layout npz written by any layout generator (OFP, C2 flow matching, oracle labels):
  filename (N,)            evaluation frames (split's label.csv names)
  layout   (N,K,S,S) f32   spall coverage 0-1 per cell (S = 32 normally; 128 also accepted)
  method   str, split str

Renderer input = the exact spall SDF channel of anylabel_layout_sdf128.npz
(scripts/export/export_anylabel_layout_sdf.py): coverage → bilinear to 128 → >= 0.5 → truncated SDF.
"""

from __future__ import annotations

import importlib.util
import os

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EXPORT = os.path.join(_ROOT, "scripts", "export", "export_anylabel_layout_sdf.py")
_sdf_mod = None


def _exporter():
    """The export script itself, so the SDF code is the same function, not a copy."""
    global _sdf_mod
    if _sdf_mod is None:
        spec = importlib.util.spec_from_file_location("_export_anylabel_layout_sdf", _EXPORT)
        _sdf_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_sdf_mod)
    return _sdf_mod


def coverage_to_mask128(cov, size: int = 128, thr: float = 0.5) -> np.ndarray:
    """cov (..., S, S) in [0, 1] → bool (..., size, size). Bilinear (cv2, half-pixel centers) then >= thr."""
    cov = np.asarray(cov, dtype=np.float32)
    lead, s = cov.shape[:-2], cov.shape[-2:]
    flat = cov.reshape((-1,) + s)
    out = np.empty((len(flat), size, size), dtype=bool)
    for i, c in enumerate(flat):
        up = c if c.shape == (size, size) else cv2.resize(c, (size, size), interpolation=cv2.INTER_LINEAR)
        out[i] = up >= thr
    return out.reshape(lead + (size, size))


def mask_to_sdf128(mask, trunc_px: float = 8.0) -> np.ndarray:
    """bool/uint8 (..., H, W) → float32 SDF in [-1, 1] (positive inside; empty = -1)."""
    mask = np.asarray(mask)
    lead = mask.shape[:-2]
    flat = mask.reshape((-1,) + mask.shape[-2:]).astype(np.uint8)
    f = _exporter().truncated_sdf
    return np.stack([f(m, float(trunc_px)) for m in flat]).reshape(lead + mask.shape[-2:]).astype(np.float32)


def coverage_to_sdf128(cov, trunc_px: float = 8.0, size: int = 128) -> np.ndarray:
    """cov (..., S, S) coverage 0-1 → float32 SDF (..., 128, 128), the renderer's layout channel."""
    return mask_to_sdf128(coverage_to_mask128(cov, size), trunc_px)


def load_layout_npz(path):
    """→ dict(filename list[str], layout float32 (N,K,S,S), method str, split str). Read once."""
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    lay = np.asarray(d["layout"], dtype=np.float32)
    if lay.ndim == 3:
        lay = lay[:, None]
    if lay.ndim != 4 or lay.shape[-1] != lay.shape[-2]:
        raise ValueError(f"layout must be (N,K,S,S), got {lay.shape}")
    return {
        "filename": [str(f) for f in d["filename"]],
        "layout": lay,
        "method": str(d.get("method", "")),
        "split": str(d.get("split", "")),
    }


def save_layout_npz(path, filename, layout, method, split):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, filename=np.asarray(filename), layout=np.asarray(layout, dtype=np.float32),
                        method=np.asarray(method), split=np.asarray(split))


def anylabel_cache(path="Data/DamagePattern/anylabel_layout_sdf128.npz"):
    """→ (name→row dict, mask bool (M,128,128), sdf float32 (M,128,128)). Read once."""
    z = np.load(os.path.join(_ROOT, path) if not os.path.isabs(path) else path)
    names = [str(f) for f in z["filename"]]
    return {f: i for i, f in enumerate(names)}, np.asarray(z["mask"]).astype(bool), np.asarray(z["sdf"], dtype=np.float32)


def onset_layout(split_filenames, key="iso32", path="Data/DamagePattern/derived/onset_field.npz"):
    """Real monotone coverage (iso32 / iso16 / cov32 ...) for the given frames → (N,S,S) float32."""
    z = np.load(os.path.join(_ROOT, path))
    names = {str(f): i for i, f in enumerate(z["filenames"])}
    arr = np.asarray(z[key], dtype=np.float32)
    return np.stack([arr[names[f]] for f in split_filenames])
