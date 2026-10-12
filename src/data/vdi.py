"""Residual-corrected visual damage-index helpers."""

import os

import numpy as np
import pandas as pd

def _resolve_r_spall_csv(pattern_root, r_spall_csv=None):
    """Prefer explicit path, then ``<pattern_root>/r_spall.csv``."""
    if r_spall_csv:
        return str(r_spall_csv)
    candidate = os.path.join(pattern_root, "r_spall.csv")
    if os.path.isfile(candidate):
        return candidate
    return None


def _load_r_spall_map(r_spall_csv):
    """Load ``filename -> R_spall`` (also index by stem for .jpg/.png mismatch)."""
    if not r_spall_csv or not os.path.isfile(r_spall_csv):
        return {}
    df = pd.read_csv(r_spall_csv)
    col = "R_spall" if "R_spall" in df.columns else None
    if col is None or "filename" not in df.columns:
        raise ValueError(
            f"r_spall_csv must contain columns ['filename', 'R_spall'], got {list(df.columns)}"
        )
    mapping = {}
    for _, row in df.iterrows():
        name = str(row["filename"])
        val = float(np.clip(float(row[col]), 0.0, 1.0))
        mapping[name] = val
        stem, _ = os.path.splitext(name)
        mapping[stem] = val
    return mapping


def _lookup_r_spall(mapping: dict, filename: str, default: float = 0.0) -> float:
    name = str(filename)
    if name in mapping:
        return float(mapping[name])
    stem, _ = os.path.splitext(name)
    if stem in mapping:
        return float(mapping[stem])
    return float(default)


def fit_r_bar(
    vdi,
    r_spall,
    method: str = "poly",
    poly_degree: int = 2,
    ma_window: int = 51,
):
    """Fit expected spall trend ``R̄(vDI)`` from (vDI, R_spall) samples.

    Returns
    -------
    r_bar : callable
        ``r_bar(x) -> float`` or array, clipped to [0, 1].
    info : dict
        Fit metadata for logging.
    """
    x = np.asarray(vdi, dtype=np.float64).reshape(-1)
    y = np.asarray(r_spall, dtype=np.float64).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    method = str(method).lower().strip()
    if len(x) < 2:
        # Degenerate: constant 0 trend.
        def r_bar_const(q):
            q = np.asarray(q, dtype=np.float64)
            out = np.zeros_like(q, dtype=np.float64)
            return float(out) if np.ndim(out) == 0 else out

        return r_bar_const, {"method": "const0", "n": int(len(x))}

    if method in ("ma", "moving_average", "moving-avg"):
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        w = max(1, int(ma_window))
        if w % 2 == 0:
            w += 1
        # Odd window centered rolling mean; edges use expanding partial window.
        kernel = np.ones(w, dtype=np.float64) / float(w)
        pad = w // 2
        ys_pad = np.pad(ys, (pad, pad), mode="edge")
        ys_ma = np.convolve(ys_pad, kernel, mode="valid")
        # Unique x for interpolation (average y at duplicate vDI).
        uniq_x, inv = np.unique(xs, return_inverse=True)
        uniq_y = np.zeros_like(uniq_x)
        counts = np.zeros_like(uniq_x)
        np.add.at(uniq_y, inv, ys_ma)
        np.add.at(counts, inv, 1.0)
        uniq_y = uniq_y / np.maximum(counts, 1.0)

        def r_bar_ma(q):
            q_arr = np.asarray(q, dtype=np.float64)
            yq = np.interp(q_arr, uniq_x, uniq_y, left=uniq_y[0], right=uniq_y[-1])
            yq = np.clip(yq, 0.0, 1.0)
            return float(yq) if np.ndim(yq) == 0 else yq

        return r_bar_ma, {
            "method": "ma",
            "ma_window": int(w),
            "n": int(len(x)),
            "x_min": float(uniq_x.min()),
            "x_max": float(uniq_x.max()),
        }

    # Default: polynomial regression.
    deg = int(max(1, min(int(poly_degree), max(1, len(x) - 1))))
    coeffs = np.polyfit(x, y, deg=deg)

    def r_bar_poly(q):
        q_arr = np.asarray(q, dtype=np.float64)
        yq = np.polyval(coeffs, q_arr)
        yq = np.clip(yq, 0.0, 1.0)
        return float(yq) if np.ndim(yq) == 0 else yq

    return r_bar_poly, {
        "method": "poly",
        "poly_degree": int(deg),
        "coeffs": [float(c) for c in coeffs],
        "n": int(len(x)),
    }


def compute_vdi(vdi_original: float, r_spall: float, r_bar, beta: float = 0.3) -> float:
    """vDI = max(0, vDI_original + β · (R̄(vDI_original) - R_spall)).

    No upper clip — vDI may exceed 1 when the original DI is already large
    or the residual correction pushes it above 1.
    """
    vo = float(vdi_original)
    rs = float(np.clip(float(r_spall), 0.0, 1.0))
    r_hat = float(r_bar(vo))
    delta = r_hat - rs
    return float(max(0.0, vo + float(beta) * delta))


def compute_di_new(di_original: float, r_spall: float, beta: float = 0.3, r_bar=None, **_kwargs) -> float:
    """Deprecated alias of :func:`compute_vdi` (needs ``r_bar``)."""
    if r_bar is None:
        # Fallback: treat expected R as 0 → Δ = -R_spall
        r_bar = lambda _x: 0.0  # noqa: E731
    return compute_vdi(di_original, r_spall, r_bar=r_bar, beta=beta)


def robust_linear_di(di_original: float, r_spall: float, beta: float = 0.3, r_bar=None, **_kwargs) -> float:
    """Deprecated alias of :func:`compute_vdi`."""
    return compute_di_new(di_original, r_spall, beta=beta, r_bar=r_bar)


def _attach_vdi_labels(
    label_df: pd.DataFrame,
    r_map: dict,
    beta: float,
    r_bar_method: str = "poly",
    poly_degree: int = 2,
    ma_window: int = 51,
    fit_label_df: pd.DataFrame = None,
):
    """Add R_spall / R_bar / delta / vDI columns. Fit R̄ on ``fit_label_df`` (default: same df)."""
    out = label_df.copy()
    r_vals = [_lookup_r_spall(r_map, fn) for fn in out["filename"].astype(str)]
    out["R_spall"] = r_vals

    fit_df = label_df if fit_label_df is None else fit_label_df
    fit_r = np.asarray(
        [_lookup_r_spall(r_map, fn) for fn in fit_df["filename"].astype(str)],
        dtype=np.float64,
    )
    fit_vdi = fit_df["DI"].to_numpy(dtype=np.float64)
    r_bar, fit_info = fit_r_bar(
        fit_vdi,
        fit_r,
        method=r_bar_method,
        poly_degree=poly_degree,
        ma_window=ma_window,
    )

    vdi_orig = out["DI"].to_numpy(dtype=np.float64)
    r_hat = np.asarray([float(r_bar(v)) for v in vdi_orig], dtype=np.float64)
    delta = r_hat - np.asarray(r_vals, dtype=np.float64)
    out["vDI_original"] = vdi_orig
    out["R_bar"] = r_hat
    out["delta"] = delta
    out["vDI"] = [
        compute_vdi(vo, rs, r_bar=r_bar, beta=beta)
        for vo, rs in zip(vdi_orig.tolist(), r_vals)
    ]
    # Keep legacy column for any external tooling.
    out["DI_new"] = out["vDI"]
    return out, r_bar, fit_info
