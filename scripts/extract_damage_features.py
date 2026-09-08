#!/usr/bin/env python3
"""Original-image EADI: CLAHE + interference whiteout + fill-below-spall-top + crack decoupling.

Pipeline
--------
1. Load original photo (not inpaint); gray+CLAHE; paint object-mask interference white
2. KMeans(k=3) on gray; clusters sorted dark→bright as K=1, K=2, K=3
3. Open → Close → largest connected component with
   ``area ≥ 200``, ``height ≥ 3``, and **not touching the top row**
   (rejects full-frame / top-down dark false positives) → fill_holes
4. **Fill below spall top:** ``np.maximum.accumulate(..., axis=0)`` carries 255
   from the first hit in each column down to the bottom of the frame.
5. ``R_crack = max(0, R_outside - μ_grid·(1 - R_spall))``
6. ``EADI = sqrt(alpha * R_crack + beta * R_spall)``

Tune ``alpha`` / ``beta`` with ``scripts/optimize_eadi_parameters.py --real``.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage


def resolve_object_mask_path(image_path: str, mask_dir: Optional[str] = None) -> Optional[str]:
    image_path = os.path.abspath(image_path)
    stem, _ = os.path.splitext(os.path.basename(image_path))
    if mask_dir is None:
        parent = os.path.dirname(os.path.dirname(image_path))
        mask_dir = os.path.join(parent, "mask")
    for path in (
        os.path.join(mask_dir, f"{stem}_mask.png"),
        os.path.join(mask_dir, f"{stem}.png"),
        os.path.join(mask_dir, os.path.basename(image_path)),
    ):
        if os.path.isfile(path):
            return path
    return None


class DamageFeatureExtractor:
    """Original-domain EADI with interference whiteout and below-top spall fill."""

    def __init__(
        self,
        mu_grid: float = 0.11,
        min_spall_area: float = 200.0,
        alpha: float = 0.323094,
        beta: float = 11.461967,
        image_size: Tuple[int, int] = (128, 128),
        open_ksize: int = 5,
        close_ksize: int = 5,
        min_spall_height: int = 3,
        max_spall_width_ratio: float = 1.0,
        reject_touch_top: bool = True,
        fill_below_spall_top: bool = True,
        keep_largest_component: bool = True,
        use_object_mask: bool = True,
        mask_dir: Optional[str] = None,
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        kmeans_random_state: int = 0,
        kmeans_n_init: int = 10,
        kmeans_k: int = 2,
    ):
        h, w = int(image_size[0]), int(image_size[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if float(alpha) < 0.0 or float(beta) < 0.0:
            raise ValueError("alpha and beta must be >= 0")

        self.mu_grid = float(mu_grid)
        self.min_spall_area = float(min_spall_area)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.image_size = (h, w)
        self.n_pixels = float(h * w)
        self.open_ksize = max(1, int(open_ksize))
        self.close_ksize = max(1, int(close_ksize))
        self.min_spall_height = max(1, int(min_spall_height))
        self.max_spall_width_ratio = float(max_spall_width_ratio)
        self.reject_touch_top = bool(reject_touch_top)
        self.fill_below_spall_top = bool(fill_below_spall_top)
        self.keep_largest_component = bool(keep_largest_component)
        self.use_object_mask = bool(use_object_mask)
        self.mask_dir = mask_dir
        self.use_clahe = bool(use_clahe)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid_size = (int(clahe_tile_grid_size[0]), int(clahe_tile_grid_size[1]))
        self.kmeans_random_state = int(kmeans_random_state)
        self.kmeans_n_init = int(kmeans_n_init)
        self.kmeans_k = max(2, int(kmeans_k))

    @staticmethod
    def _odd_ksize(k: int) -> int:
        k = max(1, int(k))
        return k if k % 2 == 1 else k + 1

    def _preprocess(self, image_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load original → resize → CLAHE gray→BGR → interference→white."""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")

        h, w = self.image_size
        if bgr.shape[0] != h or bgr.shape[1] != w:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)

        if self.use_clahe:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid_size,
            )
            bgr = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

        interference = None
        if self.use_object_mask:
            mask_path = resolve_object_mask_path(image_path, self.mask_dir)
            if mask_path is not None:
                m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    if m.shape[0] != h or m.shape[1] != w:
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    interference = m >= 128
                    bgr = bgr.copy()
                    bgr[interference] = (255, 255, 255)

        return bgr, interference

    def _kmeans_labels(
        self, image_bgr: np.ndarray, interference: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (H,W) labels in {0..k-1} sorted dark→bright, and center intensities."""
        h, w = self.image_size
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        pixels = gray.reshape(-1, 1).astype(np.float32)
        cv2.setRNGSeed(self.kmeans_random_state)
        _, labels, centers = cv2.kmeans(
            pixels,
            int(self.kmeans_k),
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01),
            max(1, self.kmeans_n_init),
            cv2.KMEANS_PP_CENTERS,
        )
        centers = centers.reshape(-1)
        order = np.argsort(centers)  # dark → bright
        remap = np.empty_like(order)
        remap[order] = np.arange(len(order))
        ranked = remap[labels.reshape(h, w)]
        if interference is not None:
            ranked = ranked.copy()
            ranked[interference] = -1
        return ranked.astype(np.int32), centers[order]

    def cluster_masks_from_path(self, image_path: str) -> Dict[int, np.ndarray]:
        bgr, interference = self._preprocess(image_path)
        return self.cluster_masks_after_morph(bgr, interference)

    def _morph_clean(self, binary_u8: np.ndarray) -> np.ndarray:
        k_open = self._odd_ksize(self.open_ksize)
        k_close = self._odd_ksize(self.close_ksize)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        cleaned = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, open_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
        return cleaned

    def cluster_masks_after_morph(
        self, image_bgr: np.ndarray, interference: Optional[np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """Per-cluster masks after open/close. K=1 darkest, K=k brightest."""
        ranked, _centers = self._kmeans_labels(image_bgr, interference)
        out = {}
        for k_id in range(int(self.kmeans_k)):
            raw = np.where(ranked == k_id, 255, 0).astype(np.uint8)
            out[k_id + 1] = self._morph_clean(raw)
        return out

    def _get_raw_mask(
        self, image_bgr: np.ndarray, interference: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        ranked, _centers = self._kmeans_labels(image_bgr, interference)
        raw_mask = np.where(ranked == 0, 255, 0).astype(np.uint8)
        r_raw = float(np.count_nonzero(raw_mask)) / self.n_pixels
        return raw_mask, r_raw

    def _filter_connected_components(self, cleaned: np.ndarray) -> np.ndarray:
        """Keep CCs that pass area/width/top-touch gates (optionally largest only)."""
        h, w = self.image_size
        empty = np.zeros((h, w), dtype=np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (cleaned > 0).astype(np.uint8), connectivity=8
        )
        candidates = []
        for idx in range(1, count):
            _x, _y, width, _height, area = (int(v) for v in stats[idx])
            width_ok = width / float(w) <= self.max_spall_width_ratio
            if not width_ok:
                continue
            if self.min_spall_area > 0 and area < self.min_spall_area:
                continue
            if self.reject_touch_top and bool(np.any(labels[0, :] == idx)):
                continue
            candidates.append((area, idx))
        if not candidates:
            return empty
        if self.keep_largest_component:
            keep = {max(candidates)[1]}
        else:
            keep = {idx for _area, idx in candidates}
        selected = np.isin(labels, list(keep))
        return ndimage.binary_fill_holes(selected).astype(np.uint8) * 255

    def _get_spalling_features(self, raw_mask: np.ndarray) -> Tuple[float, np.ndarray, int]:
        """Open → close → CC gates → optional largest-only → optional fill-below-top."""
        h, w = self.image_size
        cleaned = self._morph_clean(raw_mask)
        empty = np.zeros((h, w), dtype=np.uint8)
        spall_mask = self._filter_connected_components(cleaned)
        ys, _ = np.where(spall_mask == 255)
        if ys.size == 0:
            return 0.0, empty, 0
        min_y = int(ys.min())
        spall_height = int(h - min_y)
        if self.fill_below_spall_top:
            spall_mask = np.maximum.accumulate(spall_mask, axis=0)
        r_spall = float(np.count_nonzero(spall_mask)) / self.n_pixels
        return r_spall, spall_mask, spall_height

    def _get_crack_features(
        self, raw_mask: np.ndarray, spall_mask: np.ndarray, r_spall: float
    ) -> Tuple[float, float, np.ndarray]:
        outside_mask = cv2.bitwise_and(raw_mask, cv2.bitwise_not(spall_mask))
        r_outside = float(np.count_nonzero(outside_mask)) / self.n_pixels
        r_crack = max(0.0, r_outside - self.mu_grid * (1.0 - float(r_spall)))
        return float(r_crack), float(r_outside), outside_mask

    def calculate_EADI(self, r_crack: float, r_spall: float) -> Dict[str, float]:
        inside = self.alpha * float(r_crack) + self.beta * float(r_spall)
        return {"EADI": float(math.sqrt(max(0.0, inside))), "inside": float(inside)}

    def evaluate_bgr(
        self,
        image_bgr: np.ndarray,
        interference: Optional[np.ndarray] = None,
        apply_clahe: bool = True,
    ) -> Dict[str, Union[float, int, np.ndarray]]:
        """Same pipeline as ``evaluate``, from an already-loaded BGR array."""
        h, w = self.image_size
        bgr = np.asarray(image_bgr)
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR, got {getattr(bgr, 'shape', None)}")
        if bgr.shape[0] != h or bgr.shape[1] != w:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        if apply_clahe and self.use_clahe:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid_size,
            )
            bgr = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        if interference is not None:
            inter = np.asarray(interference)
            if inter.shape[:2] != (h, w):
                inter = cv2.resize(
                    inter.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                )
            interference = inter.astype(bool)
            bgr = bgr.copy()
            bgr[interference] = (255, 255, 255)

        raw_mask, r_raw = self._get_raw_mask(bgr, interference)
        r_spall, spall_mask, spall_height = self._get_spalling_features(raw_mask)
        r_crack, r_outside, outside_mask = self._get_crack_features(raw_mask, spall_mask, r_spall)
        scores = self.calculate_EADI(r_crack, r_spall)

        return {
            "EADI": float(scores["EADI"]),
            "R_crack": float(r_crack),
            "R_spall": float(r_spall),
            "R_outside": float(r_outside),
            "R_raw": float(r_raw),
            "spall_height": int(spall_height),
            "mu_grid": float(self.mu_grid),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "min_spall_area": float(self.min_spall_area),
            "min_spall_height": int(self.min_spall_height),
            "max_spall_width_ratio": float(self.max_spall_width_ratio),
            "reject_touch_top": bool(self.reject_touch_top),
            "fill_below_spall_top": bool(self.fill_below_spall_top),
            "raw_mask": raw_mask,
            "spall_mask": spall_mask,
            "outside_mask": outside_mask,
            "image_bgr": bgr,
            "interference_mask": (
                None if interference is None else (interference.astype(np.uint8) * 255)
            ),
        }

    def evaluate(self, image_path: str) -> Dict[str, Union[float, int, np.ndarray]]:
        image_bgr, interference = self._preprocess(image_path)
        return self.evaluate_bgr(image_bgr, interference=interference, apply_clahe=False)

    def process_image(self, image_path: str) -> Dict[str, Union[float, int, np.ndarray]]:
        return self.evaluate(image_path)


def extract_damage_features(image_path: str, **kwargs) -> Dict[str, Union[float, int, np.ndarray]]:
    return DamageFeatureExtractor(**kwargs).evaluate(image_path)


def _save_preview(result: dict, out_path: str) -> None:
    bgr = result["image_bgr"]
    raw = result["raw_mask"]
    spall = result["spall_mask"]
    outside = result["outside_mask"]
    h, w = raw.shape

    overlay = bgr.copy()
    tint = overlay.copy()
    tint[spall > 0] = (0, 255, 255)
    tint[outside > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(tint, 0.45, overlay, 0.55, 0)
    if int(result.get("spall_height", 0)) > 0:
        min_y = h - int(result["spall_height"])
        cv2.line(overlay, (0, min_y), (w - 1, min_y), (0, 255, 0), 1)

    inter = result.get("interference_mask")
    inter_vis = np.zeros_like(bgr) if inter is None else cv2.cvtColor(inter, cv2.COLOR_GRAY2BGR)

    row = np.concatenate(
        [
            bgr,
            inter_vis,
            cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(spall, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(outside, cv2.COLOR_GRAY2BGR),
            overlay,
        ],
        axis=1,
    )
    panel = cv2.resize(row, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    title = (
        f"EADI={result['EADI']:.3f} Rc={result['R_crack']:.3f} Rs={result['R_spall']:.3f} "
        f"H={result.get('spall_height', 0)} | CLAHE+mask | interfer | raw | spall↓ | out | overlay"
    )
    cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cv2.imwrite(out_path, panel)


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description="Original+mask EADI extractor (fill-below-spall-top)")
    p.add_argument("image_path", type=str)
    p.add_argument("--mu-grid", type=float, default=0.11)
    p.add_argument("--min-spall-area", type=float, default=200.0)
    p.add_argument("--alpha", type=float, default=0.323094)
    p.add_argument("--beta", type=float, default=11.461967)
    p.add_argument("--image-size", type=int, nargs=2, default=(128, 128), metavar=("H", "W"))
    p.add_argument("--no-fill-below-top", action="store_true")
    p.add_argument(
        "--allow-touch-top",
        action="store_true",
        help="Allow spall CCs that touch the top row (default: reject them)",
    )
    p.add_argument("--no-object-mask", action="store_true")
    p.add_argument("--preview", type=str, default="")
    args = p.parse_args(argv)

    ex = DamageFeatureExtractor(
        mu_grid=args.mu_grid,
        min_spall_area=args.min_spall_area,
        alpha=args.alpha,
        beta=args.beta,
        image_size=tuple(args.image_size),
        reject_touch_top=not args.allow_touch_top,
        fill_below_spall_top=not args.no_fill_below_top,
        use_object_mask=not args.no_object_mask,
    )
    result = ex.evaluate(args.image_path)
    print(
        f"EADI={result['EADI']:.6f}  R_crack={result['R_crack']:.6f}  "
        f"R_spall={result['R_spall']:.6f}  H={result['spall_height']}  "
        f"alpha={result['alpha']:.4f} beta={result['beta']:.4f}"
    )
    if args.preview:
        _save_preview(result, args.preview)
        print(f"preview -> {args.preview}")


if __name__ == "__main__":
    main()
