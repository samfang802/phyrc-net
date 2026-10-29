"""DINOv3 features + unbiased MMD² for small, conditional test sets.

Design (docs/eval/ has the reasoning and references):

- Every sample carries a *group* id = index of the test condition it came from. A real image i
  and the K fakes generated from condition i share group i. Same-group pairs are dependent, so
  every estimator here drops them (block U-statistic). Within a set that also removes the
  diagonal, which is what makes plain KID unbiased.
- All kernel sums are reduced to G×G group-sum matrices once. Any subset of conditions
  (jackknife, DI tertiles, random halves) is then cheap.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_DINO = "facebook/dinov3-vits16-pretrain-lvd1689m"


# --------------------------------------------------------------------------- images


def to_uint8_gray(x: torch.Tensor) -> np.ndarray:
    """(B,C,H,W) float in [-1,1] → (B,H,W) uint8. Channel 0 (inputs are gray replicated to 3)."""
    x01 = torch.clamp(x[:, 0].float() * 0.5 + 0.5, 0.0, 1.0)
    return torch.round(x01 * 255.0).to(torch.uint8).cpu().numpy()


def load_real_uint8(filenames, image_dir, image_size=128, use_clahe=True, clip_limit=2.0, tile=(8, 8)):
    """Real images through the generator's deterministic preprocessing (resize, gray, CLAHE)."""
    from PIL import Image

    from src.utils import build_damage_image_transform

    tf = build_damage_image_transform(
        image_size=image_size,
        use_augmentation=False,
        output_scale=True,
        include_random_erasing=False,
        use_grayscale_input=True,
        use_clahe=use_clahe,
        clahe_clip_limit=clip_limit,
        clahe_tile_grid_size=tile,
    )
    out = np.empty((len(filenames), image_size, image_size), dtype=np.uint8)
    for i, name in enumerate(filenames):
        path = os.path.join(image_dir, name)
        if not os.path.exists(path):  # id lists name .jpg; a lossless reference dir holds <stem>.png
            path = os.path.join(image_dir, os.path.splitext(name)[0] + ".png")
        x = tf(Image.open(path).convert("RGB"))
        out[i] = torch.round(torch.clamp(x[0], 0.0, 1.0) * 255.0).to(torch.uint8).numpy()
    return out


# --------------------------------------------------------------------------- tone normalization

NORMALIZE_MODES = ("none", "mean", "meanstd", "flatfield")


def normalize_u8(u8: np.ndarray, mode: str = "none", target_mean=128.0, target_std=32.0, sigma=12.0) -> np.ndarray:
    """Per-image tone normalization applied identically to real and fake images before features.

    mean       shift so each image has mean 128 (removes global brightness)
    meanstd    mean 128 and std 32 (also removes global contrast; crack density is partly contrast)
    flatfield  x - gaussian_blur(x, sigma) + 128 (removes illumination and global tone, keeps thin lines)
    """
    if mode == "none":
        return u8
    import cv2

    x = u8.astype(np.float32)
    if mode == "mean":
        x = x - x.mean(axis=(1, 2), keepdims=True) + target_mean
    elif mode == "meanstd":
        m, s = x.mean(axis=(1, 2), keepdims=True), x.std(axis=(1, 2), keepdims=True)
        x = (x - m) / np.maximum(s, 1.0) * target_std + target_mean
    elif mode == "flatfield":
        x = np.stack([im - cv2.GaussianBlur(im, (0, 0), sigmaX=sigma) for im in x]) + target_mean
    else:
        raise ValueError(f"unknown normalize mode {mode!r}; choose from {NORMALIZE_MODES}")
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- features


def stratified_patch_indices(n_tokens, n_patch=16):
    """Fixed spatially stratified token indices, identical for every image.

    On the DINOv3-S/16 14×14 grid with 16 tokens, this is the Cartesian product of
    row/col indices [1, 5, 8, 12]. Other grids scale those source coordinates.
    """
    n_tokens, n_patch = int(n_tokens), int(n_patch)
    if n_patch <= 0:
        return np.zeros(0, dtype=np.int64)
    side = int(round(n_tokens ** 0.5))
    if side * side != n_tokens:
        idx = np.linspace(0, max(n_tokens - 1, 0), n_patch)
        return np.unique(np.clip(np.rint(idx), 0, n_tokens - 1).astype(np.int64))[:n_patch]
    if n_patch == 16:
        src = np.array([1, 5, 8, 12], dtype=np.float64)
        if side == 14:
            coords = src.astype(np.int64)
        else:
            coords = np.clip(np.rint(src * (side - 1) / 13.0), 0, side - 1).astype(np.int64)
            if len(np.unique(coords)) < 4:
                coords = np.linspace(0, side - 1, 4)
                coords = np.clip(np.rint(coords), 0, side - 1).astype(np.int64)
    else:
        n_side = max(1, int(round(n_patch ** 0.5)))
        coords = np.clip(np.rint(np.linspace(0, side - 1, n_side)), 0, side - 1).astype(np.int64)
    rows, cols = np.meshgrid(coords, coords, indexing="ij")
    idx = (rows * side + cols).ravel().astype(np.int64)
    if len(idx) >= n_patch:
        return idx[:n_patch]
    extra = np.setdiff1d(np.arange(n_tokens, dtype=np.int64), idx, assume_unique=False)
    return np.concatenate([idx, extra[: n_patch - len(idx)]])


class DinoFeatures:
    """Frozen DINOv3. Global = [CLS ; mean(patch)], patch = last-layer patch tokens."""

    def __init__(self, model_id=DEFAULT_DINO, device="cuda", input_size=224):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model_id = model_id
        self.device = device
        self.input_size = int(input_size)
        self.n_prefix = 1 + int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def _prep(self, u8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(u8)).to(self.device).float().div(255.0)[:, None]
        # Antialiased bicubic, same op for real and fake (Clean-FID lesson).
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bicubic", antialias=True, align_corners=False)
        x = x.clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def __call__(self, u8: np.ndarray, batch_size=64, n_patch=0, seed=0):
        """u8: (N,H,W). Returns global (N,2D) float32 and, if n_patch>0, patches (N,n_patch,D).

        Patch tokens are a fixed spatial grid (see stratified_patch_indices), shared by every
        image. `seed` is accepted for call-site compatibility and does not move the samples.
        """
        del seed
        glob, patch, idx = [], [], None
        for i in range(0, len(u8), batch_size):
            h = self.model(pixel_values=self._prep(u8[i : i + batch_size])).last_hidden_state.float()
            cls, tok = h[:, 0], h[:, self.n_prefix :]
            glob.append(torch.cat([cls, tok.mean(dim=1)], dim=1).cpu())
            if n_patch > 0:
                if idx is None:
                    idx = torch.as_tensor(stratified_patch_indices(int(tok.shape[1]), n_patch), device=tok.device)
                patch.append(tok[:, idx].cpu())
        g = torch.cat(glob).numpy()
        p = torch.cat(patch).numpy() if n_patch > 0 else None
        return g, p


# --------------------------------------------------------------------------- kernels


def poly_kernel(a, b):
    """KID kernel (x·y/d + 1)^3 (Bińkowski et al. 2018)."""
    return (a @ b.T / a.shape[1] + 1.0) ** 3


def rbf_kernel(sigma):
    def k(a, b):
        d2 = (a * a).sum(1, keepdim=True) + (b * b).sum(1)[None] - 2.0 * a @ b.T
        return torch.exp(-d2.clamp_min(0.0) / (2.0 * sigma * sigma))

    return k


def median_distance(x: np.ndarray, max_n=4000, seed=0) -> float:
    rng = np.random.default_rng(seed)
    if len(x) > max_n:
        x = x[rng.choice(len(x), max_n, replace=False)]
    t = torch.from_numpy(x).double()
    d = torch.cdist(t, t)
    iu = torch.triu_indices(len(t), len(t), offset=1)
    return float(d[iu[0], iu[1]].median())


@torch.no_grad()
def group_sums(x, gx, y, gy, n_groups, kernel, device="cuda", chunk=4096):
    """S[g,h] = Σ_{i∈g, j∈h} k(x_i, y_j), accumulated in float64."""
    xt = torch.from_numpy(x).to(device, torch.float32)
    yt = torch.from_numpy(y).to(device, torch.float32)
    oy = F.one_hot(torch.from_numpy(gy).long(), n_groups).to(device, torch.float32)
    S = torch.zeros(n_groups, n_groups, dtype=torch.float64, device=device)
    for i in range(0, len(xt), chunk):
        ox = F.one_hot(torch.from_numpy(gx[i : i + chunk]).long(), n_groups).to(device, torch.float32)
        for j in range(0, len(yt), chunk):
            kij = kernel(xt[i : i + chunk], yt[j : j + chunk])
            S += (ox.T @ (kij @ oy[j : j + chunk])).double()
    return S.cpu().numpy()


def counts(g, n_groups):
    return np.bincount(g, minlength=n_groups).astype(np.float64)


class GroupMMD:
    """MMD² from group sums, excluding same-group pairs, for any subset of groups."""

    def __init__(self, Sxx, Syy, Sxy, nx, ny):
        self.Sxx, self.Syy, self.Sxy, self.nx, self.ny = Sxx, Syy, Sxy, nx, ny
        self.G = len(nx)

    @staticmethod
    def _off(S, a, b, wa, wb):
        num = wa @ S @ wb - np.sum(wa * wb * np.diag(S))
        den = (wa @ a) * (wb @ b) - np.sum(wa * wb * a * b)
        return num / den

    def value(self, mask=None, mask_y=None):
        """mask selects groups for x (and for y unless mask_y is given; disjoint masks = two splits)."""
        wa = np.ones(self.G) if mask is None else mask.astype(np.float64)
        wb = wa if mask_y is None else mask_y.astype(np.float64)
        xx = self._off(self.Sxx, self.nx, self.nx, wa, wa)
        yy = self._off(self.Syy, self.ny, self.ny, wb, wb)
        xy = self._off(self.Sxy, self.nx, self.ny, wa, wb)
        return float(xx + yy - 2.0 * xy)

    def jackknife(self, mask=None, blocks=None):
        """Leave-one-out estimate and standard error.

        blocks: one label per group (condition). Leave-one-block-out, so correlated neighbouring
        frames (same face, same drift level) are removed together. None = leave-one-condition-out.
        """
        base = np.ones(self.G, bool) if mask is None else mask.astype(bool)
        labels = np.arange(self.G) if blocks is None else np.asarray(blocks)
        loo = []
        for b in np.unique(labels[base]):
            m = base & (labels != b)
            if m.sum() >= 2:
                loo.append(self.value(m))
        loo = np.asarray(loo)
        n = len(loo)
        se = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2))) if n > 1 else float("nan")
        return {"value": self.value(base), "se": se, "n_groups": int(base.sum()), "n_blocks": int(n)}, loo


def soft_di_tau(di):
    """Wide DI window: half a reference standard deviation, so one specimen's frames stay coupled."""
    di = np.asarray(di, dtype=np.float64)
    tau = 0.5 * float(np.std(di))
    if not np.isfinite(tau) or tau < 1e-8:
        span = float(np.max(di) - np.min(di)) if len(di) else 0.0
        tau = 0.2 * span
    if not np.isfinite(tau) or tau < 1e-8:
        tau = 1.0
    return tau


def di_weights(di, tau=None):
    """W_ij = exp(-(DI_i-DI_j)^2 / (2 τ^2)) for i≠j, and W_ii = 0."""
    di = np.asarray(di, dtype=np.float64)
    tau = soft_di_tau(di) if tau is None else float(tau)
    diff = di[:, None] - di[None, :]
    W = np.exp(-(diff ** 2) / (2.0 * tau * tau))
    np.fill_diagonal(W, 0.0)
    return W, tau


class SoftDI:
    """KID reweighted by a wide DI window, on an existing polynomial group-sum matrix.

    Same-group pairs stay out because W_ii = 0, including the cross-set pair (real_i, fake_i).
    Leave-one-condition-out zeros that condition's row and column and reuses compare().
    """

    def __init__(self, Sxx, Syy, Sxy, nx, ny, di, tau=None):
        self.Sxx, self.Syy, self.Sxy = Sxx, Syy, Sxy
        self.nx, self.ny = np.asarray(nx, np.float64), np.asarray(ny, np.float64)
        self.di = np.asarray(di, dtype=np.float64)
        self.W, self.tau = di_weights(self.di, tau)
        self.G = len(self.nx)

    @staticmethod
    def _wmean(S, a, b, W):
        den = float(np.sum(W * (a[:, None] * b[None, :])))
        if den <= 0.0:
            return float("nan")
        return float(np.sum(W * S) / den)

    def _weights(self, drop=None):
        if drop is None:
            return self.W
        W = self.W.copy()
        W[drop, :] = 0.0
        W[:, drop] = 0.0
        return W

    def value(self, drop=None):
        W = self._weights(drop)
        xx = self._wmean(self.Sxx, self.nx, self.nx, W)
        yy = self._wmean(self.Syy, self.ny, self.ny, W)
        xy = self._wmean(self.Sxy, self.nx, self.ny, W)
        return float(xx + yy - 2.0 * xy)

    def value_between(self, mask_a, mask_b):
        """MMD² between two disjoint condition sets (real-halves noise floor)."""
        ma = np.asarray(mask_a, dtype=np.float64)
        mb = np.asarray(mask_b, dtype=np.float64)
        Wa = self.W * ma[:, None] * ma[None, :]
        Wb = self.W * mb[:, None] * mb[None, :]
        Wab = self.W * ma[:, None] * mb[None, :]
        xx = self._wmean(self.Sxx, self.nx, self.nx, Wa)
        yy = self._wmean(self.Syy, self.ny, self.ny, Wb)
        xy = self._wmean(self.Sxy, self.nx, self.ny, Wab)
        return float(xx + yy - 2.0 * xy)

    def jackknife(self, blocks=None):
        labels = np.arange(self.G) if blocks is None else np.asarray(blocks)
        loo = np.array([self.value(drop=np.flatnonzero(labels == b)) for b in np.unique(labels)], dtype=np.float64)
        n = len(loo)
        se = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))
        res = {"value": self.value(), "se": se, "n_groups": int(n), "tau": float(self.tau)}
        return res, loo


def kid_soft_di(kid_mmd, di, tau=None):
    """Weighted KID from a polynomial GroupMMD. Does not recompute features."""
    return SoftDI(kid_mmd.Sxx, kid_mmd.Syy, kid_mmd.Sxy, kid_mmd.nx, kid_mmd.ny, di, tau=tau)


def regression_diagnostics(y_gt, y_pred, di):
    """Condition-level probe diagnostics against the physical label.

    mean_bias > 0 is over-shooting. high_di_bias is that bias on the top third of reference DI.
    calib_slope / calib_intercept are np.polyfit(y_gt, y_pred, 1).
    """
    y_gt = np.asarray(y_gt, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    di = np.asarray(di, dtype=np.float64)
    err = y_pred - y_gt
    ss_tot = float(np.sum((y_gt - y_gt.mean()) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0.0 else float("nan")
    slope, intercept = np.polyfit(y_gt, y_pred, 1)
    k = max(1, int(np.ceil(len(di) / 3.0)))
    high = np.argsort(di)[-k:]
    return {
        "r2": r2,
        "mae": float(np.mean(np.abs(err))),
        "mean_bias": float(np.mean(err)),
        "calib_slope": float(slope),
        "calib_intercept": float(intercept),
        "high_di_bias": float(np.mean(err[high])),
    }


# --------------------------------------------------------------------------- GT-neighbourhood contrast


def frame_key(filename):
    """'CTR1_NE_00250_0.jpg' → ('CTR1', 'NE'): specimen and camera face."""
    parts = os.path.basename(str(filename)).split("_")
    return parts[0], parts[1] if len(parts) > 1 else ""


def gt_neighbors(filenames, di, k=4):
    """For each condition i: the k real frames of the same specimen and face with the closest DI, i excluded."""
    keys = np.array(["/".join(frame_key(n)) for n in filenames])
    di = np.asarray(di, dtype=np.float64)
    G = len(keys)
    out = []
    for i in range(G):
        cand = np.flatnonzero((keys == keys[i]) & (np.arange(G) != i))
        out.append(cand[np.argsort(np.abs(di[cand] - di[i]), kind="stable")[:k]])
    return out


def pair_distance(a, b, kind="global"):
    """Euclidean distance matrix. kind='patch': a, b are (N, P, D) at the same P positions; mean over positions."""
    a, b = torch.from_numpy(np.asarray(a)).double(), torch.from_numpy(np.asarray(b)).double()
    if kind == "global":
        return torch.cdist(a, b).numpy()
    return torch.stack([torch.cdist(a[:, p], b[:, p]) for p in range(a.shape[1])]).mean(0).numpy()


def nbr_contrast_from_dist(D, nbrs):
    """D (G, G_real) distances from one image per condition to the reference reals.

    contrast_i = mean distance to the reals outside {i} ∪ N(i) − mean distance to N(i).
    Subtracting the far term removes the overall realism gap and keeps the condition-specific part.
    The GT itself is in neither set, so a model is not rewarded for matching one frame exactly.
    """
    G = D.shape[0]
    near = np.empty(G)
    far = np.empty(G)
    for i in range(G):
        nb = nbrs[i]
        mask = np.ones(D.shape[1], bool)
        mask[nb] = False
        mask[i] = False
        near[i] = D[i, nb].mean()
        far[i] = D[i, mask].mean()
    return far - near, near


def frame_block(filename):
    """'CTR1_NE_00250_0.jpg' → 'CTR1/NE/00250': frames of one face at one drift level are near-duplicates."""
    parts = os.path.basename(str(filename)).split("_")
    return "/".join(parts[:3])


def mean_jackknife(x, blocks=None):
    """(res, loo) for a per-condition mean, in the same form as GroupMMD.jackknife for compare().

    blocks: one label per condition; leave-one-block-out instead of leave-one-condition-out, so that
    correlated neighbouring frames do not shrink the standard error.
    """
    x = np.asarray(x, dtype=np.float64)
    if blocks is None:
        blocks = np.arange(len(x))
    blocks = np.asarray(blocks)
    labels = np.unique(blocks)
    loo = np.array([x[blocks != b].mean() for b in labels])
    n = len(labels)
    se = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))
    return {"value": float(x.mean()), "se": se, "n_groups": int(len(x)), "n_blocks": int(n)}, loo


def nbr_contrast(fake, real, nbrs, kind="global", n_shuffle=20, seed=0):
    """GT-neighbourhood contrast for fakes of shape (G, K, ...); per-condition values average over K.

    Returns per-condition contrast, per-condition near distance, and the shuffled-assignment
    contrasts (fakes moved to random conditions; a condition-aware score must drop to ~0).
    """
    G, K = fake.shape[:2]
    per_k = []
    near_k = []
    dists = []
    for k in range(K):
        D = pair_distance(fake[:, k], real, kind)
        dists.append(D)
        c, nr = nbr_contrast_from_dist(D, nbrs)
        per_k.append(c)
        near_k.append(nr)
    rng = np.random.default_rng(seed)
    shuffled = []
    for _ in range(n_shuffle):
        pi = rng.permutation(G)
        shuffled.append(np.mean([nbr_contrast_from_dist(D[pi], nbrs)[0].mean() for D in dists]))
    return np.mean(per_k, axis=0), np.mean(near_k, axis=0), np.asarray(shuffled)


# --------------------------------------------------------------------------- drift progression


def drift_levels(filenames):
    """'CTR1_NE_06000_3.jpg' → group 'CTR1/NE' (one face of one specimen), level 6000 (drift code), cycle 3."""
    groups, levels, cycles = [], [], []
    for n in filenames:
        parts = os.path.splitext(os.path.basename(str(n)))[0].split("_")
        groups.append(f"{parts[0]}/{parts[1]}")
        levels.append(int(parts[2]))
        cycles.append(parts[3] if len(parts) > 3 else "0")
    return np.asarray(groups), np.asarray(levels), np.asarray(cycles)


def _centroid_sqdist(xa, xb):
    """Unbiased ||mean_a − mean_b||² (a linear-kernel MMD²); xa (Na, K, F), xb (Nb, K, F).

    Pairs of images from the same condition are dependent (and identical i=i'), so the within-set
    terms only use pairs of different conditions. The plain centroid distance is biased upward by
    tr(Σ)/n, which favours sources with fewer images per level.
    """
    def within(x):
        n, k = x.shape[:2]
        per = x.sum(1)  # (n, F)
        tot = per.sum(0)
        return (tot @ tot - np.sum(per * per)) / (k * k * n * (n - 1))

    cross = xa.sum((0, 1)) @ xb.sum((0, 1)) / (xa.shape[0] * xa.shape[1] * xb.shape[0] * xb.shape[1])
    return float(within(xa) + within(xb) - 2.0 * cross)


def progression_curve(feat, groups, levels, keep=None):
    """Distance of every drift level from the lowest level of the same face, per face.

    feat: (G, K, F), one row per condition, K images each (K=1 for real frames). The distance is the
    square root of the unbiased centroid distance (clipped at 0), so seed noise and the number of
    images per level do not inflate it. `keep` drops conditions (jackknife). Levels with fewer than
    two conditions are skipped. Returns {face: (levels ascending without the base, distances)}.
    """
    keep = np.ones(len(groups), bool) if keep is None else keep
    out = {}
    for g in np.unique(groups):
        m = (groups == g) & keep
        lv = [L for L in np.unique(levels[m]) if np.sum(m & (levels == L)) >= 2]
        if len(lv) < 2:
            continue
        base = feat[m & (levels == lv[0])]
        d = [np.sqrt(max(_centroid_sqdist(feat[m & (levels == L)], base), 0.0)) for L in lv[1:]]
        out[g] = (np.asarray(lv[1:]), np.asarray(d))
    return out


def progression_scores(curve, curve_ref):
    """Base level excluded everywhere (its distance is 0 by construction, which would make any curve
    'correlate' with GT).

    ratio: summed distance over the non-base levels relative to the reference (1 = as much change).
    corr: Pearson over the (face, level) points. monotone: mean Spearman of each face's curve vs drift.
    """
    from scipy.stats import pearsonr, spearmanr

    a, b, mono = [], [], []
    for g, (lv, d) in curve.items():
        if g not in curve_ref:
            continue
        lv_r, d_r = curve_ref[g]
        common = np.intersect1d(lv, lv_r)
        a.extend(d[np.isin(lv, common)])
        b.extend(d_r[np.isin(lv_r, common)])
        if len(lv) >= 3 and np.std(d) > 0:
            mono.append(spearmanr(lv, d).statistic)
    a, b = np.asarray(a), np.asarray(b)
    return {
        "ratio": float(a.sum() / b.sum()) if b.sum() > 0 else float("nan"),
        "corr": float(pearsonr(a, b)[0]) if len(a) >= 3 and a.std() > 0 and b.std() > 0 else float("nan"),
        "monotone": float(np.mean(mono)) if mono else float("nan"),
    }


def progression(feat, feat_ref, groups, levels, units, n_shuffle=20, seed=0):
    """Progression scores of `feat` against the reference reals, with a jackknife and a shuffle test.

    units: jackknife label per condition. With cycle indices, leaving one cycle out removes one
    repetition of the whole drift sweep from both curves. Shuffle: fakes moved to random
    conditions, which mixes the drift levels; a condition-aware score must collapse toward 0.
    """
    ref = progression_curve(feat_ref, groups, levels)
    res = progression_scores(progression_curve(feat, groups, levels), ref)
    loo = {k: [] for k in res}
    for u in np.unique(units):
        keep = units != u
        s = progression_scores(progression_curve(feat, groups, levels, keep), progression_curve(feat_ref, groups, levels, keep))
        for k in res:
            loo[k].append(s[k])
    out = {}
    for k in res:
        arr = np.asarray(loo[k], dtype=np.float64)
        n = int(np.sum(np.isfinite(arr)))
        se = float(np.sqrt((n - 1) / n * np.nansum((arr - np.nanmean(arr)) ** 2))) if n > 1 else float("nan")
        out[k] = {"value": res[k], "se": se, "n_units": n}
    rng = np.random.default_rng(seed)
    shuf = [progression_scores(progression_curve(feat[rng.permutation(len(feat))], groups, levels), ref) for _ in range(n_shuffle)]
    out["shuffled_ratio"] = float(np.mean([s["ratio"] for s in shuf]))
    out["shuffled_corr"] = float(np.nanmean([s["corr"] for s in shuf]))
    out["shuffled_monotone"] = float(np.nanmean([s["monotone"] for s in shuf]))
    out["_loo_ratio"] = np.asarray(loo["ratio"])
    out["_loo_corr"] = np.asarray(loo["corr"])
    out["_curve"] = {g: (lv.tolist(), d.tolist()) for g, (lv, d) in progression_curve(feat, groups, levels).items()}
    out["_curve_ref"] = {g: (lv.tolist(), d.tolist()) for g, (lv, d) in ref.items()}
    return out


def compare(jk_a, jk_b):
    """Relative test on a shared reference: D = MMD²(A,R) − MMD²(B,R), jackknife over conditions.

    Paired by condition, in the spirit of Bounliphone et al. 2016 (test of relative similarity).
    """
    from scipy.stats import norm

    (ra, la), (rb, lb) = jk_a, jk_b
    d, dl = ra["value"] - rb["value"], la - lb
    n = len(dl)
    se = float(np.sqrt((n - 1) / n * np.sum((dl - dl.mean()) ** 2)))
    z = d / se if se > 0 else float("inf")
    return {"diff": float(d), "se": se, "z": float(z), "p_two_sided": float(2 * norm.sf(abs(z)))}


# --------------------------------------------------------------------------- set-level helpers


def global_mmds(fake, gf, real, gr, n_groups, sigma, device):
    """KID (poly) and RBF-MMD group objects for global features."""
    out = {}
    for name, k in (("kid", poly_kernel), ("mmd_rbf", rbf_kernel(sigma))):
        out[name] = GroupMMD(
            group_sums(fake, gf, fake, gf, n_groups, k, device),
            group_sums(real, gr, real, gr, n_groups, k, device),
            group_sums(fake, gf, real, gr, n_groups, k, device),
            counts(gf, n_groups),
            counts(gr, n_groups),
        )
    return out


def patch_kid(fake_p, gf, real_p, gr, n_groups, device):
    """KID on patch tokens; patches inherit the image's group, so same-image pairs drop out."""
    nf, npf, d = fake_p.shape
    nr, npr, _ = real_p.shape
    xf, xr = fake_p.reshape(-1, d), real_p.reshape(-1, d)
    pgf, pgr = np.repeat(gf, npf), np.repeat(gr, npr)
    return GroupMMD(
        group_sums(xf, pgf, xf, pgf, n_groups, poly_kernel, device),
        group_sums(xr, pgr, xr, pgr, n_groups, poly_kernel, device),
        group_sums(xf, pgf, xr, pgr, n_groups, poly_kernel, device),
        counts(pgf, n_groups),
        counts(pgr, n_groups),
    )


def prdc(real, fake, k=5):
    """Improved precision/recall (Kynkäänniemi 2019) and density/coverage (Naeem 2020)."""
    r, f = torch.from_numpy(real).double(), torch.from_numpy(fake).double()
    drr, dff, drf = torch.cdist(r, r), torch.cdist(f, f), torch.cdist(r, f)
    rad_r = drr.kthvalue(k + 1, dim=1).values
    rad_f = dff.kthvalue(k + 1, dim=1).values
    inside = drf < rad_r[:, None]
    return {
        "precision": float(inside.any(0).double().mean()),
        "recall": float((drf < rad_f[None]).any(1).double().mean()),
        "density": float(inside.double().sum(0).mean() / k),
        "coverage": float((drf.min(1).values < rad_r).double().mean()),
    }


def sq_dist(a, b):
    a, b = torch.from_numpy(a).double(), torch.from_numpy(b).double()
    return (torch.cdist(a, b) ** 2).numpy()


def nn_distance(query, ref):
    q, r = torch.from_numpy(query).double(), torch.from_numpy(ref).double()
    d = torch.cdist(q, r)
    v, i = d.min(1)
    return v.numpy(), i.numpy()
