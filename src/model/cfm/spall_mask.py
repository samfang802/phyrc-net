"""Coarse accumulated damage mask.

``M`` predicts the cells that are already darker than a fixed luminance, not
the per-frame increment and not a per-image max. The soft-paint path sends
``G`` a feathered weight of the pixels it may change, and puts that same
weight on the OT path. The older path still sends ``relu(m_t - m_{t-1})``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def random_shift(x, max_shift):
    """Per-sample integer roll. ``max_shift`` is in pixels; 0 is a no-op."""
    max_shift = int(max_shift)
    if max_shift <= 0:
        return x
    shifts = torch.randint(-max_shift, max_shift + 1, (x.shape[0], 2), device=x.device)
    out = []
    for i in range(x.shape[0]):
        dy = int(shifts[i, 0].item())
        dx = int(shifts[i, 1].item())
        out.append(torch.roll(x[i : i + 1], shifts=(dy, dx), dims=(-2, -1)))
    return torch.cat(out, dim=0)


def to_luma(x):
    if x.shape[1] == 1:
        return x
    return x.mean(dim=1, keepdim=True)


def coarse_pool(x, grid):
    """Average-pool to a ``grid×grid`` luminance map. Darker pixels stay darker."""
    grid = int(grid)
    if grid < 2:
        raise ValueError(f"spall mask grid must be >= 2, got {grid}")
    return F.adaptive_avg_pool2d(to_luma(x), (grid, grid))


def _edge_backdrop(aligned, bg_target, border_gap):
    """Left and right edge cells that are the black backdrop, not concrete."""
    backdrop = aligned < (float(bg_target) - float(border_gap))
    edge = torch.zeros_like(aligned, dtype=torch.bool)
    edge[:, :, :, 0] = True
    edge[:, :, :, -1] = True
    return edge & backdrop


def _largest_component(binary):
    """Keep the largest 4-connected foreground component. Empty input stays empty."""
    on = binary > 0.5
    b, _, height, width = binary.shape
    n = height * width
    labels = torch.arange(n, device=binary.device, dtype=binary.dtype).view(1, 1, height, width)
    labels = labels.expand(b, 1, height, width).clone()
    sentinel = binary.new_tensor(float(n))
    labels = torch.where(on, labels, sentinel)
    for _ in range(height + width):
        padded = F.pad(labels, (1, 1, 1, 1), value=float(n))
        spread = -F.max_pool2d(-padded, kernel_size=3, stride=1, padding=0)
        labels = torch.where(on, spread, sentinel)
    out = torch.zeros_like(binary)
    for i in range(b):
        valid = on[i, 0]
        if not bool(valid.any()):
            continue
        uniq, counts = torch.unique(labels[i, 0][valid], return_counts=True)
        best = uniq[int(counts.argmax())]
        out[i, 0] = (labels[i, 0] == best).to(dtype=binary.dtype)
    return out


def _top_row(binary):
    """Topmost foreground row per sample, shape ``[B,1,1,1]``. ``height`` if empty."""
    height = binary.shape[-2]
    rows = torch.arange(height, device=binary.device, dtype=binary.dtype).view(1, 1, height, 1)
    present = binary > 0.5
    filled = torch.where(present, rows.expand_as(binary), rows.new_full(binary.shape, float(height)))
    return filled.amin(dim=(-2, -1), keepdim=True)


def _bulk_top(binary, min_width=4):
    """Highest row that is at least ``min_width`` cells wide. Falls back to any cell."""
    counts = (binary > 0.5).to(dtype=binary.dtype).sum(dim=-1, keepdim=True)
    height = binary.shape[-2]
    rows = torch.arange(height, device=binary.device, dtype=binary.dtype).view(1, 1, height, 1)
    wide = counts >= float(min_width)
    top = torch.where(wide, rows, rows.new_full(counts.shape, float(height))).amin(dim=-2, keepdim=True)
    fallback = _top_row(binary)
    return torch.where(top >= (height - 0.5), fallback, top)


def _at_or_below(mask, top):
    height = mask.shape[-2]
    rows = torch.arange(height, device=mask.device, dtype=mask.dtype).view(1, 1, height, 1)
    return mask * (rows >= top).to(dtype=mask.dtype)


def _grow_from_seed(seed, allowed):
    """Flood ``seed`` through ``allowed``. Neither map gains cells outside ``allowed``."""
    region = seed
    steps = int(seed.shape[-2] + seed.shape[-1])
    for _ in range(steps):
        region = torch.maximum(region, F.max_pool2d(region, kernel_size=3, stride=1, padding=1) * allowed)
    return (region > 0.5).to(dtype=seed.dtype)


def ground_short_gap(mask, max_gap=2):
    """Fill a bright chunk that cuts the spall off the base.

    Only columns whose foreground already stops within ``max_gap`` cells of the
    bottom are eligible. A cell in that gap is filled only when a horizontal
    neighbor is already foreground, so a full-width footing with nothing beside
    it stays empty. ``max_gap <= 0`` is a no-op.
    """
    max_gap = int(max_gap)
    if max_gap <= 0:
        return mask
    present = mask > 0.5
    height = mask.shape[-2]
    rows = torch.arange(height, device=mask.device).view(1, 1, height, 1)
    bottom = torch.where(
        present,
        rows.expand_as(mask),
        rows.new_full(mask.shape, -1.0),
    ).amax(dim=-2, keepdim=True)
    gap = (height - 1) - bottom
    short = (bottom >= 0) & (gap > 0) & (gap <= max_gap)
    zone = (rows > bottom) & short
    region = present
    for _ in range(mask.shape[-1]):
        side = F.max_pool2d(region.to(dtype=mask.dtype), kernel_size=(1, 3), stride=1, padding=(0, 1))
        region = region | (zone & (side > 0.5))
    return region.to(dtype=mask.dtype)


def extent_mask(
    x,
    grid,
    dark_level=0.0,
    bg_quantile=0.70,
    bg_target=0.30,
    border_gap=0.55,
    seed_margin=0.2,
    ground_gap=2,
):
    """Coarse damage cells, capped at the top of the darkest connected core.

    A cell is aligned by shifting the per-image ``bg_quantile`` to ``bg_target``.
    The seed is ``aligned < dark_level - seed_margin``. Only the largest seed
    component is kept, and its top is the highest row that is at least four cells
    wide, so a one-cell spike cannot lift the whole mask. Mildly dark cells (``aligned < dark_level``) count only
    when they touch that seed and sit at or below its top row, so cracked cover
    above the spall is not recruited. Holes inside that region are filled.
    Nothing is painted above the seed. A bright chunk under the spall is filled
    only when it is at most ``ground_gap`` cells tall and a neighbor column
    already reaches the base.
    """
    pool = coarse_pool(x, grid)
    bg = torch.quantile(pool.flatten(1), float(bg_quantile), dim=1).view(-1, 1, 1, 1)
    aligned = pool - bg + float(bg_target)
    drop = _edge_backdrop(aligned, bg_target, border_gap)
    halo = ((aligned < float(dark_level)) & ~drop).to(dtype=pool.dtype)
    seed = ((aligned < float(dark_level) - float(seed_margin)) & ~drop).to(dtype=pool.dtype)
    seed = _largest_component(seed)
    top = _bulk_top(seed, min_width=4)
    halo = _at_or_below(halo, top)
    grown = _grow_from_seed(seed, torch.maximum(seed, halo))
    filled = _at_or_below(fill_holes(grown), top)
    return ground_short_gap(filled, max_gap=ground_gap)


def dilate_mask(m, k=3):
    """Binary dilation. An isolated 1 becomes a ``k×k`` block."""
    k = int(k)
    if k <= 1:
        return m
    if k % 2 == 0:
        raise ValueError(f"dilation kernel must be odd, got {k}")
    return F.max_pool2d(m, kernel_size=k, stride=1, padding=k // 2)


def erode_mask(m, k=3):
    """Binary erosion. ``k<=1`` is a no-op."""
    k = int(k)
    if k <= 1:
        return m
    if k % 2 == 0:
        raise ValueError(f"erosion kernel must be odd, got {k}")
    return -F.max_pool2d(-m, kernel_size=k, stride=1, padding=k // 2)


def fill_holes(m):
    """Turn enclosed background into foreground. Border-connected zeros stay open."""
    inv = 1.0 - (m > 0.5).to(dtype=m.dtype)
    bg = torch.zeros_like(inv)
    bg[:, :, 0, :] = inv[:, :, 0, :]
    bg[:, :, -1, :] = inv[:, :, -1, :]
    bg[:, :, :, 0] = torch.maximum(bg[:, :, :, 0], inv[:, :, :, 0])
    bg[:, :, :, -1] = torch.maximum(bg[:, :, :, -1], inv[:, :, :, -1])
    steps = int(m.shape[-1] + m.shape[-2])
    for _ in range(steps):
        bg = torch.minimum(F.max_pool2d(bg, kernel_size=3, stride=1, padding=1), inv)
    return 1.0 - bg


def solidify_extent(m, close_k=3):
    """Close small gaps and fill enclosed holes, without growing past the top row.

    Closing is dilation followed by erosion. The mask is cut back to the
    topmost cell of the input, so neither closing nor hole filling can climb
    toward the upper edge.
    """
    binary = (m > 0.5).to(dtype=m.dtype)
    top = _bulk_top(binary, min_width=4)
    binary = _at_or_below(binary, top)
    k = int(close_k)
    closed = erode_mask(dilate_mask(binary, k), k) if k > 1 else binary
    closed = _at_or_below(closed, top)
    return _at_or_below(fill_holes(closed), top)


def growth_from_extent(m_prob, m_prev, threshold, dilate_k=1):
    """``relu(m_t - m_{t-1})``. ``dilate_k`` <= 1 leaves the new cells unexpanded."""
    grow = F.relu(hard_mask(m_prob, threshold) - m_prev)
    return dilate_mask(grow, dilate_k)


def soft_dice_loss(pred, target, eps=1.0):
    """Soft Dice on the coarse grid. Empty pred and empty target give 0."""
    dims = tuple(range(1, pred.ndim))
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * inter + float(eps)) / (denom + float(eps))).mean()


def neighbor_tv(m):
    """Mean absolute difference between 4-neighbors on the coarse grid."""
    dy = (m[:, :, 1:, :] - m[:, :, :-1, :]).abs().mean()
    dx = (m[:, :, :, 1:] - m[:, :, :, :-1]).abs().mean()
    return dy + dx


def repaint_known(z, z0, x_known, t, known):
    """Put the known region back on its OT path toward ``x_known``.

    ``known`` is 1 on locked pixels. At ``t=0`` this is ``z0``; at ``t=1`` it
    is ``x_known``. Free pixels are left as ``z``.
    """
    if known is None:
        return z
    if not torch.is_tensor(t):
        t = torch.tensor(float(t), device=z.device, dtype=z.dtype)
    else:
        t = t.to(device=z.device, dtype=z.dtype)
    if t.ndim == 0:
        t_view = t
    else:
        t_view = t.reshape(-1, *([1] * (z.ndim - 1)))
    known = known.to(device=z.device, dtype=z.dtype)
    z_known = (1.0 - t_view) * z0 + t_view * x_known
    return known * z_known + (1.0 - known) * z


def upsample_mask(m, size):
    return F.interpolate(m, size=size, mode="nearest")


def coarse_coords(coords, grid, ref):
    """Pool patch XY to the mask grid, or build a full-frame grid when coords are absent."""
    grid = int(grid)
    if coords is None:
        axis = torch.linspace(-1.0, 1.0, grid, device=ref.device, dtype=ref.dtype)
        gy, gx = torch.meshgrid(axis, axis, indexing="ij")
        base = torch.stack([gx, gy], dim=0).unsqueeze(0)
        return base.expand(ref.shape[0], -1, -1, -1).contiguous()
    return F.adaptive_avg_pool2d(coords, (grid, grid))


def _gaussian_blur(x, sigma):
    """Separable Gaussian. ``sigma <= 0`` returns ``x`` unchanged."""
    sigma = float(sigma)
    if sigma <= 0.0:
        return x
    k = int(2 * round(3.0 * sigma) + 1)
    k = max(3, min(k, 31))
    if k % 2 == 0:
        k += 1
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - (k // 2)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum()
    pad = k // 2
    blurred = F.conv2d(x, kernel.view(1, 1, 1, k), padding=(0, pad))
    return F.conv2d(blurred, kernel.view(1, 1, k, 1), padding=(pad, 0))


def feather_mask(m, size, sigma):
    """Bilinear upsample, then a Gaussian so the feature channel has a soft edge."""
    up = F.interpolate(m, size=size, mode="bilinear", align_corners=False)
    return _gaussian_blur(up, sigma)


def core_feather(binary, size, sigma):
    """Nearest upsample. The editable core stays 1. The Gaussian only adds an outside tail."""
    up = F.interpolate(binary, size=size, mode="nearest")
    if float(sigma) <= 0.0:
        return up
    return torch.maximum(up, _gaussian_blur(up, sigma))


def context_ot(x, s_max):
    """Push ``x`` a random distance along the OT path toward fresh noise.

    ``s`` is drawn uniformly from ``[0, s_max]`` per sample. ``s_max <= 0`` is a no-op.
    """
    s_max = float(s_max)
    if s_max <= 0.0:
        return x
    s = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype) * s_max
    return (1.0 - s) * x + s * torch.randn_like(x)


def soft_paint_alpha(
    x_prev,
    m_prob,
    m_prev,
    spatial,
    grid,
    threshold,
    close_k,
    bg_quantile,
    bg_target,
    border_gap,
    healthy_margin,
    sigma,
    lock_healthy=True,
):
    """Feathered weight of pixels ``G`` may change. Interior of that region stays 1.

    A coarse cell is locked when it is previous-frame spall, a left/right black
    backdrop, or, when ``lock_healthy`` is set, healthy concrete outside the
    current solid extent. Healthy means ``pool >=`` this frame's own
    ``bg_quantile`` minus ``healthy_margin``. Dark cells that are not yet spall
    stay paintable, so cracks are not frozen.
    """
    pool = coarse_pool(x_prev, grid)
    bg = torch.quantile(pool.flatten(1), float(bg_quantile), dim=1).view(-1, 1, 1, 1)
    aligned = pool - bg + float(bg_target)
    drop = _edge_backdrop(aligned, bg_target, border_gap).to(dtype=pool.dtype)
    if lock_healthy:
        healthy = (pool >= (bg - float(healthy_margin))).to(dtype=pool.dtype)
    else:
        healthy = torch.zeros_like(pool)
    extent = solidify_extent(hard_mask(m_prob, threshold), close_k=close_k)
    locked = torch.maximum(m_prev, torch.maximum(drop, healthy * (1.0 - extent)))
    paint = 1.0 - locked
    return core_feather(paint, spatial, sigma)


def hard_mask(m_prob, threshold):
    """Values below ``threshold`` become exact zeros. No gradient."""
    return (m_prob.detach() >= float(threshold)).to(dtype=m_prob.dtype)


class SpallMaskHead(nn.Module):
    """Coarse accumulated extent.

    The previous pool can mark damage that is already dark. Coordinates carry
    the shape of new damage, multiplied by a scalar gate from ``[g_{t-1}; g_t]``.
    The gate starts nearly closed, so a fixed bottom map is not on at low damage.
    """

    def __init__(self, physics_dim, grid=16, g_channels=32):
        super().__init__()
        self.grid = int(grid)
        del g_channels
        if self.grid < 2:
            raise ValueError(f"spall mask grid must be >= 2, got {self.grid}")
        if int(physics_dim) <= 0:
            raise ValueError(f"physics_dim must be positive, got {physics_dim}")
        self.g_gate = nn.Linear(int(physics_dim) * 2, 1)
        nn.init.zeros_(self.g_gate.weight)
        nn.init.constant_(self.g_gate.bias, -2.0)
        self.pool_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.coord_conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, pool, g_prev, g_curr, coords):
        """Return logits ``[B,1,G,G]``. ``pool`` is ``[B,1,G,G]``, ``coords`` is ``[B,2,G,G]``."""
        if pool.shape[-2:] != (self.grid, self.grid):
            raise ValueError(
                f"pool spatial {tuple(pool.shape[-2:])} must be {(self.grid, self.grid)}"
            )
        if coords.shape[-2:] != (self.grid, self.grid) or coords.shape[1] != 2:
            raise ValueError(
                f"coords must be [B,2,{self.grid},{self.grid}], got {tuple(coords.shape)}"
            )
        if g_prev.shape != g_curr.shape:
            raise ValueError(
                f"g_prev {tuple(g_prev.shape)} must match g_curr {tuple(g_curr.shape)}"
            )
        local = self.pool_conv(pool)
        template = self.coord_conv(coords.to(dtype=pool.dtype))
        gate = torch.sigmoid(self.g_gate(torch.cat([g_prev, g_curr], dim=1)))
        gate = gate.view(-1, 1, 1, 1).to(dtype=pool.dtype)
        return local + gate * template
