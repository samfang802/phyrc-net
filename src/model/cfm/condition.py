"""Condition builders and CLI value parsers for CFM generation."""

import torch


def build_generation_condition(
    info,
    hidden_state,
    direction,
    DI,
    latent_conditioning,
    direction_conditioning,
    include_di_conditioning,
    condition_mode="full",
    zero_di_condition=False,
    keep_zero_di_slot=False,
):
    mode = str(condition_mode).lower()
    include_di = scalar_condition_includes_di(
        include_di_conditioning,
        zero_di_condition,
        keep_zero_di_slot=keep_zero_di_slot,
    )
    di = torch.zeros_like(DI) if (zero_di_condition or not include_di) else DI
    if mode == "di_only":
        return di if include_di else torch.zeros_like(DI)
    if mode in ("split", "adagn"):
        # Scalar AdaGN path: optional DI + direction. Physics g is fused separately.
        # zero_di_condition drops the DI channel (does not keep a dead zero pad).
        return build_scalar_condition(
            direction,
            di,
            direction_conditioning,
            include_di=include_di,
        )
    if mode == "latent_direction":
        return torch.cat([hidden_state, direction], dim=1)

    if latent_conditioning:
        parts = [hidden_state]
        if direction_conditioning:
            parts.append(direction)
        if include_di:
            parts.append(di)
        return torch.cat(parts, dim=1)

    parts = [info]
    if direction_conditioning:
        parts.append(direction)
    if include_di:
        parts.append(di)
    return torch.cat(parts, dim=1)


def scalar_condition_includes_di(
    include_di_conditioning, zero_di_condition, keep_zero_di_slot=False
):
    """DI enters e_c when requested. zero_di drops the channel unless keep_zero_di_slot."""
    if not bool(include_di_conditioning):
        return False
    if bool(zero_di_condition) and not bool(keep_zero_di_slot):
        return False
    return True


def build_scalar_condition(direction, DI, direction_conditioning=True, include_di=True):
    """Scalar path for split/adagn: optional DI + direction → AdaGN."""
    parts = []
    if include_di:
        parts.append(DI)
    if direction_conditioning:
        parts.append(direction)
    if not parts:
        raise ValueError("scalar condition is empty: enable DI or direction")
    return torch.cat(parts, dim=1)


def build_physics_condition(hidden_state, condition_mode="full"):
    """Physics path: GRU hidden g for split (gated AdaGN) or adagn (concat AdaGN)."""
    if str(condition_mode).lower() in ("split", "adagn"):
        return hidden_state
    return None


def parse_fixed_di_values(raw_values, fallback_count):
    if isinstance(raw_values, (list, tuple)):
        parsed = [float(v) for v in raw_values]
    else:
        parsed = []
        for token in str(raw_values).split(","):
            token = token.strip()
            if not token:
                continue
            parsed.append(float(token))
    if not parsed:
        parsed = [0.0, 0.4, 0.75, 0.95]
    if len(parsed) < fallback_count:
        parsed.extend([parsed[-1]] * (fallback_count - len(parsed)))
    return parsed[:fallback_count]


def parse_patch_size_probs(raw_values, expected_count=3):
    """Parse stochastic patch schedule probabilities (full / half / quarter)."""
    if isinstance(raw_values, (list, tuple)):
        parsed = [float(v) for v in raw_values]
    else:
        parsed = []
        for token in str(raw_values).split(","):
            token = token.strip()
            if not token:
                continue
            parsed.append(float(token))
    if len(parsed) != expected_count:
        raise ValueError(
            f"patch_size_probs must have {expected_count} values (got {len(parsed)}): {raw_values}"
        )
    if any(p < 0.0 for p in parsed):
        raise ValueError(f"patch_size_probs must be non-negative: {parsed}")
    total = sum(parsed)
    if total <= 0.0:
        raise ValueError(f"patch_size_probs must sum to a positive value: {parsed}")
    return [p / total for p in parsed]


def interpolate_patch_size_probs(probs_a, probs_b, alpha):
    """Linear mix of two (full, half, quarter) schedules, then renormalize."""
    a = parse_patch_size_probs(probs_a)
    b = parse_patch_size_probs(probs_b)
    t = min(1.0, max(0.0, float(alpha)))
    mixed = [(1.0 - t) * x + t * y for x, y in zip(a, b)]
    return parse_patch_size_probs(mixed)


def patch_pixel_mass(probs):
    """Expected spatial mass vs a full image: 1, 1/4, 1/16 for the three sizes."""
    p_full, p_half, p_quarter = parse_patch_size_probs(probs)
    return p_full + 0.25 * p_half + 0.0625 * p_quarter


def probs_from_pixel_mass(mass, p_quarter=0.0):
    """Invert M = p_full + 0.25 p_half + 0.0625 p_quarter with a fixed p_quarter.

    Remaining mass goes to full vs half. ``mass`` is clamped to the feasible
    interval for the given quarter probability.
    """
    p_q = min(1.0, max(0.0, float(p_quarter)))
    mass_min = 0.25 - 0.1875 * p_q
    mass_max = 1.0 - 0.9375 * p_q
    mass_c = min(mass_max, max(mass_min, float(mass)))
    p_full = (mass_c - 0.25 + 0.1875 * p_q) / 0.75
    p_half = 1.0 - p_full - p_q
    return parse_patch_size_probs([p_full, p_half, p_q])


def patch_size_probs_pixel_mass_at_step(
    step,
    mass_start,
    mass_end,
    quarter_start,
    quarter_end,
    total_steps,
    enabled=True,
    fallback_probs=None,
):
    """Linear interpolate pixel mass M over [0, total_steps]; quarter ramps separately."""
    if not enabled:
        return parse_patch_size_probs(fallback_probs if fallback_probs is not None else "0.5,0.3,0.2")
    total_steps = max(1, int(total_steps))
    step = int(step)
    alpha = 0.0 if step <= 0 else (1.0 if step >= total_steps else float(step) / float(total_steps))
    mass = (1.0 - alpha) * float(mass_start) + alpha * float(mass_end)
    p_q = (1.0 - alpha) * float(quarter_start) + alpha * float(quarter_end)
    return probs_from_pixel_mass(mass, p_q)


def boost_half_by_cfm_t(probs, t_mean, t_min=0.5, extra_half=0.20):
    """Move probability from full → half as OT-CFM t goes from t_min to 1.

    CFM: t≈0 is noise (keep global / full), t≈1 is clean (allow more 64px half).
    Quarter is never increased. ``extra_half`` is the additional p_half at t=1,
    taken from p_full and clamped so p_full stays non-negative.
    """
    p_full, p_half, p_q = parse_patch_size_probs(probs)
    extra = max(0.0, float(extra_half))
    t_lo = min(1.0, max(0.0, float(t_min)))
    t = min(1.0, max(0.0, float(t_mean)))
    if extra <= 0.0 or t <= t_lo:
        return [p_full, p_half, p_q]
    alpha = (t - t_lo) / max(1e-8, 1.0 - t_lo)
    delta = min(p_full, extra * alpha)
    return parse_patch_size_probs([p_full - delta, p_half + delta, p_q])


def patch_size_probs_at_step(
    step,
    probs_start,
    probs_mid,
    probs_end,
    mid_step,
    end_start_step,
    total_steps,
    enabled=True,
):
    """Two-segment linear patch schedule with a mid plateau.

    [0, mid_step]: start → mid
    (mid_step, end_start_step]: hold mid
    (end_start_step, total_steps]: mid → end
    """
    mid = parse_patch_size_probs(probs_mid)
    if not enabled:
        return mid
    start = parse_patch_size_probs(probs_start)
    end = parse_patch_size_probs(probs_end)
    step = int(step)
    mid_step = max(0, int(mid_step))
    end_start_step = max(mid_step, int(end_start_step))
    total_steps = max(end_start_step, int(total_steps))
    if step <= 0:
        return start if mid_step > 0 else mid
    if mid_step > 0 and step < mid_step:
        return interpolate_patch_size_probs(start, mid, float(step) / float(mid_step))
    if step <= end_start_step or total_steps <= end_start_step:
        return mid
    if step >= total_steps:
        return end
    span = float(total_steps - end_start_step)
    return interpolate_patch_size_probs(mid, end, float(step - end_start_step) / span)


# Back-compat private aliases used by older call sites.
_parse_fixed_di_values = parse_fixed_di_values
_parse_patch_size_probs = parse_patch_size_probs
