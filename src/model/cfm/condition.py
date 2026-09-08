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
):
    mode = str(condition_mode).lower()
    di = torch.zeros_like(DI) if zero_di_condition else DI
    if mode == "di_only":
        return di
    if mode in ("split", "adagn"):
        # Scalar AdaGN path: DI (+ direction). Physics g is fused separately.
        # Ablation: zero_di_condition keeps the DI channel but fills it with 0.
        return build_scalar_condition(direction, di, direction_conditioning)
    if mode == "latent_direction":
        return torch.cat([hidden_state, direction], dim=1)

    if latent_conditioning:
        parts = [hidden_state]
        if direction_conditioning:
            parts.append(direction)
        if include_di_conditioning:
            parts.append(di)
        return torch.cat(parts, dim=1)

    parts = [info]
    if direction_conditioning:
        parts.append(direction)
    parts.append(di)
    return torch.cat(parts, dim=1)


def build_scalar_condition(direction, DI, direction_conditioning=True):
    """Scalar path for split/adagn: DI (+ direction) → AdaGN."""
    parts = [DI]
    if direction_conditioning:
        parts.append(direction)
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


# Back-compat private aliases used by older call sites.
_parse_fixed_di_values = parse_fixed_di_values
_parse_patch_size_probs = parse_patch_size_probs
