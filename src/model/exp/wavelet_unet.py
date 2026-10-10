"""Frequency-decoupled Patch-CFM U-Net with Haar DWT / IWT.

Design goals (small RC-damage datasets, ~500 images):
  * One shared U-Net (no DiT / Transformer) so patches share parameters.
  * Haar wavelet down/upsampling instead of strided conv / max-pool, so
    LL / LH / HL / HH are *forced* to split before any mixing.
  * HH (diagonal) is the natural band for shear cracks vs orthogonal formwork
    grid lines (mostly LH / HL).
  * Bottleneck ResBlocks see **LL only**; high-frequency subbands skip around
    them and re-enter at IWT.
  * AdaGN injects flow time ``t`` (and optional class / physics embeddings).

Haar 2×2 (invertible, factor 1/2 on both analysis and synthesis)::

    x00 x01      LL = (x00+x01+x10+x11)/2     average / structure
    x10 x11      LH = (x00-x01+x10-x11)/2     horizontal high-freq (vertical edges)
                 HL = (x00+x01-x10-x11)/2     vertical high-freq   (horizontal edges)
                 HH = (x00-x01-x10+x11)/2     diagonal (shear cracks / spall chips)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.adm import (
    AdaGNResBlock,
    _num_groups,
    sinusoidal_timestep_embedding,
)


def _pad_to_even(x):
    """Reflect-pad H/W to even so Haar 2×2 tiling is well-defined (patch-safe)."""
    h, w = x.shape[-2:]
    pad_h = int(h % 2)
    pad_w = int(w % 2)
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, pad_h, pad_w


def haar_dwt2d(x):
    """Discrete Haar wavelet: ``[B, C, H, W]`` → four ``[B, C, H/2, W/2]``.

    Feature split (not a learned conv):
      * ``ll`` continues down the encoder / bottleneck (semantics, background).
      * ``lh, hl, hh`` are stored as skip tensors and **must not** be pooled.
    """
    x, pad_h, pad_w = _pad_to_even(x)
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh, (pad_h, pad_w)


def haar_iwt2d(ll, lh, hl, hh, crop_hw=None):
    """Inverse Haar wavelet: four half-res bands → ``[B, C, 2H, 2W]``.

    Exact inverse of :func:`haar_dwt2d` (same 1/2 factor). ``crop_hw`` drops
    the even-padding added on the analysis side, if any.
    """
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    bsz, ch, h, w = ll.shape
    out = ll.new_empty(bsz, ch, h * 2, w * 2)
    out[..., 0::2, 0::2] = a
    out[..., 0::2, 1::2] = b
    out[..., 1::2, 0::2] = c
    out[..., 1::2, 1::2] = d
    if crop_hw is not None:
        th, tw = int(crop_hw[0]), int(crop_hw[1])
        out = out[..., :th, :tw]
    return out


def haar_highpass_energy(x):
    """Per-sample mean energy of LH+HL+HH (for optional HF-boosted L2, not JPEG)."""
    _ll, lh, hl, hh, _pad = haar_dwt2d(x)
    hf = lh.square() + hl.square() + hh.square()
    return hf.flatten(1).mean(dim=1)


class WaveletDownsample(nn.Module):
    """Freq-aware downsampling: Haar DWT then LL-only channel projection.

    Returns
        ll:  ``[B, out_channels, H/2, W/2]`` — routed into the next encoder stage
        hf:  tuple ``(lh, hl, hh)`` each ``[B, in_channels, H/2, W/2]`` —
             skip-only; **no** strided conv / pooling on these bands
        meta: ``(pad_h, pad_w, src_h, src_w)`` for exact IWT crop
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.ll_proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        src_h, src_w = int(x.shape[-2]), int(x.shape[-1])
        ll, lh, hl, hh, pads = haar_dwt2d(x)
        ll = self.ll_proj(ll)
        return ll, (lh, hl, hh), (pads[0], pads[1], src_h, src_w)


class WaveletUpsample(nn.Module):
    """Freq-aware upsampling: match LL channels to skip HF, then Haar IWT.

    ``hf`` must be the matching encoder skip ``(lh, hl, hh)``.  Decoder LL is
    projected to the skip channel count so IWT is band-wise invertible.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.in_channels = int(in_channels)
        self.skip_channels = int(skip_channels)
        self.out_channels = int(out_channels)
        self.ll_to_skip = (
            nn.Identity()
            if in_channels == skip_channels
            else nn.Conv2d(in_channels, skip_channels, kernel_size=1)
        )
        self.post = (
            nn.Identity()
            if skip_channels == out_channels
            else nn.Conv2d(skip_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, ll, hf, meta):
        lh, hl, hh = hf
        pad_h, pad_w, src_h, src_w = meta
        ll = self.ll_to_skip(ll)
        if ll.shape[1] != lh.shape[1]:
            raise ValueError(
                f"IWT channel mismatch: LL {ll.shape[1]} vs HF {lh.shape[1]}"
            )
        rec = haar_iwt2d(ll, lh, hl, hh, crop_hw=(src_h, src_w))
        return self.post(rec)


class FrequencyBottleneck(nn.Module):
    """Deepest stage: AdaGN ResBlocks on **LL only**; HF is an identity bypass.

    The high-frequency skip from the last DWT is *not* an input here — it is
    consumed later by :class:`WaveletUpsample`.  If a caller still passes ``hf``,
    it is returned unchanged (no ResNet, no attention).
    """

    def __init__(self, channels, cond_dim, num_blocks=2, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                AdaGNResBlock(channels, channels, cond_dim, dropout=dropout)
                for _ in range(max(1, int(num_blocks)))
            ]
        )

    def forward(self, ll, cond, hf=None):
        h = ll
        for block in self.blocks:
            h = block(h, cond)
        return h, hf


class WaveletUNet(nn.Module):
    """Single shared Patch-CFM velocity U-Net with wavelet spatial operators.

    Forward signature matches :class:`ADMUNetCFM` so it can replace
    ``velocity_net`` without changing the OT-CFM training / ODE loop.

    Extra hooks (future damage-type / mask features):
      * ``class_emb``: ``[B, cond_dim]`` added into AdaGN condition.
      * ``seg_feat``:  ``[B, C_seg, H, W]`` concatenated on the spatial input
        when ``seg_channels > 0``.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        model_channels=32,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=1,
        cond_dim=128,
        latent_dim=3,
        dropout=0.0,
        use_damage_mask_condition=False,
        use_split_conditioning=False,
        use_physics_adagn=False,
        physics_dim=1024,
        gated_fusion_alpha_dim="scalar",
        image_channels=None,
        self_cond_channels=0,
        use_undamaged_ref=False,
        undamaged_ref_dim=128,
        seg_channels=0,
        bottleneck_blocks=None,
        **_ignored,
    ):
        super().__init__()
        self.model_channels = int(model_channels)
        self.cond_dim = int(cond_dim)
        self.use_damage_mask_condition = bool(use_damage_mask_condition)
        self.use_split_conditioning = bool(use_split_conditioning)
        self.use_physics_adagn = bool(use_physics_adagn)
        self.physics_dim = int(physics_dim)
        self.use_undamaged_ref = bool(use_undamaged_ref)
        self.undamaged_ref_dim = int(undamaged_ref_dim)
        self.seg_channels = max(0, int(seg_channels))
        self.self_cond_channels = max(0, int(self_cond_channels))
        self.use_physics_cross_attn = False
        self.use_vss_u2 = False
        self.use_vss_l2 = False
        # FreCaS / logging compatibility (no spatial self-attn in this net).
        self.mid_attn = nn.Identity()
        self.mid_attn_heads = 1

        if image_channels is None:
            self.image_channels = int(in_channels)
        else:
            self.image_channels = int(image_channels)

        alpha_mode = str(gated_fusion_alpha_dim).lower().strip()
        if alpha_mode not in ("scalar", "vector"):
            raise ValueError(f"gated_fusion_alpha_dim must be scalar|vector, got {gated_fusion_alpha_dim!r}")
        self.gated_fusion_alpha_dim = alpha_mode

        if use_split_conditioning and use_physics_adagn:
            raise ValueError("use_split_conditioning and use_physics_adagn are mutually exclusive")

        n_levels = len(channel_mult)
        if n_levels < 2:
            raise ValueError(f"channel_mult needs >= 2 stages, got {channel_mult}")

        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        if self.use_split_conditioning:
            self.scalar_mlp = nn.Sequential(
                nn.Linear(latent_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
            self.e_c_ln = nn.LayerNorm(cond_dim)
            self.physics_linear = nn.Linear(physics_dim, cond_dim)
            self.e_g_ln = nn.LayerNorm(cond_dim)
            if self.gated_fusion_alpha_dim == "vector":
                self.alpha = nn.Parameter(torch.zeros(cond_dim))
            else:
                self.alpha = nn.Parameter(torch.zeros(1))
            if self.use_undamaged_ref:
                self.ref_linear = nn.Linear(self.undamaged_ref_dim, cond_dim)
                self.e_ref_ln = nn.LayerNorm(cond_dim)
                if self.gated_fusion_alpha_dim == "vector":
                    self.alpha_ref = nn.Parameter(torch.zeros(cond_dim))
                else:
                    self.alpha_ref = nn.Parameter(torch.zeros(1))
            else:
                self.ref_linear = None
                self.e_ref_ln = None
                self.alpha_ref = None
        elif self.use_physics_adagn:
            self.physics_to_cond = nn.Sequential(
                nn.LayerNorm(physics_dim),
                nn.Linear(physics_dim, cond_dim),
            )
            self.scalar_mlp = nn.Sequential(
                nn.Linear(cond_dim + int(latent_dim), cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
            self.alpha_ref = None
        else:
            self.cond_mlp = nn.Sequential(
                nn.Linear(latent_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
            self.alpha_ref = None
        # Optional class embedding (Crack / Spalling / …) fused into AdaGN cond.
        self.class_proj = nn.Identity()
        self.joint_mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        stem_in = int(in_channels) + self.seg_channels
        if self.use_damage_mask_condition:
            stem_in += 1
        self.in_conv = nn.Conv2d(stem_in, model_channels, kernel_size=3, padding=1)
        with torch.no_grad():
            if self.self_cond_channels > 0:
                s = self.image_channels
                e = self.image_channels + self.self_cond_channels
                self.in_conv.weight[:, s:e].zero_()
            if self.use_damage_mask_condition:
                self.in_conv.weight[:, -1:].zero_()

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = model_channels
        skip_hf_channels = []
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * int(mult)
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(AdaGNResBlock(ch, out_ch, cond_dim, dropout=dropout))
                ch = out_ch
            self.down_blocks.append(blocks)
            if level != n_levels - 1:
                next_ch = model_channels * int(channel_mult[level + 1])
                skip_hf_channels.append(ch)
                self.downsamples.append(WaveletDownsample(ch, next_ch))
                ch = next_ch

        n_bot = int(num_res_blocks if bottleneck_blocks is None else bottleneck_blocks)
        self.bottleneck = FrequencyBottleneck(ch, cond_dim, num_blocks=n_bot, dropout=dropout)

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for level in range(n_levels - 2, -1, -1):
            skip_ch = skip_hf_channels.pop()
            out_ch = model_channels * int(channel_mult[level])
            self.upsamples.append(WaveletUpsample(ch, skip_ch, skip_ch))
            ch = skip_ch
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(AdaGNResBlock(ch, out_ch, cond_dim, dropout=dropout))
                ch = out_ch
            self.up_blocks.append(blocks)

        self.out_norm = nn.GroupNorm(_num_groups(ch), ch, eps=1e-6)
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def _build_cond(self, t, c, gru_feat=None, ref_feat=None, class_emb=None):
        t_emb = self.time_mlp(sinusoidal_timestep_embedding(t, self.model_channels))
        if self.use_split_conditioning:
            if gru_feat is None:
                raise ValueError("gru_feat is required when use_split_conditioning=True")
            e_c = self.e_c_ln(self.joint_mlp(t_emb + self.scalar_mlp(c)))
            e_g = self.e_g_ln(self.physics_linear(gru_feat))
            h = e_c + self.alpha * e_g
            if self.use_undamaged_ref:
                if ref_feat is None:
                    ref_feat = torch.zeros(
                        (c.shape[0], self.undamaged_ref_dim),
                        device=c.device,
                        dtype=c.dtype,
                    )
                h = h + self.alpha_ref * self.e_ref_ln(self.ref_linear(ref_feat))
        elif self.use_physics_adagn:
            if gru_feat is None:
                raise ValueError("gru_feat is required when use_physics_adagn=True")
            phys = self.physics_to_cond(gru_feat)
            h = self.joint_mlp(t_emb + self.scalar_mlp(torch.cat([phys, c], dim=1)))
        else:
            h = self.joint_mlp(t_emb + self.cond_mlp(c))
        if class_emb is not None:
            if class_emb.shape[-1] != h.shape[-1]:
                raise ValueError(
                    f"class_emb dim {class_emb.shape[-1]} must match cond_dim {h.shape[-1]}"
                )
            h = h + class_emb
        return h

    def get_gated_fusion_alpha(self):
        if not self.use_split_conditioning or not hasattr(self, "alpha"):
            return None
        return self.alpha.detach().float().mean()

    def get_undamaged_ref_alpha(self):
        if not self.use_undamaged_ref or getattr(self, "alpha_ref", None) is None:
            return None
        return self.alpha_ref.detach().float().mean()

    def forward(
        self,
        z_t,
        t,
        c,
        gru_feat=None,
        damage_mask=None,
        return_attn_map=False,
        ref_feat=None,
        class_emb=None,
        seg_feat=None,
    ):
        """Predict OT-CFM velocity ``v_θ(z_t, t, c)`` at arbitrary even patch size.

        Tensor flow (example ``H=W=128``, ``channel_mult=(1,2,4,4)``)::

            in_conv:     [B, Cin, 128, 128] → [B, C, 128, 128]
            DWT ×3:      HF skips at 64 / 32 / 16; LL path → [B, 4C, 16, 16]
            bottleneck:  ResBlocks on LL only
            IWT ×3:      fuse skipped (LH,HL,HH) → [B, C, 128, 128]
            out_conv:    [B, Cout, 128, 128]
        """
        cond = self._build_cond(t, c, gru_feat=gru_feat, ref_feat=ref_feat, class_emb=class_emb)
        if self.seg_channels > 0:
            if seg_feat is None:
                seg_feat = z_t.new_zeros(
                    (z_t.shape[0], self.seg_channels, z_t.shape[-2], z_t.shape[-1])
                )
            elif seg_feat.shape[-2:] != z_t.shape[-2:]:
                seg_feat = F.interpolate(seg_feat, size=z_t.shape[-2:], mode="nearest")
            z_t = torch.cat([z_t, seg_feat], dim=1)
        if self.use_damage_mask_condition:
            if damage_mask is None:
                damage_mask = z_t.new_zeros((z_t.shape[0], 1, z_t.shape[-2], z_t.shape[-1]))
            elif damage_mask.ndim == 3:
                damage_mask = damage_mask.unsqueeze(1)
            if damage_mask.shape[-2:] != z_t.shape[-2:]:
                damage_mask = F.interpolate(damage_mask, size=z_t.shape[-2:], mode="nearest")
            z_t = torch.cat([z_t, damage_mask], dim=1)

        h = self.in_conv(z_t)
        hf_skips = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, cond)
            if i < len(self.downsamples):
                # Split: LL continues; (LH, HL, HH) go to the matching IWT skip.
                h, hf, meta = self.downsamples[i](h)
                hf_skips.append((hf, meta))

        h, _ = self.bottleneck(h, cond, hf=None)

        for up, blocks in zip(self.upsamples, self.up_blocks):
            hf, meta = hf_skips.pop()
            h = up(h, hf, meta)
            for block in blocks:
                h = block(h, cond)

        if hf_skips:
            raise RuntimeError("Wavelet skip stack not empty — encoder/decoder depth mismatch")

        out = self.out_conv(F.silu(self.out_norm(h)))
        if return_attn_map:
            return out, {}
        return out


def wavelet_velocity_loss(pred, target, valid_mask=None, hf_boost=0.0):
    """Equal-weight L2 on velocity, plus optional Haar-HF residual boost.

    ``hf_boost=0`` → plain MSE (high and low frequencies treated equally).
    ``hf_boost>0`` → extra MSE on LH/HL/HH of ``(pred-target)`` (not JPEG-style
    down-weighting of high frequencies).
    """
    diff = pred - target
    if valid_mask is None:
        base = diff.square().flatten(1).mean(dim=1)
    else:
        vm = valid_mask.to(device=diff.device, dtype=diff.dtype)
        if vm.ndim == 3:
            vm = vm.unsqueeze(1)
        if vm.shape[-2:] != diff.shape[-2:]:
            vm = F.interpolate(vm, size=diff.shape[-2:], mode="nearest")
        denom = (vm.sum(dim=(1, 2, 3)) * diff.shape[1]).clamp_min(1e-8)
        base = (diff.square() * vm).sum(dim=(1, 2, 3)) / denom
    w = float(hf_boost)
    if w <= 0.0:
        return base
    hf = haar_highpass_energy(diff)
    return base + w * hf
