import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.Component.vss_block import VSSBlock2d
from src.model.Component.freeu import freeu_fuse, freeu_params_for_stage


def _interpolate_self_attn_map(attn, src_hw, dst_hw):
    """Bilinear-resize self-attention ``[B, heads, sq, sk]`` to ``dst_hw`` token grid."""
    if attn.ndim != 4:
        raise ValueError(f"Expected attn [B, heads, sq, sk], got {tuple(attn.shape)}")
    sh, sw = int(src_hw[0]), int(src_hw[1])
    dh, dw = int(dst_hw[0]), int(dst_hw[1])
    sq, sk = int(attn.shape[-2]), int(attn.shape[-1])
    if (sh, sw) == (dh, dw) and sq == sh * sw and sk == sh * sw:
        return attn
    b, heads = attn.shape[0], attn.shape[1]
    x = attn.reshape(b * heads, 1, sq, sk)
    x = F.interpolate(
        x,
        size=(dh * dw, dh * dw),
        mode="bilinear",
        align_corners=False,
    )
    return x.reshape(b, heads, dh * dw, dh * dw)


def _num_groups(channels, max_groups=32):
    """Pick the largest GroupNorm group count that divides ``channels``."""
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(1, groups)


def _expand_conv_in_with_zero_mask_channel(conv_in):
    """Expand conv_in from C to C+1 channels with zero-init extra channel."""
    out_ch, in_ch, kh, kw = conv_in.weight.shape
    expanded = nn.Conv2d(in_ch + 1, out_ch, kernel_size=conv_in.kernel_size, stride=conv_in.stride, padding=conv_in.padding)
    with torch.no_grad():
        expanded.weight[:, :in_ch, :, :] = conv_in.weight
        expanded.weight[:, in_ch:, :, :] = 0.0
        if conv_in.bias is not None:
            expanded.bias.copy_(conv_in.bias)
    return expanded


def sinusoidal_timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(half, 1))
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class SelfAttention2d(nn.Module):
    """Spatial self-attention with optional multi-head split over channels."""

    def __init__(self, channels, num_heads=1):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"SelfAttention2d channels={channels} must be divisible by num_heads={num_heads}"
            )
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.channels // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        # Optional capture / FreCaS Stage-1 → Stage-2 attention reuse.
        self.store_attn = False
        self.last_attn = None
        self.last_hw = None
        self._ref_attn = None
        self._attn_fusion_weight = None

    def set_reference_attn(self, ref_attn, fusion_weight=0.6):
        """Install Stage-1 mid self-attention for linear fusion in Stage-2."""
        self._ref_attn = ref_attn
        self._attn_fusion_weight = float(fusion_weight)

    def clear_reference_attn(self):
        self._ref_attn = None
        self._attn_fusion_weight = None

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {c}")

        h_in = x
        x = self.norm(x)
        q, k, v = torch.chunk(self.qkv(x), 3, dim=1)
        seq = h * w
        q = q.reshape(b, self.num_heads, self.head_dim, seq)
        k = k.reshape(b, self.num_heads, self.head_dim, seq)
        v = v.reshape(b, self.num_heads, self.head_dim, seq)

        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = torch.softmax(attn, dim=-1)

        ref_attn = self._ref_attn
        w_c = self._attn_fusion_weight
        if ref_attn is not None and w_c is not None:
            if ref_attn.shape[0] == 1 and b > 1:
                ref_attn = ref_attn.expand(b, -1, -1, -1)
            if ref_attn.shape[-2:] != attn.shape[-2:]:
                src_side = int(round(ref_attn.shape[-2] ** 0.5))
                src_hw = self.last_hw or (src_side, src_side)
                ref_attn = _interpolate_self_attn_map(ref_attn, src_hw, (h, w))
            attn = w_c * ref_attn.to(device=attn.device, dtype=attn.dtype) + (
                1.0 - w_c
            ) * attn

        if self.store_attn:
            self.last_attn = attn.detach()
            self.last_hw = (int(h), int(w))

        out = torch.matmul(attn, v.transpose(-2, -1))
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        out = self.proj(out)
        return h_in + out


class CrossAttention2d(nn.Module):
    """Spatial cross-attention: Q from image features, K/V from context token sequence."""

    def __init__(self, channels, context_dim, num_heads=4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"CrossAttention2d channels={channels} must be divisible by num_heads={num_heads}"
            )
        self.channels = int(channels)
        self.context_dim = int(context_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.channels // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Linear(context_dim, channels)
        self.v_proj = nn.Linear(context_dim, channels)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        # Optional capture for visualization hooks (attn: [B, heads, H*W, L]).
        self.store_attn = False
        self.last_attn = None
        self.last_hw = None

    def forward(self, x, context):
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {c}")
        if context.ndim != 3 or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"Expected context [B, L, {self.context_dim}], got {tuple(context.shape)}"
            )

        residual = x
        x = self.norm(x)
        seq = h * w
        q = self.q(x).reshape(b, self.num_heads, self.head_dim, seq)

        k = self.k_proj(context).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = self.v_proj(context).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        if self.store_attn:
            # Detach for hooks / heatmaps; shape [B, heads, H*W, L_context].
            self.last_attn = attn.detach()
            self.last_hw = (int(h), int(w))
        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        out = self.proj(out)
        return residual + out


class AdaGNResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels, eps=1e-6)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels, eps=1e-6)

        self.ada1 = nn.Linear(cond_dim, in_channels * 2)
        self.ada2 = nn.Linear(cond_dim, out_channels * 2)
        nn.init.zeros_(self.ada1.weight)
        nn.init.zeros_(self.ada1.bias)
        nn.init.zeros_(self.ada2.weight)
        nn.init.zeros_(self.ada2.bias)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def _apply_adagn(self, h, cond_proj):
        gamma, beta = torch.chunk(cond_proj, 2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return F.silu(h) * (gamma + 1.0) + beta

    def forward(self, x, cond):
        h = self.norm1(x)
        h = self._apply_adagn(h, self.ada1(cond))
        h = self.conv1(h)

        h = self.norm2(h)
        h = self._apply_adagn(h, self.ada2(cond))
        h = self.dropout(h)
        h = self.conv2(h)
        return self.skip(x) + h


class ADMUNetCFM(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        cond_dim=512,
        latent_dim=512,
        dropout=0.0,
        use_mid_attention=True,
        mid_attn_heads=4,
        use_vss_l3_u3=True,
        use_vss_u2=False,
        use_vss_l2=False,
        vss_d_state=16,
        vss_expand=2,
        use_u1_extra_resblock=False,
        use_damage_mask_condition=False,
        use_split_conditioning=False,
        use_physics_adagn=False,
        use_physics_cross_attn=False,
        physics_dim=1024,
        physics_num_tokens=16,
        physics_token_dim=128,
        physics_adagn_dim=None,
        gated_fusion_alpha_dim="scalar",
        image_channels=None,
        self_cond_channels=0,
        use_undamaged_ref=False,
        undamaged_ref_dim=128,
        use_freeu=False,
        freeu_b1=1.2,
        freeu_b2=1.4,
        freeu_s1=0.9,
        freeu_s2=0.2,
        freeu_r_thresh=0.25,
    ):
        super().__init__()
        if use_split_conditioning and use_physics_adagn:
            raise ValueError("use_split_conditioning and use_physics_adagn are mutually exclusive")
        self.model_channels = model_channels
        self.cond_dim = cond_dim
        self.use_damage_mask_condition = bool(use_damage_mask_condition)
        self.use_split_conditioning = bool(use_split_conditioning)
        self.use_physics_adagn = bool(use_physics_adagn)
        self.use_physics_cross_attn = bool(use_physics_cross_attn)
        self.physics_dim = int(physics_dim)
        self.physics_num_tokens = int(physics_num_tokens)
        self.physics_token_dim = int(physics_token_dim)
        self.physics_adagn_dim = int(cond_dim if physics_adagn_dim is None else physics_adagn_dim)
        self.use_undamaged_ref = bool(use_undamaged_ref)
        self.undamaged_ref_dim = int(undamaged_ref_dim)
        if self.use_undamaged_ref and not self.use_split_conditioning:
            raise ValueError(
                "use_undamaged_ref requires condition_mode=split so the 128-d "
                "ref embedding gets an independent zero-init alpha_ref (not concat onto c)."
            )
        self.mid_attn_heads = max(1, int(mid_attn_heads))
        self.use_vss_l3_u3 = bool(use_vss_l3_u3)
        self.use_vss_u2 = bool(use_vss_u2)
        self.use_vss_l2 = bool(use_vss_l2)
        self.use_u1_extra_resblock = bool(use_u1_extra_resblock)
        self.use_freeu = bool(use_freeu)
        self.freeu_b1 = float(freeu_b1)
        self.freeu_b2 = float(freeu_b2)
        self.freeu_s1 = float(freeu_s1)
        self.freeu_s2 = float(freeu_s2)
        self.freeu_r_thresh = float(freeu_r_thresh)
        if self.use_physics_cross_attn and not self.use_split_conditioning:
            raise ValueError(
                "use_physics_cross_attn requires use_split_conditioning=True "
                "(Dual-Track: AdaGN + Cross-Attn on the same c_physics)."
            )
        # Decoder stage indices for 128² generation (channel_mult length N):
        #   N=4, P=128 → U3@32 (i=1), U2@64 (i=2), U1@128 (i=3)
        #   N=3, P=128 → U3@32 (i=0), U2@64 (i=1), U1@128 (i=2)
        n_levels = len(channel_mult)
        if n_levels < 3:
            raise ValueError(
                "U-Net requires channel_mult with >= 3 stages, "
                f"got {channel_mult}"
            )
        self.l3_level = n_levels - 1
        self.l2_level = n_levels - 2  # doc L2 @ P/4
        self.u3_level = n_levels - 3  # ~32×32 for P=128
        self.u2_level = n_levels - 2  # ~64×64
        self.u1_level = n_levels - 1  # ~128×128
        # Deepest up-block = doc U3 @ P/8 (VSS already here via u3_vss).
        # Doc U2 @ P/4 is the next up stage (index 1 when N=4).
        self.deep_up_level = 0
        self.doc_u2_up_level = 1
        l3_dim = model_channels * channel_mult[-1]
        l2_dim = model_channels * channel_mult[self.l2_level]
        self.vss_d_state = int(vss_d_state)
        self.vss_expand = int(vss_expand)
        self.self_cond_channels = max(0, int(self_cond_channels))
        # Layout before optional mask: [image (C) | self_cond (C') | coords...]
        if image_channels is None:
            if self.self_cond_channels > 0:
                raise ValueError("image_channels is required when self_cond_channels > 0")
            self.image_channels = int(in_channels)
        else:
            self.image_channels = int(image_channels)
        if self.self_cond_channels > 0:
            expected_min = self.image_channels + self.self_cond_channels
            if int(in_channels) < expected_min:
                raise ValueError(
                    f"in_channels={in_channels} too small for image({self.image_channels})"
                    f"+self_cond({self.self_cond_channels})"
                )
        alpha_mode = str(gated_fusion_alpha_dim).lower().strip()
        if alpha_mode not in ("scalar", "vector"):
            raise ValueError(
                f"gated_fusion_alpha_dim must be 'scalar' or 'vector', got {gated_fusion_alpha_dim!r}"
            )
        self.gated_fusion_alpha_dim = alpha_mode

        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        if self.use_split_conditioning:
            # Dual-Track Track A — AdaGN (Global Mechanics):
            #   e_c = LN_1(MLP_c(t, DI, d_dir))
            #   e_g = LN_2(Linear_g(g))           # g = c_physics [B, 1024]
            #   h_cond = e_c + alpha * e_g        # alpha ≡ 0 at init
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
                # Independent gate: h = e_c + alpha * e_g + alpha_ref * e_ref
                # alpha_ref ≡ 0 at init so a loaded Patch ckpt behaves as before.
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
            # Dual-Track Track B — Cross-Attn (Spatial Geometry):
            #   g [B, 1024] → tokens [B, L, D] (default L=16, D=128)
            if self.use_physics_cross_attn:
                self.physics_norm = nn.LayerNorm(physics_dim)
                self.physics_proj = nn.Linear(
                    physics_dim, self.physics_num_tokens * self.physics_token_dim
                )
            else:
                self.physics_norm = None
                self.physics_proj = None
        elif self.use_physics_adagn:
            # Legacy concat path (kept for C²OT): GRU 1024→128 then cat with DI/dir.
            self.physics_to_cond = nn.Sequential(
                nn.LayerNorm(physics_dim),
                nn.Linear(physics_dim, self.physics_adagn_dim),
            )
            scalar_in_dim = self.physics_adagn_dim + int(latent_dim)
            self.scalar_mlp = nn.Sequential(
                nn.Linear(scalar_in_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
            self.physics_norm = None
            self.physics_proj = None
        else:
            self.cond_mlp = nn.Sequential(
                nn.Linear(latent_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
            self.physics_norm = None
            self.physics_proj = None
        self.joint_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        # Zero-init self-cond input channels so the net starts identical to no-self-cond.
        if self.self_cond_channels > 0:
            with torch.no_grad():
                s = self.image_channels
                e = self.image_channels + self.self_cond_channels
                self.in_conv.weight[:, s:e].zero_()
        if self.use_damage_mask_condition:
            self.in_conv = _expand_conv_in_with_zero_mask_channel(self.in_conv)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = model_channels
        self.skip_channels = []
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(AdaGNResBlock(ch, out_ch, cond_dim, dropout=dropout))
                ch = out_ch
                self.skip_channels.append(ch)
            self.down_blocks.append(blocks)
            if level != len(channel_mult) - 1:
                self.downsamples.append(Downsample(ch))

        mid_ch = ch
        # Decoder walks reversed channel_mult; U3/U2 channel widths:
        #   N=4, mult=(1,2,4,4) → U3@i=1 ch=128, U2@i=2 ch=64
        u3_ch = model_channels * channel_mult[-(self.u3_level + 1)]
        u2_ch = model_channels * channel_mult[-(self.u2_level + 1)]

        self.mid_block1 = AdaGNResBlock(ch, ch, cond_dim, dropout=dropout)
        self.mid_attn = (
            SelfAttention2d(ch, num_heads=self.mid_attn_heads)
            if use_mid_attention
            else nn.Identity()
        )
        if self.use_physics_cross_attn:
            # Inject physics tokens at mid (16×16), U3 (~32×32), U2 (~64×64).
            self.mid_cross_attn = CrossAttention2d(
                mid_ch, self.physics_token_dim, num_heads=self.mid_attn_heads
            )
            self.u3_cross_attn = CrossAttention2d(
                u3_ch, self.physics_token_dim, num_heads=self.mid_attn_heads
            )
            self.u2_cross_attn = CrossAttention2d(
                u2_ch, self.physics_token_dim, num_heads=self.mid_attn_heads
            )
        else:
            self.mid_cross_attn = nn.Identity()
            self.u3_cross_attn = nn.Identity()
            self.u2_cross_attn = nn.Identity()
        self.mid_block2 = AdaGNResBlock(ch, ch, cond_dim, dropout=dropout)

        # VSS map (ARCHITECTURE_UPDATE_V8 naming, P=128 → scale ×2 for P=256):
        #   L3 + doc-U3 @ P/8  → use_vss_l3_u3 (l3_vss / u3_vss)
        #   doc-L2 @ P/4       → use_vss_l2 (l2_vss); encoder spatial understanding
        #   doc-U2 @ P/4       → use_vss_u2 (u2_vss); decoder long-range
        if self.use_vss_l3_u3:
            self.l3_vss = VSSBlock2d(
                dim=l3_dim, d_state=vss_d_state, expand=vss_expand, dropout=dropout
            )
            self.u3_vss = VSSBlock2d(
                dim=l3_dim, d_state=vss_d_state, expand=vss_expand, dropout=dropout
            )
        else:
            self.l3_vss = nn.Identity()
            self.u3_vss = nn.Identity()

        if self.use_vss_l2:
            self.l2_vss = VSSBlock2d(
                dim=l2_dim, d_state=vss_d_state, expand=vss_expand, dropout=dropout
            )
        else:
            self.l2_vss = nn.Identity()

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        doc_u2_ch = None
        for _up_i, (level, mult) in enumerate(reversed(list(enumerate(channel_mult)))):
            out_ch = model_channels * mult
            blocks = nn.ModuleList()
            for _block_j in range(num_res_blocks):
                skip_ch = self.skip_channels.pop()
                blocks.append(AdaGNResBlock(ch + skip_ch, out_ch, cond_dim, dropout=dropout))
                ch = out_ch
            if _up_i == self.doc_u2_up_level:
                doc_u2_ch = int(ch)
            self.up_blocks.append(blocks)
            if level != 0:
                self.upsamples.append(Upsample(ch))

        if self.use_vss_u2:
            if doc_u2_ch is None:
                raise ValueError("use_vss_u2=True but doc U2 up stage was not built")
            self.u2_vss = VSSBlock2d(
                dim=doc_u2_ch, d_state=vss_d_state, expand=vss_expand, dropout=dropout
            )
        else:
            self.u2_vss = nn.Identity()

        # Optional extra AdaGN ResBlock at U1 (128×128) after skip-merge, same hyperparameters.
        if self.use_u1_extra_resblock:
            self.u1_extra_resblock = AdaGNResBlock(ch, ch, cond_dim, dropout=dropout)
        else:
            self.u1_extra_resblock = None

        self.out_norm = nn.GroupNorm(_num_groups(ch), ch, eps=1e-6)
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def _build_cond(self, t, c, gru_feat=None, ref_feat=None):
        t_emb = sinusoidal_timestep_embedding(t, self.model_channels)
        t_emb = self.time_mlp(t_emb)
        if self.use_split_conditioning:
            # Track A (AdaGN): e_c = LayerNorm_1(MLP_c(t, DI, d_dir))
            if gru_feat is None:
                raise ValueError("gru_feat is required when use_split_conditioning=True")
            c_emb = self.scalar_mlp(c)
            e_c = self.e_c_ln(self.joint_mlp(t_emb + c_emb))
            # Physics AdaGN branch: e_g = LayerNorm_2(Linear_g(g))
            e_g = self.e_g_ln(self.physics_linear(gru_feat))
            # Zero-init gated fusion: h_cond = e_c + alpha * e_g
            h = e_c + self.alpha * e_g
            if self.use_undamaged_ref:
                if ref_feat is None:
                    ref_feat = torch.zeros(
                        (c.shape[0], self.undamaged_ref_dim),
                        device=c.device,
                        dtype=c.dtype,
                    )
                e_ref = self.e_ref_ln(self.ref_linear(ref_feat))
                h = h + self.alpha_ref * e_ref
            return h
        if self.use_physics_adagn:
            if gru_feat is None:
                raise ValueError("gru_feat is required when use_physics_adagn=True")
            phys = self.physics_to_cond(gru_feat)
            c_emb = self.scalar_mlp(torch.cat([phys, c], dim=1))
            return self.joint_mlp(t_emb + c_emb)
        c_emb = self.cond_mlp(c)
        return self.joint_mlp(t_emb + c_emb)

    def _build_physics_tokens(self, gru_feat):
        """Map c_physics [B, 1024] → token sequence [B, L, D] for Cross-Attn."""
        if not self.use_physics_cross_attn:
            return None
        if gru_feat is None:
            raise ValueError("gru_feat is required when use_physics_cross_attn=True")
        # [B, 1024] → [B, L*D] → [B, L, D]
        h = self.physics_norm(gru_feat)
        h = self.physics_proj(h)
        return h.view(gru_feat.shape[0], self.physics_num_tokens, self.physics_token_dim)

    def get_gated_fusion_alpha(self):
        """Return current gate magnitude for logging (mean if vector)."""
        if not self.use_split_conditioning or not hasattr(self, "alpha"):
            return None
        return self.alpha.detach().float().mean()

    def get_undamaged_ref_alpha(self):
        if not self.use_undamaged_ref or self.alpha_ref is None:
            return None
        return self.alpha_ref.detach().float().mean()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Drop removed AttentionGate / deep-sup heads from older checkpoints.
        # Strip u2_vss only when this build does not use it (legacy / ablation loads).
        stale_substrings = [
            f"{prefix}attn_gate_u1.",
            f"{prefix}attn_gate_u2.",
            f"{prefix}aux_head_u2.",
            f"{prefix}aux_head_u3.",
        ]
        if not self.use_vss_u2:
            stale_substrings.append(f"{prefix}u2_vss.")
        if not self.use_vss_l2:
            stale_substrings.append(f"{prefix}l2_vss.")
        stale_substrings = tuple(stale_substrings)
        stale = [k for k in list(state_dict.keys()) if any(s in k for s in stale_substrings)]
        for k in stale:
            del state_dict[k]
        # Backward compatibility for channel mismatches (damage-mask / patch-coord configs).
        conv_key = f"{prefix}in_conv.weight"
        if conv_key in state_dict:
            ckpt_weight = state_dict[conv_key]
            expected_in = self.in_conv.in_channels
            if ckpt_weight.ndim == 4 and ckpt_weight.shape[1] != expected_in:
                if ckpt_weight.shape[1] < expected_in:
                    pad = expected_in - ckpt_weight.shape[1]
                    zero_channel = torch.zeros(
                        (ckpt_weight.shape[0], pad, ckpt_weight.shape[2], ckpt_weight.shape[3]),
                        dtype=ckpt_weight.dtype,
                        device=ckpt_weight.device,
                    )
                    state_dict[conv_key] = torch.cat([ckpt_weight, zero_channel], dim=1)
                else:
                    state_dict[conv_key] = ckpt_weight[:, :expected_in, :, :]

        out_w_key = f"{prefix}out_conv.weight"
        if out_w_key in state_dict:
            ckpt_weight = state_dict[out_w_key]
            expected_out = self.out_conv.out_channels
            if ckpt_weight.ndim == 4 and ckpt_weight.shape[0] != expected_out:
                if ckpt_weight.shape[0] < expected_out:
                    pad = expected_out - ckpt_weight.shape[0]
                    zero_out = torch.zeros(
                        (pad, ckpt_weight.shape[1], ckpt_weight.shape[2], ckpt_weight.shape[3]),
                        dtype=ckpt_weight.dtype,
                        device=ckpt_weight.device,
                    )
                    state_dict[out_w_key] = torch.cat([ckpt_weight, zero_out], dim=0)
                else:
                    state_dict[out_w_key] = ckpt_weight[:expected_out]

        out_b_key = f"{prefix}out_conv.bias"
        if out_b_key in state_dict and self.out_conv.bias is not None:
            ckpt_bias = state_dict[out_b_key]
            expected_out = self.out_conv.out_channels
            if ckpt_bias.ndim == 1 and ckpt_bias.shape[0] != expected_out:
                if ckpt_bias.shape[0] < expected_out:
                    pad = expected_out - ckpt_bias.shape[0]
                    zero_bias = torch.zeros(pad, dtype=ckpt_bias.dtype, device=ckpt_bias.device)
                    state_dict[out_b_key] = torch.cat([ckpt_bias, zero_bias], dim=0)
                else:
                    state_dict[out_b_key] = ckpt_bias[:expected_out]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, z_t, t, c, gru_feat=None, damage_mask=None, return_attn_map=False, ref_feat=None):
        """
        Dual-Track OT-CFM U-Net velocity prediction.

        Track A — AdaGN: every ResBlock modulated by gated fusion of scalar + physics.
        Track B — Cross-Attn (optional): pixels query physics tokens at mid / U3 / U2.

        Skip path: plain concat at all decoder stages.
        Returns ``final_out`` [B, out_channels, H, W], or ``(final_out, attn_maps)``
        when ``return_attn_map=True``.
        """
        cond = self._build_cond(t, c, gru_feat=gru_feat, ref_feat=ref_feat)
        physics_tokens = self._build_physics_tokens(gru_feat) if self.use_physics_cross_attn else None
        if self.use_damage_mask_condition:
            if damage_mask is None:
                damage_mask = torch.zeros(
                    (z_t.shape[0], 1, z_t.shape[-2], z_t.shape[-1]),
                    device=z_t.device,
                    dtype=z_t.dtype,
                )
            elif damage_mask.shape[-2:] != z_t.shape[-2:]:
                damage_mask = F.interpolate(
                    damage_mask, size=z_t.shape[-2:], mode="nearest"
                )
            z_t = torch.cat([z_t, damage_mask], dim=1)
        h = self.in_conv(z_t)
        skips = []
        attn_maps = {}

        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, cond)
                if self.use_vss_l2 and i == self.l2_level:
                    h = self.l2_vss(h)  # doc L2 @ P/4
                if self.use_vss_l3_u3 and i == self.l3_level:
                    h = self.l3_vss(h)
                skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        if self.use_physics_cross_attn:
            if return_attn_map and hasattr(self.mid_cross_attn, "store_attn"):
                self.mid_cross_attn.store_attn = True
            h = self.mid_cross_attn(h, physics_tokens)
            if return_attn_map and getattr(self.mid_cross_attn, "last_attn", None) is not None:
                attn_maps["mid"] = self.mid_cross_attn.last_attn
                self.mid_cross_attn.store_attn = False
        h = self.mid_block2(h, cond)

        upsample_idx = 0
        for i, blocks in enumerate(self.up_blocks):
            for block in blocks:
                skip = skips.pop()
                if self.use_freeu:
                    b_l, s_l = freeu_params_for_stage(
                        i, self.freeu_b1, self.freeu_b2, self.freeu_s1, self.freeu_s2
                    )
                    h = freeu_fuse(h, skip, b_l, s_l, r_thresh=self.freeu_r_thresh)
                else:
                    h = torch.cat([h, skip], dim=1)
                h = block(h, cond)
                if self.use_vss_l3_u3 and i == self.deep_up_level:
                    h = self.u3_vss(h)  # doc U3 @ P/8
                if self.use_vss_u2 and i == self.doc_u2_up_level:
                    h = self.u2_vss(h)  # doc U2 @ P/4

            if self.use_physics_cross_attn and physics_tokens is not None:
                if i == self.u3_level:
                    if return_attn_map and hasattr(self.u3_cross_attn, "store_attn"):
                        self.u3_cross_attn.store_attn = True
                    h = self.u3_cross_attn(h, physics_tokens)
                    if return_attn_map and getattr(self.u3_cross_attn, "last_attn", None) is not None:
                        attn_maps["u3"] = self.u3_cross_attn.last_attn
                        self.u3_cross_attn.store_attn = False
                elif i == self.u2_level:
                    if return_attn_map and hasattr(self.u2_cross_attn, "store_attn"):
                        self.u2_cross_attn.store_attn = True
                    h = self.u2_cross_attn(h, physics_tokens)
                    if return_attn_map and getattr(self.u2_cross_attn, "last_attn", None) is not None:
                        attn_maps["u2"] = self.u2_cross_attn.last_attn
                        self.u2_cross_attn.store_attn = False

            # Extra same-channel AdaGN ResBlock only at U1 (128×128).
            if i == self.u1_level and self.u1_extra_resblock is not None:
                h = self.u1_extra_resblock(h, cond)

            if i < len(self.upsamples):
                h = self.upsamples[upsample_idx](h)
                upsample_idx += 1

        h = F.silu(self.out_norm(h))
        final_out = self.out_conv(h)

        if return_attn_map:
            return final_out, attn_maps
        return final_out