import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32 if channels >= 32 else 1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        h_in = x
        x = self.norm(x)
        q, k, v = torch.chunk(self.qkv(x), 3, dim=1)
        q = q.reshape(b, c, h * w).transpose(1, 2)  # [B, HW, C]
        k = k.reshape(b, c, h * w)                   # [B, C, HW]
        attn = torch.softmax(torch.bmm(q, k) * (c ** -0.5), dim=-1)
        v = v.reshape(b, c, h * w).transpose(1, 2)   # [B, HW, C]
        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        out = self.proj(out)
        return h_in + out


class AdaGNResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        groups_in = 32 if in_channels >= 32 else 1
        groups_out = 32 if out_channels >= 32 else 1
        self.norm1 = nn.GroupNorm(groups_in, in_channels, eps=1e-6)
        self.norm2 = nn.GroupNorm(groups_out, out_channels, eps=1e-6)

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
    ):
        super().__init__()
        self.model_channels = model_channels
        self.cond_dim = cond_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(latent_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.joint_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

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

        self.mid_block1 = AdaGNResBlock(ch, ch, cond_dim, dropout=dropout)
        self.mid_attn = SelfAttention2d(ch) if use_mid_attention else nn.Identity()
        self.mid_block2 = AdaGNResBlock(ch, ch, cond_dim, dropout=dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = self.skip_channels.pop()
                blocks.append(AdaGNResBlock(ch + skip_ch, out_ch, cond_dim, dropout=dropout))
                ch = out_ch
            self.up_blocks.append(blocks)
            if level != 0:
                self.upsamples.append(Upsample(ch))

        self.out_norm = nn.GroupNorm(32 if ch >= 32 else 1, ch, eps=1e-6)
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def _build_cond(self, t, c):
        t_emb = sinusoidal_timestep_embedding(t, self.model_channels)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.cond_mlp(c)
        return self.joint_mlp(t_emb + c_emb)

    def forward(self, z_t, t, c):
        cond = self._build_cond(t, c)
        h = self.in_conv(z_t)
        skips = []

        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, cond)
                skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)

        upsample_idx = 0
        for i, blocks in enumerate(self.up_blocks):
            for block in blocks:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, cond)
            if i < len(self.upsamples):
                h = self.upsamples[upsample_idx](h)
                upsample_idx += 1

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
