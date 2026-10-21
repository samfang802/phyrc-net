"""Frozen DINOv3 image embedding used as an oracle stand-in for GRU g.

The GT frame is embedded and fed through the same physics path (Linear → gated AdaGN)
that normally takes the GRU hidden state. The gap to the GRU run bounds how much a
better upstream condition could help.

The encoder is not an nn.Module attribute of the CFM model, so its weights stay out
of the optimizer and the checkpoint. It is reloaded from the HF cache on construction.
"""

import torch
import torch.nn.functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoOracleEncoder:
    """pos_blur removes where things are from the GT frame before DINOv3, keeping local appearance.

    shift:   random translation (reflect padded) plus random h/v flips. Train only;
             eval embeds the frame unchanged, so the net has to ignore position.
    shuffle: cut the frame into grid x grid tiles and permute them. Random permutation in
             train, one fixed permutation in eval. Destroys layout, keeps crack/spall texture.
    """

    def __init__(
        self, model_id, pool="cls_mean", size=224, pos_blur="none", shift_frac=0.3, shuffle_grid=4
    ):
        from transformers import AutoModel

        if pool not in ("cls", "mean", "cls_mean"):
            raise ValueError(f"oracle_dino_pool must be cls, mean or cls_mean, got {pool}")
        if pos_blur not in ("none", "shift", "shuffle"):
            raise ValueError(f"oracle_dino_pos_blur must be none, shift or shuffle, got {pos_blur}")
        if pos_blur == "shuffle" and int(size) % int(shuffle_grid) != 0:
            raise ValueError(f"size {size} must be divisible by shuffle_grid {shuffle_grid}")
        self.model_id = str(model_id)
        self.pool = pool
        self.size = int(size)
        self.pos_blur = pos_blur
        self.shift_frac = float(shift_frac)
        self.shuffle_grid = int(shuffle_grid)
        self.model = AutoModel.from_pretrained(self.model_id).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        hidden = int(self.model.config.hidden_size)
        self.num_prefix = 1 + int(getattr(self.model.config, "num_register_tokens", 0))
        self.feature_dim = hidden * (2 if pool == "cls_mean" else 1)

    def _shift(self, x):
        b = x.shape[0]
        theta = x.new_zeros(b, 2, 3)
        theta[:, 0, 0] = torch.where(torch.rand(b, device=x.device) < 0.5, -1.0, 1.0)
        theta[:, 1, 1] = torch.where(torch.rand(b, device=x.device) < 0.5, -1.0, 1.0)
        theta[:, :, 2] = (torch.rand(b, 2, device=x.device) * 2 - 1) * 2 * self.shift_frac
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    def _shuffle(self, x, training):
        b, c, hh, ww = x.shape
        g = self.shuffle_grid
        h, w, n = hh // g, ww // g, g * g
        tiles = x.view(b, c, g, h, g, w).permute(0, 1, 2, 4, 3, 5).reshape(b, c, n, h, w)
        if training:
            idx = torch.argsort(torch.rand(b, n, device=x.device), dim=1)
        else:
            gen = torch.Generator().manual_seed(0)
            idx = torch.randperm(n, generator=gen).to(x.device).expand(b, n)
        tiles = tiles.gather(2, idx.view(b, 1, n, 1, 1).expand(b, c, n, h, w))
        return tiles.view(b, c, g, g, h, w).permute(0, 1, 2, 4, 3, 5).reshape(b, c, hh, ww)

    @torch.no_grad()
    def embed(self, images, training=False):
        """images: [B, C, H, W] in [-1, 1], C = 1 or 3. Returns [B, feature_dim] fp32."""
        device = images.device
        if next(self.model.parameters()).device != device:
            self.model.to(device)
        x = images.float() * 0.5 + 0.5
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(self.size, self.size), mode="bicubic", align_corners=False)
        x = x.clamp(0.0, 1.0)
        if self.pos_blur == "shift" and training:
            x = self._shift(x)
        elif self.pos_blur == "shuffle":
            x = self._shuffle(x, training)
        mean = x.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        tokens = self.model(pixel_values=(x - mean) / std).last_hidden_state
        cls = tokens[:, 0]
        if self.pool == "cls":
            feat = cls
        elif self.pool == "mean":
            feat = tokens[:, self.num_prefix :].mean(dim=1)
        else:
            # Skip CLS and register tokens for the patch mean.
            feat = torch.cat([cls, tokens[:, self.num_prefix :].mean(dim=1)], dim=-1)
        # DINO features carry a few very large dims; put them on the GRU g scale.
        return F.layer_norm(feat, feat.shape[-1:])
