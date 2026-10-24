"""Damage-state encoder (docs/DAMAGE_STATE_ENCODER.md).

Static specimen descriptors -> s. The simulated loop (+ s) -> causal GRU -> z, the state so far.
``input_mode="tokens"``: one input per half-cycle, z_k at the k-th load peak.
``input_mode="points"``: the GRU extractor's own per-point input (17 column-info features, drift,
simulated force), z_t at every point, an extra per-point experimental-force head; the damage heads
read z at the peaks 30k. ``seq_info="dimless"`` swaps the 17 column-info features for the 7
dimensionless descriptors; ``force_target="abs"`` makes the per-point force target |F| so z need
not carry the loading direction.
Heads read (z, s) and predict damage-state targets of the experimental loop. Training mirrors loops
at random (all signed features flip together) so z does not encode the loading direction.
"""

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.damage_state import (
    FAILURE_TYPES, N_INFO, SEG, SIGNED_TOKEN_FEATURES, STATIC_FEATURES, TARGETS, TOKEN_FEATURES,
)


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))


class DamageStateEncoder(pl.LightningModule):
    def __init__(self, static_dim=64, token_dim=128, hidden=256, layers=2, dropout=0.1, lr=1e-3, weight_decay=1e-4,
                 mirror_prob=0.5, fail_weight=0.5, input_mode="tokens", static_info=False, force_weight=1.0,
                 seq_info="raw", force_target="signed", **kwargs):
        super().__init__()
        self.save_hyperparameters()
        n_static = len(STATIC_FEATURES) + (N_INFO if static_info else 0)
        self.static_net = _mlp(n_static + len(FAILURE_TYPES), static_dim, static_dim)
        self.points = input_mode == "points"
        self.seq_dim = (N_INFO if seq_info == "raw" else len(STATIC_FEATURES)) + 2
        self.token_in = nn.Linear(self.seq_dim if self.points else len(TOKEN_FEATURES), token_dim)
        if self.points:
            self.force_head = _mlp(hidden + static_dim, 64, 1)
        self.gru = nn.GRU(token_dim + static_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.reg_head = _mlp(hidden + static_dim, 128, len(TARGETS))
        self.fail_head = _mlp(hidden + static_dim, 64, 1)
        self.register_buffer("signed_idx", torch.tensor([TOKEN_FEATURES.index(k) for k in SIGNED_TOKEN_FEATURES]))

    def encode_static(self, static, failure_type):
        if not self.hparams.static_info:
            static = static[:, : len(STATIC_FEATURES)]
        onehot = F.one_hot(failure_type, len(FAILURE_TYPES)).to(static.dtype)
        return self.static_net(torch.cat([static, onehot], dim=1))

    def forward(self, tokens, static, failure_type, lengths=None):
        """Returns z [B, T, hidden] (causal) and s [B, static_dim]. In points mode ``tokens`` is the
        per-point input and z has one state per point."""
        s = self.encode_static(static, failure_type)
        x = torch.cat([self.token_in(tokens), s[:, None].expand(-1, tokens.shape[1], -1)], dim=2)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            z, _ = self.gru(x)
            z, _ = nn.utils.rnn.pad_packed_sequence(z, batch_first=True, total_length=tokens.shape[1])
        else:
            z, _ = self.gru(x)
        return z, s

    def _mirror(self, x, idx, flip):
        sign = torch.ones_like(x[:, :1, :])
        sign[flip.nonzero(as_tuple=True)[0][:, None], 0, idx[None]] = -1.0
        return x * sign

    def peak_states(self, batch):
        """(z at the load peaks [B, T, hidden], s, z at every point or None)."""
        z_pts = None
        if self.points:
            z_pts, s = self(batch["seq"], batch["static"], batch["failure_type"], batch["point_lengths"])
            idx = SEG * (torch.arange(batch["tokens"].shape[1], device=z_pts.device) + 1)
            z = z_pts[:, idx.clamp(max=z_pts.shape[1] - 1)]
        else:
            z, s = self(batch["tokens"], batch["static"], batch["failure_type"], batch["lengths"])
        return z, s, z_pts

    def _step(self, batch, stage):
        force_exp = batch.get("force_exp")
        if stage == "train" and self.hparams.mirror_prob > 0:
            flip = torch.rand(batch["tokens"].shape[0], device=batch["tokens"].device) < self.hparams.mirror_prob
            batch = dict(batch)
            batch["tokens"] = self._mirror(batch["tokens"], self.signed_idx, flip)
            if self.points:
                seq_idx = torch.tensor([self.seq_dim - 2, self.seq_dim - 1], device=flip.device)
                batch["seq"] = self._mirror(batch["seq"], seq_idx, flip)
                force_exp = torch.where(flip[:, None], -force_exp, force_exp)
        if self.points and self.hparams.force_target == "abs":
            force_exp = force_exp.abs()
        z, s, z_pts = self.peak_states(batch)
        tokens = batch["tokens"]
        h = torch.cat([z, s[:, None].expand(-1, z.shape[1], -1)], dim=2)
        t = torch.arange(z.shape[1], device=z.device)[None]
        in_len = (t < batch["lengths"].to(z.device)[:, None]).float()
        m = in_len * batch["valid"]
        reg = self.reg_head(h)
        per = ((reg - batch["targets"]) ** 2) * m[..., None]
        reg_loss = per.sum() / (m.sum() * reg.shape[-1]).clamp(min=1.0)
        fail = self.fail_head(h).squeeze(-1)
        fail_loss = (F.binary_cross_entropy_with_logits(fail, batch["failed"], reduction="none") * in_len).sum() / in_len.sum()
        loss = reg_loss + self.hparams.fail_weight * fail_loss
        bs = tokens.shape[0]
        if self.points:
            hp = torch.cat([z_pts, s[:, None].expand(-1, z_pts.shape[1], -1)], dim=2)
            tp = torch.arange(z_pts.shape[1], device=z.device)[None]
            mp = (tp < batch["point_lengths"].to(z.device)[:, None]).float()
            force_loss = (((self.force_head(hp).squeeze(-1) - force_exp) ** 2) * mp).sum() / mp.sum()
            loss = loss + self.hparams.force_weight * force_loss
            self.log(f"{stage}/force_loss", force_loss, batch_size=bs, on_epoch=True, on_step=False)
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=bs, on_epoch=True, on_step=False)
        self.log(f"{stage}/reg_loss", reg_loss, batch_size=bs, on_epoch=True, on_step=False)
        self.log(f"{stage}/fail_loss", fail_loss, batch_size=bs, on_epoch=True, on_step=False)
        for j, name in enumerate(TARGETS):
            mse = per[..., j].sum() / m.sum().clamp(min=1.0)
            self.log(f"{stage}/mse_{name}", mse, batch_size=bs, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
