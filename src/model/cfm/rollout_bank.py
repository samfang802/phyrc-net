"""Rollout bank (rfmv2b): the model's own multi-hop rollouts of the train split, pasted into S during training.

The window loss never shows the refiner the S it meets at sampling: every resampling hop restarts from the real
frame at t_s, while a sampled frame is drawn from pure noise h times in a row (fog, tone drift, lost and false
cracks; docs/RFMV2_ARCHITECTURE.md §10). The bank stores, for every train row j, the degradation of the model's
own h-hop rollout against the real frame, ``delta = S_self(j) - X(j)``, and its structure error
``memory_error(S_self(j), X(j))``. The window loss adds ``delta`` to S inside random feathered cells (on top of
the photometric augmentation, hence a delta and not the frame) and injects the error into the memory, so
- lost cracks in a cell are positives: the memory and the target hold them, S does not;
- fog / false cracks in a cell and spurious memory entries are negatives: the target lacks them.

Rollouts (as scripts/eval/rollout_vs_resample_probe.py): chains of ``chain`` generated frames start from the real
frame at every ``stride``-th walk position (random offset per build), memory written from the real history up to
the chain start and then from the generated frames, S confidence ``refiner_state_conf_value`` and hops q - 1
as at sampling. Every row is reached by ``chain / stride`` chains with different hops (slots). The bank keeps
the last ``keep`` builds (a replay of the run's own earlier checkpoints) and is rebuilt every ``every`` steps.
"""

from __future__ import annotations

import json
import os
import time

import torch
from torch.utils.data import DataLoader, Subset

from src.model.cfm.refiner_memory import memory_error, memory_feature

EPS = 0.03


def load_rows(cfm, ds, device):
    """Per dataset row (label order, no augmentation): target-space frame, c, g, condition mask."""
    xs, cs, gs, ms = [], [], [], []
    rows = Subset(ds, range(len(ds.label)))  # one item per label row (the train split repeats virtually)
    for batch in DataLoader(rows, batch_size=32, shuffle=False, num_workers=0):
        real, hs, dr, _exp, _ts, di, info, _valid, cmask, _ar = cfm._unpack_generation_batch(batch)
        c, g = cfm._unpack_conditions(info.to(device), hs.to(device), dr.to(device), di.to(device))
        xs.append(cfm._encode_to_target_space(real.to(device)).float())
        cs.append(c)
        gs.append(g)
        ms.append(None if cmask is None else cfm._align_spatial_to_target(cmask.to(device)).float())
    cat = lambda v: None if v[0] is None else torch.cat(v)
    return torch.cat(xs), torch.cat(cs), cat(gs), cat(ms)


def walks_of(ds):
    out = []
    for _, rows in ds.label.groupby(["exp", "direction"], sort=True):
        out.append([int(r) for r in rows.sort_values("timestep", kind="stable").index.to_numpy()])
    return out


class RolloutBank:
    def __init__(self, n_rows, chain=8, stride=4, keep=3):
        if chain % stride:
            raise ValueError("rollout bank: chain must be a multiple of stride")
        self.n_rows, self.chain, self.stride, self.keep = int(n_rows), int(chain), int(stride), int(keep)
        self.slots = self.chain // self.stride
        self.builds = []  # dicts: delta [N, S, C, H, W] f16, err [N, S, 1, H, W] f16, hop [N, S] int (0 = none)
        self._data = None

    def ready(self):
        return bool(self.builds)

    # ------------------------------------------------------------------ build
    @torch.no_grad()
    def build(self, cfm, ds, step, seed, batch=32, log_dir=None):
        device = next(cfm.parameters()).device
        was_training = cfm.training
        cfm.eval()
        t0 = time.time()
        try:
            if self._data is None:
                X, C, G, M = load_rows(cfm, ds, device)
                if X.shape[0] != self.n_rows:
                    raise ValueError(f"rollout bank: {X.shape[0]} rows, expected {self.n_rows}")
                self._data = (X, C, G, M, walks_of(ds))
            out = self._rollouts(cfm, step, seed, batch)
        finally:
            cfm.train(was_training)
        self.builds = (self.builds + [out])[-self.keep:]
        stats = self._stats(out)
        stats.update(step=int(step), seconds=round(time.time() - t0, 1), builds_kept=len(self.builds))
        if torch.cuda.is_available():
            stats.update(peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                         peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2))
        print("[rollout_bank] " + json.dumps(stats), flush=True)
        if log_dir:
            with open(os.path.join(log_dir, "rollout_bank_log.jsonl"), "a") as fh:
                fh.write(json.dumps(stats) + "\n")
        return stats

    def _rollouts(self, cfm, step, seed, batch):
        X, C, G, M, walks = self._data
        device = X.device
        sig, tau, win = float(cfm.refiner_memory_sigma), float(cfm.refiner_memory_tau), int(cfm.refiner_memory_window)
        conf = float(getattr(cfm, "refiner_state_conf_value", 1.0))
        use_dg = bool(getattr(cfm, "use_delta_g", False)) and G is not None
        F_all = memory_feature(X, sig, tau).float()
        gen = torch.Generator().manual_seed(int(seed))
        offset = int(torch.randint(0, self.stride, (1,), generator=gen))
        # chains: (walk, start position); memory after the real history up to the start, per walk prefix
        chains, mem_after = [], {}
        for w, walk in enumerate(walks):
            value = torch.zeros_like(F_all[:1])
            for p in range(len(walk)):
                lo = max(0, p - win + 1)
                value = torch.minimum(value, F_all[walk[lo:p + 1]].median(dim=0, keepdim=True).values)
                mem_after[(w, p)] = value
            for s in range(offset, len(walk) - 1, self.stride):
                chains.append((w, s))
        N, S_ = self.n_rows, self.slots
        C_, H, W_ = X.shape[1:]
        delta = torch.zeros(N, S_, C_, H, W_, dtype=torch.float16, device=device)
        err = torch.zeros(N, S_, 1, H, W_, dtype=torch.float16, device=device)
        hop = torch.zeros(N, S_, dtype=torch.long, device=device)
        nan = torch.full_like(F_all[:1], float("nan"))
        for b0 in range(0, len(chains), batch):
            grp = chains[b0:b0 + batch]
            B = len(grp)
            start_rows = [walks[w][s] for w, s in grp]
            state = X[start_rows].clone()
            value = torch.cat([mem_after[(w, s)] for w, s in grp])
            recent = []
            for k in range(win):  # NaN-padded window of the real history before the chain start
                recent.append(torch.cat([F_all[walks[w][s - win + 1 + k]:walks[w][s - win + 1 + k] + 1]
                                         if s - win + 1 + k >= 0 else nan for w, s in grp]))
            seeds = [int(seed) * 1000003 + b0 + i for i in range(B)]
            with cfm.refiner_noise_streams(seeds, [1] * B, device):
                shared = cfm._refiner_start_noise(None, B, device) if cfm.refiner_shared_lowfreq_noise else None
                for q in range(1, self.chain + 1):
                    pos = [min(s + q, len(walks[w]) - 1) for w, s in grp]
                    ok = torch.tensor([s + q <= len(walks[w]) - 1 for w, s in grp], device=device)
                    if not bool(ok.any()):
                        break
                    i = [walks[w][p] for (w, _s), p in zip(grp, pos)]
                    j = [walks[w][p - 1] for (w, _s), p in zip(grp, pos)]
                    z0 = cfm._refiner_start_noise(shared, B, device)
                    mask = None
                    if getattr(cfm, "use_damage_mask_condition", False):
                        mask = M[i] if M is not None else torch.ones(B, 1, H, W_, device=device)
                    hint = torch.cat([state, value.to(state.dtype)], dim=1)
                    hint = cfm._refiner_conf_hint(hint, conf, hops=q - 1)
                    out = cfm.sample_with_ode(C[i], z0=z0, gru_feat=None if G is None else G[i], damage_mask=mask,
                                              lr_cond=hint, gru_delta=(G[i] - G[j]) if use_dg else None).float()
                    state = torch.where(ok.view(B, 1, 1, 1), out, state)
                    f = memory_feature(state, sig, tau).float()
                    recent = recent[1:] + [f]
                    value = torch.minimum(value, torch.nanmedian(torch.stack(recent), dim=0).values)
                    slot = (q - 1) // self.stride
                    for b in torch.nonzero(ok).flatten().tolist():
                        r = i[b]
                        delta[r, slot] = (state[b] - X[r]).half()
                        err[r, slot] = memory_error(state[b:b + 1], X[r:r + 1], sig, tau)[0].half()
                        hop[r, slot] = q
        return {"delta": delta, "err": err, "hop": hop, "step": int(step)}

    @staticmethod
    def _stats(out):
        """Gray levels (target space [-1, 1] -> 127.5 per unit); crack shares on the memory feature."""
        hop, d, e = out["hop"], out["delta"].float(), out["err"].float()
        res = {"filled": float((hop > 0).float().mean())}
        for name, sel in (("h1-4", (hop >= 1) & (hop <= 4)), ("h5-8", hop >= 5)):
            if not bool(sel.any()):
                continue
            dd, ee = d[sel], e[sel]
            res[f"{name}_d_mean"] = round(float(dd.mean()) * 127.5, 2)
            res[f"{name}_abs_delta"] = round(float(dd.abs().mean()) * 127.5, 2)
            res[f"{name}_miss_px"] = round(float((ee > EPS).float().mean()), 4)
            res[f"{name}_spur_px"] = round(float((ee < -EPS).float().mean()), 4)
        return res

    # ------------------------------------------------------------------ sample
    def sample(self, rows, device):
        """rows [B] global train rows (-1 = none) -> delta [B, C, H, W], err [B, 1, H, W], ok [B]: a random build
        and a random filled slot per row."""
        B = rows.shape[0]
        rows = rows.to(device).long()
        bi = torch.randint(0, len(self.builds), (B,), device=device)
        H0 = self.builds[0]["delta"]
        delta = torch.zeros(B, *H0.shape[2:], dtype=torch.float32, device=device)
        err = torch.zeros(B, *self.builds[0]["err"].shape[2:], dtype=torch.float32, device=device)
        ok = torch.zeros(B, dtype=torch.bool, device=device)
        for k, bd in enumerate(self.builds):
            sel = torch.nonzero((bi == k) & (rows >= 0)).flatten()
            if sel.numel() == 0:
                continue
            r = rows[sel]
            filled = bd["hop"][r] > 0  # [n, S]
            score = torch.where(filled, torch.rand(filled.shape, device=device), torch.full(filled.shape, -1.0,
                                                                                           device=device))
            slot = score.argmax(dim=1)
            has = filled.any(dim=1)
            delta[sel] = bd["delta"][r, slot].float()
            err[sel] = bd["err"][r, slot].float()
            ok[sel] = has
        delta = delta * ok.view(B, 1, 1, 1)
        err = err * ok.view(B, 1, 1, 1)
        return delta, err, ok


def feathered_cells(B, H, W, grid, rate_lo, rate_hi, sigma, device):
    """[B, 1, H, W] in [0, 1]: random grid cells (share U[rate_lo, rate_hi] per row), Gaussian-feathered."""
    from src.model.cfm.refiner_memory import _gauss_1d
    import torch.nn.functional as F

    rate = rate_lo + (rate_hi - rate_lo) * torch.rand(B, 1, 1, 1, device=device)
    m = F.interpolate((torch.rand(B, 1, grid, grid, device=device) < rate).float(), size=(H, W), mode="nearest")
    if sigma <= 0:
        return m
    k, r = _gauss_1d(sigma, device, torch.float32)
    m = F.conv2d(F.pad(m, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    m = F.conv2d(F.pad(m, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return m.clamp(0.0, 1.0)
