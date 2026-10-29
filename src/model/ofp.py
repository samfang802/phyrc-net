"""OFP (Onset-Field Progression): absorbing-state stochastic hazard cellular automaton on a 32x32 grid.

docs/redesign/phase1_ofp.md.  Cell state 0 = intact, 1 = spalled (absorbing).  Between two consecutive drift levels
of one sequence (clock tau = max |drift| reached) an intact cell spalls with probability

    p = 1 - exp(-dtau * (lam_nuc + lam_spr)),
    lam_nuc = exp(b_n + pos(x, y) + g_n * log1p(tau_mid))              nucleation (independent of neighbours)
    lam_spr = exp(b_s + pos(x, y) + g_s * log1p(tau_mid)) * sum_k softplus(w_k) * S_neighbour_k   spread
    g_n, g_s >= 0 (softplus), kernel weights >= 0, so p is monotone in tau and in dtau.

p = sigma(h) with h = log expm1(Lambda): same family as sigma(h_theta) but exactly monotone in dtau and consistent
under splitting a step (1 - p is multiplicative).  Training: discrete-time survival likelihood, BCE on at-risk
(still intact) cells only, teacher forcing with the real state.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

G = 32


# ----------------------------------------------------------------------------------------------- data
def load_sequences(onset_npz="Data/DamagePattern/derived/onset_field.npz",
                   clocks_csv="results/phase0/clock/frames_with_clocks.csv"):
    """Per sequence: levels (nominal drift level, sorted), tau (max drift at the last frame of the level), states
    (J,32,32) = iso32 > 0.5 of that last frame, plus the frame table.  A level with a single frame at tau = 0 is kept."""
    d = np.load(onset_npz, allow_pickle=True)
    fn, seq, iso, cov, md = d["filenames"], d["seq"], d["iso32"], d["cov32"], d["max_drift"]
    C = pd.read_csv(clocks_csv).set_index("filename")
    out = {}
    for s in d["seq_names"]:
        idx = np.where(seq == s)[0]
        tbl = pd.DataFrame({"i": idx, "filename": fn[idx], "md": md[idx]})
        tbl["level"] = C.loc[tbl.filename, "level"].values
        tbl["split"] = C.loc[tbl.filename, "split"].values
        tbl["DI"] = C.loc[tbl.filename, "DI"].values
        tbl = tbl.sort_values("i")
        levels = sorted(tbl.level.unique())
        tau, states, lvl_of = [], [], {}
        for j, L in enumerate(levels):
            sub = tbl[tbl.level == L]
            last = sub.i.max()
            tau.append(float(md[last]))
            states.append(iso[last] > 0.5)
            lvl_of[L] = j
        tbl["j"] = tbl.level.map(lvl_of)
        tbl["gt"] = [iso[i] > 0.5 for i in tbl.i]
        tbl["cov"] = [cov[i] for i in tbl.i]
        out[str(s)] = dict(levels=np.array(levels), tau=np.array(tau, np.float32),
                           states=np.stack(states).astype(np.float32), frames=tbl.reset_index(drop=True),
                           split=tbl.split.iloc[0])
    return out


def sequences_of(seqs, exp_names):
    return {k: v for k, v in seqs.items() if k.split("_")[0] in exp_names}


# ----------------------------------------------------------------------------------------------- model
class OFP(nn.Module):
    """variant: 'a' nucleation only, 'b' nucleation + spread, 'c' = 'b' + coordinate (position prior)."""

    def __init__(self, variant="b", ks=3, hidden=8):
        super().__init__()
        assert variant in "abc" and ks % 2 == 1
        self.variant, self.ks = variant, ks
        self.b_n = nn.Parameter(torch.tensor(-3.0))
        self.g_n_raw = nn.Parameter(torch.tensor(0.0))
        if variant != "a":
            self.b_s = nn.Parameter(torch.tensor(-1.0))
            self.g_s_raw = nn.Parameter(torch.tensor(0.0))
            self.w_raw = nn.Parameter(torch.full((ks * ks,), -1.0))
        if variant == "c":
            self.pos_mlp = nn.Sequential(nn.Linear(5, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.pos_mlp[2].weight)
            nn.init.zeros_(self.pos_mlp[2].bias)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, G), torch.linspace(-1, 1, G), indexing="ij")  # y: top -1, bottom +1
        self.register_buffer("coords", torch.stack([xx, yy, xx * xx, yy * yy, xx * yy], -1))

    def pos(self):
        if self.variant != "c":
            return 0.0
        return self.pos_mlp(self.coords).squeeze(-1)  # (G,G)

    def kernel(self):
        w = F.softplus(self.w_raw).clone()
        w[(self.ks * self.ks) // 2] = 0.0  # no self term (state is 0 for at-risk cells anyway)
        return w.view(1, 1, self.ks, self.ks)

    def log_rates(self, tau_mid, S):
        """S: (B,G,G) previous state.  Returns lam (B,G,G)."""
        lt = math.log1p(float(tau_mid)) if not torch.is_tensor(tau_mid) else torch.log1p(tau_mid)
        pos = self.pos()
        lam = torch.exp(self.b_n + pos + F.softplus(self.g_n_raw) * lt)
        lam = lam.expand(S.shape[0], G, G) if torch.is_tensor(lam) and lam.dim() == 2 else lam * torch.ones_like(S)
        if self.variant != "a":
            nb = F.conv2d(S.unsqueeze(1), self.kernel(), padding=self.ks // 2).squeeze(1)
            lam_s = torch.exp(self.b_s + pos + F.softplus(self.g_s_raw) * lt) * nb
            lam = lam + lam_s
        return lam

    def prob(self, S, tau_prev, tau_cur):
        dt = max(float(tau_cur - tau_prev), 0.0)
        lam = self.log_rates(0.5 * (tau_prev + tau_cur), S)
        return -torch.expm1(-dt * lam), dt * lam  # p, Lambda

    # -- training -----------------------------------------------------------------------------------
    def nll(self, seqs, reduce=True):
        """Teacher-forced at-risk BCE, summed over transitions.  Returns (sum nll, n at-risk, n events)."""
        tot, nrisk, nev = 0.0, 0, 0
        for s in seqs.values():
            tau = np.concatenate([[0.0], s["tau"]]).astype(np.float32)
            St = torch.from_numpy(np.concatenate([np.zeros((1, G, G), np.float32), s["states"]]))
            # states are monotone by construction (iso); enforce it for safety
            St = torch.cummax(St, 0).values
            for j in range(1, len(tau)):
                if tau[j] <= tau[j - 1]:
                    continue
                prev, cur = St[j - 1:j], St[j]
                risk = (prev[0] == 0)
                p, Lam = self.prob(prev, tau[j - 1], tau[j])
                p, Lam = p[0], Lam[0]
                y = cur
                ll = y * torch.log((-torch.expm1(-Lam)).clamp_min(1e-12)) + (1 - y) * (-Lam)
                tot = tot - (ll * risk).sum()
                nrisk += int(risk.sum())
                nev += int((risk & (cur > 0.5)).sum())
        return tot, nrisk, nev

    # -- sampling -----------------------------------------------------------------------------------
    @torch.no_grad()
    def rollout(self, tau, K=8, gen=None):
        """tau: (J,) clock per level.  Returns (K,J,G,G) float states, same k = one onset field."""
        S = torch.zeros(K, G, G)
        out = []
        tp = 0.0
        for t in tau:
            if t > tp:
                p, _ = self.prob(S, tp, float(t))
                new = (torch.rand(p.shape, generator=gen) < p).float()
                S = torch.maximum(S, new * (S == 0))
            tp = max(tp, float(t))
            out.append(S.clone())
        return torch.stack(out, 1)

    @torch.no_grad()
    def mean_prob_rollout(self, tau, K=64, gen=None):
        return self.rollout(tau, K, gen).mean(0)


def fit(variant, seqs, ks=3, hidden=8, wd=1e-3, lr=0.05, steps=400, seed=0, verbose=False):
    torch.manual_seed(seed)
    m = OFP(variant, ks, hidden)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    nr = None
    for it in range(steps):
        opt.zero_grad()
        nll, nr, ne = m.nll(seqs)
        reg = sum((p ** 2).sum() for n, p in m.named_parameters() if n.startswith("pos_mlp"))
        loss = nll / nr + wd * reg
        loss.backward()
        opt.step()
        if verbose and it % 100 == 0:
            print(it, float((nll / nr).detach()))
    return m


# ----------------------------------------------------------------------------------------------- metrics
def layout_for_frames(roll, frames):
    """roll (K,J,G,G) -> (N,K,G,G): frame layout = rollout state at the frame's level."""
    j = frames.j.values
    return np.ascontiguousarray(roll.numpy()[:, j].transpose(1, 0, 2, 3)).astype(np.float32)


def comp_stats(binmap):
    from scipy import ndimage
    lab, n = ndimage.label(binmap > 0.5)
    if n == 0:
        return 0, float("nan")
    sizes = np.bincount(lab.ravel())[1:]
    return n, float(sizes.max() / sizes.sum())
