#!/usr/bin/env python3
"""Figures for docs/redesign/phase1_ofp.md: learned position preference and sample layouts.

  python scripts/eval/ofp_figures.py --dir results/phase1_ofp
"""
import argparse, os, sys
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.model import ofp as O

TRAIN = ("R315", "R330", "R615", "R1015")


def load(d, v):
    ck = torch.load(f"{d}/models/ofp_{v}.pt")
    c = ck["cfg"]
    m = O.OFP(c["variant"], c["ks"], c["hidden"]); m.load_state_dict(ck["state"]); return m.eval()


def mean_onset(states, tau):
    """Mean over sequences of the onset drift (nan -> 12 = never) per cell: (G,G)."""
    T = np.where(states.max(0) > 0, tau[np.argmax(states > 0.5, 0)], np.nan)
    return T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results/phase1_ofp"); a = ap.parse_args()
    seqs = O.load_sequences()
    tr = O.sequences_of(seqs, TRAIN); te = O.sequences_of(seqs, ("CTR1",)); va = O.sequences_of(seqs, ("R307",))
    ms = {"a": load(a.dir, "a"), "b": load(a.dir, "b"), "c": load(a.dir, "c_iou")}  # c = R307-IoU-selected
    # (1) row profile of spalled fraction at fixed drift levels (rows = height, top -> bottom)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for i, (title, sq) in enumerate((("train (4 columns)", tr), ("val R307", va), ("test CTR1", te))):
        for lev in (3.0, 4.0, 6.0):
            prof = []
            for s in sq.values():
                j = np.where(s["levels"] == lev)[0]
                if len(j): prof.append(s["states"][j[0]].mean(1))
            if prof: ax[i].plot(np.mean(prof, 0), np.arange(32), color="k", ls={3.0: ":", 4.0: "--", 6.0: "-"}[lev],
                                label=f"GT {lev:g}%")
        for v, c in zip("abc", ("tab:blue", "tab:orange", "tab:green")):
            for lev, ls in ((3.0, ":"), (4.0, "--"), (6.0, "-")):
                prof = []
                for n, s in sq.items():
                    j = np.where(s["levels"] == lev)[0]
                    if not len(j): continue
                    g = torch.Generator().manual_seed(0)
                    r = ms[v].rollout(torch.from_numpy(s["tau"]), 64, g)[:, j[0]].mean(0).mean(1).numpy()
                    prof.append(r)
                if prof: ax[i].plot(np.mean(prof, 0), np.arange(32), color=c, ls=ls, alpha=.8, label=f"{v} {lev:g}%" if i == 0 else None)
        ax[i].set_title(title); ax[i].invert_yaxis(); ax[i].set_xlabel("spalled fraction in row")
    ax[0].set_ylabel("row (0 = top, 31 = bottom of photo)"); ax[0].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(f"{a.dir}/fig_row_profile.png", dpi=130); plt.close(fig)
    # (2) learned position term and mean onset field
    m = ms["c"]
    with torch.no_grad():
        pos = m.pos().numpy()
        lam_n = np.exp(float(m.b_n) + pos + torch.nn.functional.softplus(m.g_n_raw).item() * np.log1p(3.0))
        lam_s = np.exp(float(m.b_s) + pos + torch.nn.functional.softplus(m.g_s_raw).item() * np.log1p(3.0))
    T_gt = np.nanmean([mean_onset(s["states"], s["tau"]) for s in tr.values()], 0)
    Ts = []
    for s in tr.values():
        r = m.rollout(torch.from_numpy(s["tau"]), 32, torch.Generator().manual_seed(0)).numpy()
        Ts += [np.where(x.max(0) > 0, s["tau"][np.argmax(x > 0.5, 0)], np.nan) for x in r]
    T_m = np.nanmean(Ts, 0)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for x, t, kw in ((ax[0], "pos(x,y) of variant c (log-rate offset)", dict(data=pos, cmap="coolwarm")),
                     (ax[1], "nucleation rate at tau=3% (c)", dict(data=lam_n, cmap="viridis")),
                     (ax[2], "mean onset drift, GT (train, %)", dict(data=T_gt, cmap="magma_r", vmin=2, vmax=8)),
                     (ax[3], "mean onset drift, variant c rollouts (%)", dict(data=T_m, cmap="magma_r", vmin=2, vmax=8))):
        im = x.imshow(kw.pop("data"), **kw); x.set_title(t, fontsize=9); plt.colorbar(im, ax=x, fraction=.046)
    fig.tight_layout(); fig.savefig(f"{a.dir}/fig_position_prior.png", dpi=130); plt.close(fig)
    print("pos range", pos.min(), pos.max(), "top-half mean", pos[:16].mean(), "bottom-half mean", pos[16:].mean())
    # (3) CTR1 samples: GT vs a/b/c at 4 and 6 %
    fig, ax = plt.subplots(4, 7, figsize=(14, 8))
    s = te["CTR1_NE"]
    for r, v in enumerate(("gt", "a", "b", "c")):
        for cidx, lev in enumerate((3.0, 4.0, 6.0, 8.0, 6.0, 6.0, 6.0)):
            j = np.where(s["levels"] == lev)[0][0]
            if v == "gt": img = s["states"][j]
            else:
                roll = ms[v].rollout(torch.from_numpy(s["tau"]), 8, torch.Generator().manual_seed(1)); img = roll[cidx if cidx < 4 else cidx - 3, j].numpy()
            ax[r, cidx].imshow(img, cmap="gray_r", vmin=0, vmax=1); ax[r, cidx].set_xticks([]); ax[r, cidx].set_yticks([])
            if r == 0: ax[r, cidx].set_title(f"{lev:g}%", fontsize=8)
        ax[r, 0].set_ylabel(v)
    fig.suptitle("CTR1_NE: GT (top) and rollouts of a / b / c (columns 6-7: other samples at 6%)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{a.dir}/fig_ctr1_samples.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
