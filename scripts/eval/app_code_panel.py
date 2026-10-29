#!/usr/bin/env python3
"""Visual panel for the appearance code: GT | v11 | app null / mean / random / seq0..7, high-DI frames.
python scripts/eval/app_code_panel.py --root results/eval_suite/v11_app --split val --out panel_val.png"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--split", required=True)
p.add_argument("--out", required=True)
p.add_argument("--models", nargs="*", default=["v11", "app_null", "app_mean", "app_random"] + [f"app_seq{i}" for i in range(8)])
p.add_argument("--rows", type=int, default=5)
a = p.parse_args()
sd = os.path.join(a.root, a.split, "samples")
z = {m: np.load(os.path.join(sd, f"{m}.npz")) for m in a.models}
first = z[a.models[0]]
DI, real, fn = first["DI"], first["real"], first["filename"]
hi = np.where(DI >= np.quantile(DI, 2 / 3))[0]
sel = hi[np.linspace(0, len(hi) - 1, a.rows).astype(int)]
fig, ax = plt.subplots(len(sel), 1 + len(a.models), figsize=(1.6 * (1 + len(a.models)), 1.65 * len(sel)))
for r, i in enumerate(sel):
    ax[r, 0].imshow(real[i], cmap="gray", vmin=0, vmax=255)
    ax[r, 0].set_ylabel(f"DI {DI[i]:.2f}\n{str(fn[i])[:14]}", fontsize=6)
    for c, m in enumerate(a.models):
        ax[r, c + 1].imshow(z[m]["fake"][i, 0], cmap="gray", vmin=0, vmax=255)
        if r == 0:
            ax[r, c + 1].set_title(m.replace("app_", ""), fontsize=8)
    if r == 0:
        ax[r, 0].set_title("GT", fontsize=8)
for x in ax.ravel():
    x.set_xticks([]); x.set_yticks([])
plt.tight_layout(); plt.savefig(a.out, dpi=130)
print("wrote", a.out)
