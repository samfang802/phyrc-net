#!/usr/bin/env python3
"""Derived CSVs (alignment parameters, photometry with normalisation gains) and figures for docs/redesign/dataset_audit.md."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "Data/DamagePattern/derived"; R = "results/dataset_audit"
A = pd.read_csv(f"{D}/frame_audit.csv")
C = pd.read_csv("results/phase0/clock/frames_with_clocks.csv")[["filename", "level", "max_drift", "cycle"]]
A = A.merge(C, on="filename").sort_values(["seq", "timestep"]).reset_index(drop=True)
ref = A.groupby("seq")[["pitch_x", "pitch_y"]].transform(lambda x: x.iloc[:10].median())
A["grid_fit_ok"] = ((A.pitch_x - ref.pitch_x).abs() < 4) & ((A.pitch_y - ref.pitch_y).abs() < 4) & (A.gdx_ref.abs() < 30) & (A.gdy_ref.abs() < 30)
A["strip_ok"] = A.st_ref_ncc > 0.8
al = A[["filename", "seq", "timestep", "level", "max_drift", "grid_fit_ok", "gdx_ref", "gdy_ref", "gdx_prev", "gdy_prev", "strip_ok", "st_ref_dx", "st_ref_dy", "st_prev_dx", "st_prev_dy", "st_ref_ncc"]].copy()
# suggested translation to bring a frame onto the sequence's first frame (grid frame): minus the grid shift, only if the fit is ok
al["align_dx"] = np.where(al.grid_fit_ok, -al.gdx_ref, 0.0); al["align_dy"] = np.where(al.grid_fit_ok, -al.gdy_ref, 0.0)
al.to_csv(f"{D}/seq_alignment.csv", index=False)
# photometry
TARGET = 170.0
P = A[["filename", "seq", "timestep", "level", "max_drift", "conc_frac", "g_p10", "g_p50", "g_p70", "g_p90", "g_std", "mean_r", "mean_g", "mean_b",
       "grid_dark", "c128_p10", "c128_p50", "c128_p70", "c128_p90"]].copy()
P["contrast_p90_p10"] = P.g_p90 - P.g_p10; P["tint_b_minus_r"] = P.mean_b - P.mean_r
P["gain_to_p70_170"] = TARGET / P.g_p70
P["rel_p10"] = P.g_p10 / P.g_p70; P["rel_p90"] = P.g_p90 / P.g_p70
P.to_csv(f"{D}/photometry_per_frame.csv", index=False)
for c in ["g_p70", "contrast_p90_p10", "rel_p10", "rel_p90", "tint_b_minus_r"]:
    print(c, "sd total %.3f within %.3f between-seq-means %.3f" % (P[c].std(), P.groupby("seq")[c].transform(lambda x: x - x.mean()).std(), P.groupby("seq")[c].mean().std()))
# figures
seqs = sorted(A.seq.unique())
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for s in seqs:
    g = A[A.seq == s]; ax[0].plot(np.arange(len(g)), g.g_p70, lw=1, label=s)
    ok = g[g.grid_fit_ok]; ax[1].plot(ok.max_drift, ok.gdx_ref, lw=.8, alpha=.7); ax[2].plot(ok.max_drift, ok.gdy_ref, lw=.8, alpha=.7)
ax[0].set_title("concrete-region grey p70 per frame (raw)"); ax[0].set_xlabel("frame index in sequence"); ax[0].legend(fontsize=6, ncol=2)
ax[1].set_title("grid x shift vs first frame (px)"); ax[2].set_title("grid y shift vs first frame (px)")
for a_ in ax[1:]: a_.set_xlabel("max drift (%)"); a_.axhline(0, c="k", lw=.5)
plt.tight_layout(); plt.savefig(f"{R}/photometry_alignment.png", dpi=80); plt.close()
# onset maps
Z = np.load(f"{D}/onset_field.npz"); sn = list(Z["seq_names"]); on = Z["onset16"]
fig, ax = plt.subplots(2, 6, figsize=(18, 6.4))
for i, s in enumerate(sn):
    a_ = ax.flat[i]; im = a_.imshow(np.where(np.isnan(on[i]), np.nan, on[i]), cmap="viridis_r", vmin=2, vmax=8); a_.set_title(s + " onset drift (%)", fontsize=9); a_.axis("off")
plt.colorbar(im, ax=ax, shrink=.6); plt.savefig(f"{R}/onset_maps16.png", dpi=70); plt.close()
# one sequence example: coverage vs isotonic over time for 3 cells + age map
fn = Z["filenames"]; seq = Z["seq"]; ix = np.where(seq == "CTR1_NE")[0]; k = sn.index("CTR1_NE")
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
cells = [(13, 4), (13, 8), (11, 8)]
for (u, v) in cells:
    ax[0].plot(Z["cov16"][ix, u, v], ".", ms=3); ax[0].plot(Z["iso16"][ix, u, v], "-", label=f"cell ({u},{v})")
ax[0].set_title("CTR1_NE: coverage (dots) / isotonic (line)", fontsize=9); ax[0].legend(fontsize=7)
last = ix[-1]
ax[1].imshow(Z["cov16"][last], vmin=0, vmax=1, cmap="gray_r"); ax[1].set_title("last-frame coverage", fontsize=9)
ax[2].imshow(Z["onset16"][k], cmap="viridis_r", vmin=2, vmax=8); ax[2].set_title("T* onset drift %", fontsize=9)
ax[3].imshow(Z["age_drift16"][last], cmap="magma"); ax[3].set_title("age at last frame (drift-%)", fontsize=9)
for a_ in ax[1:]: a_.axis("off")
plt.tight_layout(); plt.savefig(f"{R}/onset_example_CTR1_NE.png", dpi=80); plt.close()
# spall by level compare
Cm = pd.read_csv(f"{D}/spall_compare_levels.csv"); Cm = Cm[Cm.scenario == "A"]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for e, g in Cm.groupby("exp"):
    for a_, c in zip(ax, ["R_lastcycle", "area_cm2_lastcycle", "hinge_frac_lastcycle"]):
        a_.plot(g.lv, g[c], "o-", label=e, lw=2.2 if e == "CTR1" else 1)
for a_, t in zip(ax, ["R_spall (photo fraction), last cycle of level", "spall area cm2 (10 cm grid)", "spall fraction of bottom-D band"]):
    a_.set_title(t, fontsize=9); a_.set_xlabel("drift level (%)")
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{R}/spall_scaled_compare.png", dpi=80); plt.close()
