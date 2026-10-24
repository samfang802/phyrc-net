#!/usr/bin/env python3
"""Hysteresis condition and GRU hidden state g, as the image generator sees it.

For the six imaged specimens (train R315/R330/R615/R1015, val R307, test CTR1):
  1. where their parameters sit in the 160-specimen hysteresis set
  2. GRU force-prediction error per specimen and per GRU split
  3. where each labelled frame sits on the loop
  4. e_g = LayerNorm(Linear(g)) with gn / v106F weights: variance by specimen / drift level / sign,
     nearest train frame
  5. ridge probes e_g -> R_spall / DI / log drift, leave-one-specimen-out on train specimens
  6. how much g_t depends on the loop after t (stacked bidirectional GRU)
  7. Δg between consecutive labelled frames of one face
Report: <out>/report.txt, per-frame table: <out>/frames.csv, figure: <out>/condition_g.png.
Doc: docs/eval/condition_g.md.

  python scripts/eval/analyze_condition_g.py results/condition_g
"""
import argparse, glob, importlib, json, os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import RidgeCV, LogisticRegression

sys.path.insert(0, os.getcwd())
from src.data.hysteresis import HystereticDataset

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(2)
dev = "cuda" if torch.cuda.is_available() else "cpu"
IMG = {"R315": "train", "R330": "train", "R615": "train", "R1015": "train", "R307": "val", "CTR1": "test"}
TR = [k for k, v in IMG.items() if v == "train"]
rep = {}

# ---------- 1. specimen parameters across the 160-specimen hysteresis set ----------
info = pd.read_csv("Data/Hysteresis/info_w_mander_big.csv", dtype={"id": str})
info["n_axial"] = info.axialLoad * 1e3 / (info.concreteStrength * info.width * info.depth)
info["L_over_h"] = info.lMeasured / info.depth
info["L_over_b"] = info.lMeasured / info.width
cols = ["concreteStrength", "n_axial", "L_over_h", "reinforcementRatio", "volTransReinfRatio", "axialLoad", "width", "depth", "lMeasured", "Vn"]
gsplit = json.load(open("Data/Hysteresis/id_list_big.json"))
info["gru_split"] = info.id.map({i: k for k, v in gsplit.items() for i in v}).fillna("-")
sel = info[info.id.isin(IMG)].set_index("id").loc[list(IMG)]
pct = pd.DataFrame({c: [(info[c] < sel.loc[e, c]).mean() * 100 for e in IMG] for c in cols}, index=list(IMG))
rep["params"] = sel[cols + ["gru_split", "failureType"]].round(4)
rep["params_pct"] = pct.round(0)
rep["params_range_all"] = info[cols].describe().loc[["min", "50%", "max"]].round(4)

# ---------- 2. GRU over every specimen ----------
a = argparse.Namespace(**json.load(open("phase1_extractor/args.json")))
ext = getattr(importlib.import_module("src.model.HystereticPrediction"), a.architecture)(**vars(a))
ck = glob.glob("phase1_extractor/**/best.ckpt", recursive=True)[0]
ext.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]); ext.eval().to(dev)
allids = [i for i in info.id if os.path.exists(f"Data/Hysteresis/Simulate/{i}.csv") and os.path.exists(f"Data/Hysteresis/Experiment/{i}.csv")]
ds = HystereticDataset(root="Data/Hysteresis", specified=allids, **vars(a))
H, acc = {}, []
with torch.no_grad():
    for i, e in enumerate(ds.exp_list):
        inp, tf, L, mx, _ = ds[i]
        out, _ = ext.L(inp[None].to(dev)); pf = ext.P(out).reshape(-1).cpu().numpy()
        if e in IMG:
            H[e] = out[0].cpu().numpy()
        t = tf.numpy().reshape(-1)
        acc.append(dict(id=e, gru_split=info.set_index("id").loc[e, "gru_split"], rmse=float(np.sqrt(np.mean((pf[:len(t)] - t) ** 2))),
                        peak_err=float(np.abs(pf).max() - np.abs(t).max())))
acc = pd.DataFrame(acc).set_index("id")
rep["gru_rmse_by_split"] = acc.groupby("gru_split").rmse.describe()[["count", "50%", "75%", "max"]].round(3)
rep["gru_img_specimens"] = acc.loc[list(IMG)].round(3)
rep["gru_rmse_rank_ctr1_in_val"] = f"{(acc[acc.gru_split == 'val'].rmse < acc.loc['CTR1', 'rmse']).mean() * 100:.0f} percentile"

# ---------- 3. labelled frames: where on the loop ----------
lab = pd.read_csv("Data/DamagePattern/label.csv")
lab = lab[lab.exp.isin(IMG)].copy()
lab["level"] = lab.filename.str.split("_").str[2].astype(int)
rows = []
for _, r in lab.iterrows():
    sim = pd.read_csv(f"Data/Hysteresis/Simulate/{r.exp}.csv")
    d = sim["Drift(%)"].values; t = int(r.timestep)
    rows.append(dict(exp=r.exp, file=r.filename, dirn=r.direction, level=r.level, ts=t, drift_at=d[t], max_drift_so_far=np.abs(d[: t + 1]).max(), DI=r.DI))
Fr = pd.DataFrame(rows)
rs = pd.read_csv("Data/DamagePattern/r_spall.csv").set_index("filename")["R_spall"]
Fr["R_spall"] = Fr.file.map(rs)
rep["frame_position_example"] = Fr[(Fr.exp == "CTR1") & (Fr.dirn == "Push")].sort_values("ts")[["file", "level", "ts", "drift_at", "max_drift_so_far"]].head(12).round(3)
rep["frame_drift_vs_level"] = (Fr.assign(ratio=Fr.max_drift_so_far / (Fr.level / 1000).replace(0, np.nan))
                               .groupby("exp").ratio.describe()[["min", "50%", "max"]].round(2))

# ---------- 4. e_g as the generator sees it (gn and v106F weights) ----------
G = np.stack([H[r.exp][int(r.ts)] for _, r in Fr.iterrows()])
def load_proj(path):
    sd = torch.load(path, map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd)
    W, b = sd["velocity_net.physics_linear.weight"].numpy(), sd["velocity_net.physics_linear.bias"].numpy()
    ln_w, ln_b = sd["velocity_net.e_g_ln.weight"].numpy(), sd["velocity_net.e_g_ln.bias"].numpy()
    z = G @ W.T + b
    z = (z - z.mean(1, keepdims=True)) / np.sqrt(z.var(1, keepdims=True) + 1e-5)
    return z * ln_w + ln_b, float(sd["velocity_net.alpha"].reshape(-1)[0])
gn_ck = glob.glob("results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit/**/chjqbp9m/checkpoints/last.ckpt", recursive=True)[0]
E = {}
E["gn"], al_gn = load_proj(gn_ck)
E["v106F"], al_f = load_proj("results/DamagePatternGeneration/CFMADM/2026-10-23_02-44-06_v106F/checkpoint_step_015000.pt")
rep["alpha"] = {"gn": al_gn, "v106F": al_f}

def decompose(X, mask):
    X = X[mask]; f = Fr[mask]
    tot = ((X - X.mean(0)) ** 2).sum()
    def between(key):
        s = 0.0
        for _, idx in f.groupby(key).groups.items():
            ii = f.index.get_indexer(idx); s += len(ii) * ((X[ii].mean(0) - X.mean(0)) ** 2).sum()
        return s / tot
    return {"specimen": between("exp"), "drift_level": between("level"), "drift_sign": between("sign"),
            "face": between("dirn"), "specimen x level x sign": between(["exp", "level", "sign"])}
trm = Fr.exp.isin(TR).values
Fr["sign"] = np.sign(Fr.drift_at).astype(int)
rep["var_decomp"] = pd.DataFrame({
    "raw g (train 4)": decompose(G, trm), "e_g gn (train 4)": decompose(E["gn"], trm),
    "raw g (all 6)": decompose(G, np.ones(len(Fr), bool)), "e_g gn (all 6)": decompose(E["gn"], np.ones(len(Fr), bool))}).round(3)

# specimen identity: can e_g tell the specimens apart at the same drift level?
Z = E["gn"]
clf = LogisticRegression(max_iter=2000, C=1.0)
from sklearn.model_selection import cross_val_score
rep["specimen_id_acc_from_e_g"] = float(cross_val_score(clf, Z, Fr.exp, cv=5).mean())

# nearest train frame in e_g space (cosine), per specimen and level
def nn_info(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    res = []
    for i in range(len(Fr)):
        m = trm & (Fr.exp.values != Fr.exp.values[i])
        s = Xn[m] @ Xn[i]; j = np.where(m)[0][s.argmax()]
        res.append((1 - s.max(), Fr.exp.values[j], Fr.level.values[j]))
    return res
nn = nn_info(E["gn"])
Fr["eg_nn_dist"], Fr["eg_nn_exp"], Fr["eg_nn_level"] = zip(*nn)
rep["eg_nn_dist"] = Fr.pivot_table(index="level", columns="exp", values="eg_nn_dist", aggfunc="mean").round(3)
rep["eg_nn_level_ratio"] = (Fr.assign(r=Fr.eg_nn_level / Fr.level.replace(0, np.nan))
                            .pivot_table(index="level", columns="exp", values="r", aggfunc="median").round(2))
rep["eg_nn_exp"] = Fr[Fr.exp.isin(["R307", "CTR1"])].groupby(["exp", "eg_nn_exp"]).size().unstack(fill_value=0)

# ---------- 5. probes on e_g (LOSO for train specimens) ----------
Fr["log_drift"] = np.log10(Fr.level.clip(lower=100))
def probe(y):
    out = np.full(len(Fr), np.nan)
    ok = Fr[y].notna().values
    for e in IMG:
        tr = ok & trm & (Fr.exp.values != e)
        m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Z[tr], Fr.loc[tr, y])
        out[Fr.exp.values == e] = m.predict(Z[Fr.exp.values == e])
    return out
lv = [500, 1000, 2000, 3000, 4000, 6000, 8000]
for y in ("R_spall", "DI", "log_drift"):
    Fr[y + "_hat"] = probe(y)
    t = Fr[Fr.level.isin(lv)].groupby(["exp", "level"]).agg(a=(y, "mean"), p=(y + "_hat", "mean"))
    rep[f"probe_{y}"] = t.apply(lambda r: f"{r.a:.2f}/{r.p:.2f}", axis=1).unstack().loc[list(IMG)]

# ---------- 6. bidirectional: does g_t use the future of the loop? ----------
cs = []
with torch.no_grad():
    for e in ["R315", "R307", "CTR1"]:
        i = ds.exp_list.index(e); inp = ds[i][0]
        f = Fr[(Fr.exp == e) & (Fr.dirn == "Push")].sort_values("ts")
        for _, r in f.iloc[::2].iterrows():
            t = int(r.ts)
            full = H[e][t]
            trunc = ext.L(inp[None, : t + 1].to(dev))[0][0, t].cpu().numpy()
            fwd = slice(0, len(full) // 2)
            cs.append(dict(exp=e, level=r.level, cos_all=float(np.dot(full, trunc) / np.linalg.norm(full) / np.linalg.norm(trunc)),
                           cos_fwd_half=float(np.dot(full[fwd], trunc[fwd]) / np.linalg.norm(full[fwd]) / np.linalg.norm(trunc[fwd])),
                           cos_bwd_half=float(np.dot(full[len(full)//2:], trunc[len(full)//2:]) / np.linalg.norm(full[len(full)//2:]) / np.linalg.norm(trunc[len(full)//2:]))))
rep["causality"] = pd.DataFrame(cs).round(3)

# ---------- 7. Δg between consecutive labelled frames of one face (the refiner's previous frame) ----------
En = Z / np.linalg.norm(Z, axis=1, keepdims=True)
dg = []
for (e, d), f in Fr.sort_values("ts").groupby(["exp", "dirn"]):
    ii = f.index.values
    for a_, b_ in zip(ii[:-1], ii[1:]):
        dg.append(dict(exp=e, same_sign=bool(Fr.sign[a_] == Fr.sign[b_]),
                       dg_rel=float(np.linalg.norm(G[b_] - G[a_]) / np.linalg.norm(G[a_])), cos_eg=float(En[a_] @ En[b_])))
dg = pd.DataFrame(dg)
rep["delta_g"] = dg.groupby("same_sign").agg(pairs=("dg_rel", "size"), dg_rel_median=("dg_rel", "median"), cos_eg_median=("cos_eg", "median")).round(3)

# ---------- figure ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
col = {"R315": "C0", "R330": "C1", "R615": "C2", "R1015": "C3", "R307": "C4", "CTR1": "k"}
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
for e in IMG:
    s_ = pd.read_csv(f"Data/Hysteresis/Experiment/{e}.csv"); mxf = pd.read_csv(f"Data/Hysteresis/Simulate/{e}.csv")["Force(kN)"].abs().max()
    lw = 1.6 if e in ("CTR1", "R307") else 0.7
    ax[0, 0].plot(s_["Drift(%)"], s_["Force(kN)"], color=col[e], lw=lw, label=e)
    ax[0, 1].plot(s_["Drift(%)"], s_["Force(kN)"] / mxf, color=col[e], lw=lw, label=e)
U, S, Vt = np.linalg.svd(Z[trm] - Z[trm].mean(0), full_matrices=False)
P = (Z - Z[trm].mean(0)) @ Vt[:2].T
for e in IMG:
    m = (Fr.exp == e).values
    ax[1, 0].scatter(P[m, 0], P[m, 1], s=14 if e in ("CTR1", "R307") else 6, color=col[e], label=e)
    f = Fr[m & Fr.level.isin([3000, 4000, 6000, 8000]).values].groupby("level")[["R_spall", "R_spall_hat"]].mean()
    ax[1, 1].plot(f.R_spall, f.R_spall_hat, "-o", color=col[e], label=e)
ax[1, 1].plot([0, 0.65], [0, 0.65], "--", color="0.6")
ax[0, 0].set(title="Experiment force (kN)", xlabel="drift (%)")
ax[0, 1].set(title="Force / max simulated force (GRU input scale)", xlabel="drift (%)")
ax[1, 0].set(title=f"e_g (gn weights), PCA of train frames ({(S[:2]**2/(S**2).sum()).round(2)})", xlabel="PC1", ylabel="PC2")
ax[1, 1].set(title="R_spall: actual vs probe from e_g (3-8% drift)", xlabel="actual", ylabel="predicted")
for x in ax.ravel(): x.grid(alpha=.3); x.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "condition_g.png"), dpi=120)

with open(os.path.join(OUT, "report.txt"), "w") as fh:
    for k, v in rep.items():
        s = f"\n### {k}\n{v.to_string() if hasattr(v, 'to_string') else v}\n"
        print(s); fh.write(s)
Fr.to_csv(os.path.join(OUT, "frames.csv"), index=False)
