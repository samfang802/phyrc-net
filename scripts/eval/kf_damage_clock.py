"""PCA + Kalman-filter damage clock tau from the hysteresis loop, probed against R_spall.

Per half-cycle token k (src/data/damage_state.py) the observation is
  y_k = [1 - strength_ratio, 1 - stiffness_ratio, log1p(cum_energy), |residual drift|]
computed from the experimental force ('exp') or the simulated force ('sim', what the GRU / encoder see).
PCA on the non-imaged specimens gives a 1-D latent z = C^T (y - mu). The latent dynamics
  z_{k+1} = a z_k + b^T u_k + c + w,  u_k = [d log1p(max drift so far), log1p |peak drift|]
are fitted by least squares on the PCA projections; Q and diagonal R are the residual variances.
The filter skips the update on invalid tokens (return to zero) and clips the posterior mean to be
non-decreasing (tau never goes back). 'open loop' runs the dynamics without any observation.

Probe: same protocol as scripts/eval/screen_condition_encoder.py (RidgeCV, leave one imaged specimen
out, trained on R315 / R330 / R615 / R1015; error = mean prediction - mean label per drift level),
plus an isotonic probe on 1-D clocks. GRU g (Data/DamagePattern/derived/gru_g.npz) is probed the same way as
the reference. --label phys replaces R_spall by the physical spall area / a 75 x 80 cm window
(Data/DamagePattern/derived/spall_scaled_frames.csv, scenario A), which undoes the 15 % vertical stretch
of the R315 photos (docs/redesign/dataset_audit.md section 1). Output: results/kf_clock[_phys]/.
"""

import argparse

import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.getcwd())
from src.data.damage_state import IMAGE_SPECIMENS, SEG, _residual_drift, damage_targets, valid_tokens
from src.metrics import spall_labels

ROOT = "Data/Hysteresis"
TR = ["R315", "R330", "R615", "R1015"]
OBS = ("1-strength", "1-stiffness", "log1p cum_energy", "|residual drift|")
WINDOW_CM2 = 75 * 80


def specimen(e, force):
    sim = pd.read_csv(os.path.join(ROOT, "Simulate", f"{e}.csv"))
    drift = sim["Drift(%)"].to_numpy(np.float64)
    scale = float(np.abs(sim["Force(kN)"]).max())
    src = sim if force == "sim" else pd.read_csv(os.path.join(ROOT, "Experiment", f"{e}.csv"))
    f = src["Force(kN)"].to_numpy(np.float64)[: len(drift)] / scale
    t = damage_targets(drift, f)
    y = np.stack([1 - t["strength_ratio"], np.clip(1 - t["stiffness_ratio"], -0.5, 1.5),
                  np.log1p(np.maximum(t["cum_energy"], 0)), np.abs(_residual_drift(drift, f))], 1)
    pk = drift[np.arange(1, (len(drift) - 1) // SEG + 1) * SEG]
    md = np.log1p(np.maximum.accumulate(np.abs(pk)))
    u = np.stack([np.diff(md, prepend=0.0), np.log1p(np.abs(pk))], 1)
    return y, u, valid_tokens(drift), md


def fit(data, ids):
    Y = np.concatenate([data[e][0][data[e][2]] for e in ids])
    mu, sd = Y.mean(0), Y.std(0) + 1e-6
    Yn = (Y - mu) / sd
    w, V = np.linalg.eigh(np.cov(Yn.T))
    C = V[:, -1] * np.sign(V[:, -1].sum())          # damage increases along +z
    evr = w[::-1] / w.sum()
    Zp, Zn, U = [], [], []
    for e in ids:
        y, u, v, _ = data[e]
        z = ((y - mu) / sd) @ C
        z[~v] = np.maximum.accumulate(np.where(v, z, -np.inf))[~v]  # carry the last valid value
        Zp.append(np.r_[0.0, z[:-1]]); Zn.append(z); U.append(u)
    Zp, Zn, U = map(np.concatenate, (Zp, Zn, U))
    A = np.c_[Zp, U, np.ones_like(Zp)]
    th, *_ = np.linalg.lstsq(A, Zn, rcond=None)
    Q = float(np.var(Zn - A @ th))
    Rd = np.var(Yn - np.outer(Yn @ C, C), 0) + 1e-3
    z0 = float(((np.zeros(len(mu)) - mu) / sd) @ C)  # undamaged column
    return dict(mu=mu, sd=sd, C=C, a=th[0], b=th[1:3], c=th[3], Q=Q, R=np.diag(Rd), z0=z0, evr=evr)


def run_kf(m, y, u, v, observe=True):
    z, P, out = m["z0"], 0.0, []
    for k in range(len(u)):
        zp, Pp = m["a"] * z + m["b"] @ u[k] + m["c"], m["a"] ** 2 * P + m["Q"]
        if observe and v[k]:
            H = m["C"][:, None]
            S = Pp * H @ H.T + m["R"]
            K = Pp * H.T @ np.linalg.inv(S)
            r = (y[k] - m["mu"]) / m["sd"] - H[:, 0] * zp
            zp, Pp = zp + float((K @ r)[0]), float((1 - K @ H)[0, 0] * Pp)
        z, P = (max(zp, z) if k else zp), Pp
        out.append(z)
    return np.r_[m["z0"], out]                      # index t // SEG, row 0 = before the first peak


def probe(X, F, y, trm, iso=False):
    out = np.full(len(F), np.nan)
    ok = F[y].notna().values
    for e in IMAGE_SPECIMENS:
        tr = ok & trm & (F.exp.values != e)
        te = F.exp.values == e
        if iso:
            out[te] = IsotonicRegression(out_of_bounds="clip").fit(X[tr, 0], F.loc[tr, y]).predict(X[te, 0])
        else:
            out[te] = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(X[tr], F.loc[tr, y]).predict(X[te])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=("anylabel", "phys"), default="anylabel")
    cli = ap.parse_args()
    out = "results/kf_clock" + ("_phys" if cli.label == "phys" else "")
    os.makedirs(out, exist_ok=True)
    fit_ids = sorted(f[:-4] for f in os.listdir(os.path.join(ROOT, "Simulate")) if f[:3].isdigit())
    lab = pd.read_csv("Data/DamagePattern/label.csv")
    F = lab[lab.exp.isin(IMAGE_SPECIMENS)].reset_index(drop=True)
    F["level"] = F.filename.str.split("_").str[2].astype(int)
    if cli.label == "phys":
        sc = pd.read_csv("Data/DamagePattern/derived/spall_scaled_frames.csv").query("scenario == 'A'")
        F["R_spall"] = F.filename.map(sc.set_index("filename").area_cm2 / WINDOW_CM2)
    else:
        F["R_spall"] = F.filename.map(spall_labels.load())
    k = (F.timestep // SEG).astype(int).values
    trm = F.exp.isin(TR).values
    gz = np.load("Data/DamagePattern/derived/gru_g.npz")
    gi = {f: i for i, f in enumerate(gz["filename"])}
    lines, models = [], {}
    reps = {"GRU g": gz["g"][[gi[f] for f in F.filename]]}

    def emit(title, obj):
        s = f"\n### {title}\n{obj.to_string() if hasattr(obj, 'to_string') else obj}\n"
        print(s); lines.append(s)

    for force in ("sim", "exp"):
        data = {e: specimen(e, force) for e in fit_ids + list(IMAGE_SPECIMENS)}
        m = fit(data, fit_ids); models[force] = m
        emit(f"[{force}] model", pd.Series({"PC1 explained": m["evr"][0], "PC2 explained": m["evr"][1],
                                             "a": m["a"], "b_dmax": m["b"][0], "b_|peak|": m["b"][1], "Q": m["Q"],
                                             **{f"C {o}": c for o, c in zip(OBS, m["C"])}}).round(3))
        tab = {e: {} for e in IMAGE_SPECIMENS}
        for e in IMAGE_SPECIMENS:
            y, u, v, md = data[e]
            tab[e]["KF"] = run_kf(m, y, u, v)
            tab[e]["open"] = run_kf(m, y, u, v, observe=False)
            pca = ((y - m["mu"]) / m["sd"]) @ m["C"]
            tab[e]["PCA"] = np.r_[m["z0"], np.maximum.accumulate(np.where(v, pca, -np.inf))]
            tab[e]["obs"] = np.r_[np.zeros((1, 4)), (y - m["mu"]) / m["sd"]]
            tab[e]["md"] = np.r_[0.0, md]
        pick = lambda key: np.stack([tab[e][key][kk] for e, kk in zip(F.exp, k)]).reshape(len(F), -1)
        if force == "sim":
            reps["max drift (log1p)"] = pick("md")
        reps[f"[{force}] 4 obs (ridge)"] = pick("obs")
        reps[f"[{force}] PCA1 running max"] = pick("PCA")
        reps[f"[{force}] KF open loop"] = pick("open")
        reps[f"[{force}] KF tau"] = pick("KF")
        reps[f"[{force}] KF tau + max drift"] = np.c_[pick("KF"), pick("md")]
        F[f"tau_{force}"] = pick("KF")[:, 0]

    for name in list(reps):
        X = reps[name]; reps[name] = (X - X[trm].mean(0)) / (X[trm].std(0) + 1e-6)

    rows = []
    for name, X in reps.items():
        for iso in ((False, True) if X.shape[1] == 1 else (False,)):
            F["hat"] = probe(X, F, "R_spall", trm, iso)
            g = F.groupby(["exp", "level"])[["R_spall", "hat"]].mean()
            err = g.hat - g.R_spall
            loso = g[g.index.get_level_values(0).isin(TR)]
            r = {"rep": name + (" [isotonic]" if iso else "")}
            for e, lv in (("CTR1", 6000), ("CTR1", 8000), ("R307", 6000), ("R307", 8000)):
                r[f"{e} {lv // 1000}%"] = err.get((e, lv), np.nan)
            r["LOSO RMSE"] = float(np.sqrt(((loso.hat - loso.R_spall) ** 2).mean()))
            rows.append(r)
    emit(f"R_spall ({cli.label}) probe error = predicted - actual (mean over frames of the level)",
         pd.DataFrame(rows).set_index("rep").round(3))

    lv = F[F.level.isin([3000, 4000, 6000, 8000])].assign(face=lambda d: d.filename.str.split("_").str[1])
    last = lv.sort_values("timestep").groupby(["exp", "face", "level"]).last().groupby(["level", "exp"])
    emit("KF tau at the last labelled frame of each drift level (mean of faces) vs R_spall",
         last[["tau_sim", "tau_exp", "R_spall"]].mean().round(3))
    F.drop(columns="hat").to_csv(os.path.join(out, "frames.csv"), index=False)
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
