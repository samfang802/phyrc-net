#!/usr/bin/env python3
"""Phase-0 clock extrapolation check (docs/redesign/phase0_clock.md, docs/redesign/README.md §5).

Question: is there a monotone damage clock tau, computed only from the simulated hysteresis loop and the
column parameters (both known at inference), such that a monotone regression R_spall = f(tau) fitted on the
4 training columns (8 column-face sequences) extrapolates the spalled area of CTR1 (test) and R307 (val)?

Labels: Data/DamagePattern/anylabel_r_spall.csv (hand-drawn spall polygons), frame metadata from label.csv,
split from id_list_val_R307.json. Every frame at loop timestep t uses the loop up to t (t is a multiple of 30,
i.e. a drift peak).

Clocks (all non-decreasing in t, except the raw encoder outputs, which are made monotone with a running max):
  max_drift       largest |drift| so far (%)                                         baseline
  E_raw           cumulative dissipated energy of the simulated loop (kN*m)
  E_FL            E / (F_max,sim * L) * 100  (= the encoder token cum_energy, drift-% units)
  E_FyDy          E / (F_max,sim * delta_y), delta_y = displacement at 0.75 F_max / 0.75 (sim loop)
  E_c             E / (f'c * A_g * L) * 1e3   (energy per unit concrete capacity)
  mu              max drift / yield drift (displacement ductility, sim loop)
  enc_strength    1 - encoder strength ratio (running max)          } v11 damage scalars,
  enc_stiffness   1 - encoder stiffness ratio (running max)         } A_info encoder run
  enc_energy      encoder cumulative energy (running max)           }
  BE_spall        max drift / Berry-Eberhard spalling-onset drift (PEER 2003-18 Eq. 7.3)
  BE_buckle       max drift / Berry-Eberhard bar-buckling-onset drift (Eq. 7.2, k_e = 50 rectangular)
  BE_spall_w      as BE_spall with D = width instead of depth (loading direction not verified)
  m_drift         max drift * F_max,sim * L / (f'c * b * D^2)  (drift x normalized moment capacity)
  combos          2-D monotone logistic on two clocks

Models: isotonic (flat outside the training range), 3-parameter logistic (slope >= 0, ceiling <= 1), hinge
linear R = w * max(0, tau - tau0) with w >= 0. The model per clock is chosen by leave-one-column-out RMSE on
the 4 training columns, then refit on all 4 and applied to R307 and CTR1.

  OMP_NUM_THREADS=2 python scripts/eval/phase0_clock_check.py [--out results/phase0/clock]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.getcwd())

TRAIN = ["R315", "R330", "R615", "R1015"]
VAL, TEST = "R307", "CTR1"
COLS = TRAIN + [VAL, TEST]
SEG = 30
ENC_RUN = "results/DamageStateEncoder/A_info/2026-10-23_22-34-47"
PASS_TOL = 0.10
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ----------------------------------------------------------------------------------------------- data
def load_frames(r_spall_csv):
    lab = pd.read_csv("Data/DamagePattern/label.csv")
    rs = pd.read_csv(r_spall_csv)[["filename", "R_spall"]]
    df = lab.merge(rs, on="filename", how="inner")
    parts = df["filename"].str.replace(".jpg", "", regex=False).str.split("_")
    df["face"] = parts.str[1]
    df["level"] = parts.str[2].astype(int) / 1000.0
    df["cycle"] = parts.str[3].astype(int)
    split = json.load(open("Data/DamagePattern/id_list_val_R307.json"))
    which = {f: k for k, v in split.items() for f in v}
    df["split"] = df["filename"].map(which)
    for k, exps in (("train", TRAIN), ("val", [VAL]), ("test", [TEST])):
        got = sorted(df.loc[df["split"] == k, "exp"].unique())
        assert got == sorted(exps), (k, got)
    df = df[df["exp"].isin(COLS)].copy()
    assert (df["timestep"] % SEG == 0).all()
    return df.reset_index(drop=True)


def berry_eberhard(row, depth_key="depth"):
    """Berry & Eberhard (PEER 2003-18). Eq. 7.3: Delta_spall/L (%) = 1.6 (1 - P/(Ag f'c)) (1 + L/(10 D)).
    Eq. 7.2: Delta_bb/L (%) = 3.25 (1 + k_e rho_eff d_b/D) (1 - P/(Ag f'c)) (1 + L/(10 D)),
    rho_eff = rho_s f_ys / f'c, k_e = 50 (rectangular ties)."""
    fc, b, h = float(row["concreteStrength"]), float(row["width"]), float(row["depth"])
    D = float(row[depth_key])
    L = float(row["lMeasured"])
    n = float(row["axialLoad"]) * 1e3 / (fc * b * h)
    rho_eff = float(row["volTransReinfRatio"]) * float(row["transverse_yield"]) / fc
    db = float(row["longitudinal_corner_diameter"])
    sp = 1.6 * (1 - n) * (1 + L / (10 * D))
    bb = 3.25 * (1 + 50 * rho_eff * db / D) * (1 - n) * (1 + L / (10 * D))
    return {"n_axial": n, "L_over_D": L / D, "rho_eff": rho_eff, "db_over_D": db / D, "BE_spall_drift": sp, "BE_buckle_drift": bb}


def loop_clocks(exp, info):
    """Per-peak clocks from the simulated loop. Index k = t // 30 (k = 0 is the unloaded state)."""
    sim = pd.read_csv(os.path.join("Data/Hysteresis/Simulate", f"{exp}.csv"))
    drift = sim["Drift(%)"].to_numpy(float)
    disp = sim["Displacement(mm)"].to_numpy(float)
    force = sim["Force(kN)"].to_numpy(float)
    m = (len(drift) - 1) // SEG
    seg_e = np.array([abs(_trapz(force[SEG * (k - 1): SEG * k + 1], disp[SEG * (k - 1): SEG * k + 1])) for k in range(1, m + 1)])
    E = np.concatenate([[0.0], np.cumsum(seg_e)]) / 1e3  # kN*mm -> kN*m
    peaks = np.arange(0, m + 1) * SEG
    maxd = np.maximum.accumulate(np.abs(drift))[peaks]
    row = info.loc[exp]
    fc, b, h, L = (float(row[k]) for k in ("concreteStrength", "width", "depth", "lMeasured"))
    fmax = float(np.abs(force).max())
    i75 = int(np.argmax(np.abs(force) >= 0.75 * fmax))
    dy_mm = abs(disp[i75]) / 0.75
    dy_pct = dy_mm / L * 100
    be = berry_eberhard(row)
    be_w = berry_eberhard(row, "width")
    m_n = fmax * 1e3 * L / (fc * b * h ** 2)
    out = {
        "max_drift": maxd,
        "E_raw": E,
        "E_FL": E * 1e3 / (fmax * L) * 100,
        "E_FyDy": E * 1e3 / (fmax * dy_mm),
        "E_c": E * 1e6 / (fc * b * h * L) * 1e3,
        "mu": maxd / dy_pct,
        "BE_spall": maxd / be["BE_spall_drift"],
        "BE_buckle": maxd / be["BE_buckle_drift"],
        "BE_spall_w": maxd / be_w["BE_spall_drift"],
        "m_drift": maxd * m_n,
    }
    static = dict(be, F_max_kN=fmax, delta_y_pct=dy_pct, m_n=m_n, BE_spall_drift_w=be_w["BE_spall_drift"])
    return out, static


def encoder_clocks(exps):
    """v11 damage scalars of the A_info encoder, in raw units, per peak index k."""
    import torch  # noqa: F401  (imported by damage_condition)
    from src.data.damage_condition import build_frame_conditions
    from src.data.damage_state import TARGETS

    st = {k: np.asarray(v) for k, v in json.load(open(os.path.join(ENC_RUN, "stats.json"))).items()}
    mu_, sd = st["tgt_mean"], st["tgt_std"]
    j = {k: TARGETS.index(k) for k in ("strength_ratio", "stiffness_ratio", "cum_energy")}
    cond = build_frame_conditions(ENC_RUN, exps)
    out = {}
    for e, v in cond.items():
        s = v.numpy()[::SEG, -4:]  # rows at t = 30k; row k=0 is the undamaged state
        strength = s[:, 0] * sd[j["strength_ratio"]] + mu_[j["strength_ratio"]]
        stiff = s[:, 1] * sd[j["stiffness_ratio"]] + mu_[j["stiffness_ratio"]]
        energy = s[:, 2] * sd[j["cum_energy"]] + mu_[j["cum_energy"]]
        out[e] = {
            "enc_strength": np.maximum.accumulate(1 - strength),
            "enc_stiffness": np.maximum.accumulate(1 - stiff),
            "enc_energy": np.maximum.accumulate(energy),
            "enc_strength_raw": strength, "enc_stiffness_raw": stiff, "enc_energy_raw": energy,
        }
    return out


# --------------------------------------------------------------------------------------------- models
class Isotonic:
    name = "isotonic"

    def fit(self, x, y):
        self.m = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0, y_max=1).fit(x[:, 0], y)
        return self

    def predict(self, x):
        return self.m.predict(x[:, 0])


class Hinge:
    """R = w * max(0, tau - tau0), w >= 0, tau0 on a grid of training quantiles."""
    name = "hinge"

    def fit(self, x, y):
        x = x[:, 0]
        best = (np.inf, 0.0, 0.0)
        for t0 in np.unique(np.quantile(x, np.linspace(0, 0.98, 99))):
            h = np.maximum(0, x - t0)
            w = max(0.0, float(h @ y / max(h @ h, 1e-12)))
            r = float(np.mean((w * h - y) ** 2))
            if r < best[0]:
                best = (r, w, t0)
        _, self.w, self.t0 = best
        return self

    def predict(self, x):
        return np.clip(self.w * np.maximum(0, x[:, 0] - self.t0), 0, 1)


class Logistic:
    """R = c * sigmoid(sum_i a_i z_i + b), a_i >= 0, 0 < c <= 1, z standardized on the training data."""
    name = "logistic"

    def fit(self, x, y):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-9
        z = (x - self.mu) / self.sd
        d = z.shape[1]

        def f(p, z):
            a = np.exp(p[:d])
            c = 1 / (1 + np.exp(-p[d + 1]))
            return c / (1 + np.exp(-(z @ a + p[d])))

        best = None
        for a0 in (0.0, 1.0, 2.0):
            for b0 in (-4.0, -2.0, 0.0):
                p0 = np.r_[np.full(d, a0), b0, 0.0]
                r = least_squares(lambda p: f(p, z) - y, p0, method="lm" if len(y) > len(p0) else "trf", max_nfev=5000)
                if best is None or r.cost < best.cost:
                    best = r
        self.p, self.f = best.x, f
        return self

    def predict(self, x):
        return self.f(self.p, (x - self.mu) / self.sd)


class LogLogistic(Logistic):
    """Logistic on log(tau + eps): a power-law sigmoid. For two clocks it fits R = f(tau1^a1 * tau2^a2)."""
    name = "loglogistic"

    def _log(self, x):
        return np.log(np.maximum(x, 0) + self.eps)

    def fit(self, x, y):
        self.eps = np.array([0.05 * np.median(c[c > 0]) if (c > 0).any() else 1e-3 for c in x.T])
        return super().fit(self._log(x), y)

    def predict(self, x):
        return super().predict(self._log(x))


MODELS_1D = (Isotonic, Logistic, LogLogistic, Hinge)
MODELS_2D = (Logistic, LogLogistic)


# ------------------------------------------------------------------------------------------- analysis
def bias_report(d, pred):
    err = pred - d["R_spall"].to_numpy()
    di = d["DI"].to_numpy()
    k = max(1, int(np.ceil(len(di) / 3)))
    hi = np.argsort(di)[-k:]
    lvl = d["level"].to_numpy()
    out = {"bias_all": float(err.mean()), "bias_hiDI": float(err[hi].mean()), "rmse": float(np.sqrt(np.mean(err ** 2)))}
    for L in (4.0, 6.0, 8.0):
        s = lvl == L
        out[f"bias_{int(L)}pct"] = float(err[s].mean()) if s.any() else float("nan")
    return out


def run_clock(df, cols):
    """LOCO over the train columns for every model; refit the best on all train columns; predict val/test."""
    X = df[cols].to_numpy(float)
    y = df["R_spall"].to_numpy(float)
    tr = df["exp"].isin(TRAIN).to_numpy()
    models = MODELS_1D if len(cols) == 1 else MODELS_2D
    loco = {}
    for M in models:
        pred = np.full(len(df), np.nan)
        per = {}
        for hold in TRAIN:
            fit = tr & (df["exp"] != hold).to_numpy()
            te = (df["exp"] == hold).to_numpy()
            pred[te] = M().fit(X[fit], y[fit]).predict(X[te])
            per[hold] = float(np.sqrt(np.mean((pred[te] - y[te]) ** 2)))
        loco[M.name] = {"rmse": float(np.sqrt(np.nanmean((pred[tr] - y[tr]) ** 2))), "per_column": per,
                        "hiDI_bias": {h: bias_report(df[df["exp"] == h], pred[(df["exp"] == h).to_numpy()])["bias_hiDI"] for h in TRAIN}}
    best = min(loco, key=lambda k: loco[k]["rmse"])
    res = {"loco": loco, "best_model": best, "eval": {}}
    for name in loco:
        M = next(m for m in models if m.name == name)
        m = M().fit(X[tr], y[tr])
        res["eval"][name] = {}
        for e in (VAL, TEST):
            s = (df["exp"] == e).to_numpy()
            res["eval"][name][e] = bias_report(df[s], m.predict(X[s]))
        if name == best:
            res["fit_all"] = m
    return res


def separation(df, clock):
    """At matched (drift level, cycle) frames >= 3 %, is CTR1's clock later than R315's / R330's?"""
    g = df.groupby(["exp", "level", "cycle"])[clock].mean()
    rs = df.groupby(["exp", "level", "cycle"])["R_spall"].mean()
    out = {}
    for ref in ("R315", "R330"):
        ratios, later, d_r = [], [], []
        for (lv, cy), v in g.loc[TEST].items():
            if lv < 3 or (ref, lv, cy) not in g.index:
                continue
            r = g.loc[(ref, lv, cy)]
            ratios.append(v / r if r > 0 else np.nan)
            later.append(v > r * 1.0001)
            d_r.append(rs.loc[(TEST, lv, cy)] - rs.loc[(ref, lv, cy)])
        out[ref] = {"n": len(ratios), "frac_later": float(np.mean(later)) if later else float("nan"),
                    "median_ratio": float(np.nanmedian(ratios)) if ratios else float("nan"),
                    "mean_dR_true": float(np.mean(d_r)) if d_r else float("nan")}
    lv = {}
    for L in (3.0, 4.0, 6.0, 8.0):
        lv[L] = {e: float(df.loc[(df["exp"] == e) & (df["level"] == L), clock].mean()) for e in COLS}
    out["by_level_mean"] = lv
    return out


def plot(df, clocks, results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(clocks)
    nc = 4
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.4 * nr), squeeze=False)
    colors = {"R315": "#4C72B0", "R330": "#55A868", "R615": "#8172B2", "R1015": "#937860", "R307": "#DD8452", "CTR1": "#C44E52"}
    for ax, c in zip(axs.flat, clocks):
        for e in COLS:
            s = df[df["exp"] == e]
            ax.scatter(s[c], s["R_spall"], s=8 if e in TRAIN else 14, color=colors[e], alpha=0.55 if e in TRAIN else 0.9,
                       marker="o" if e in TRAIN else "^", label=e)
        m = results[c].get("fit_all")
        if m is not None:
            xs = np.linspace(0, df[c].max() * 1.02, 300)[:, None]
            ax.plot(xs[:, 0], m.predict(xs), "k-", lw=1.2)
        r = results[c]
        b = r["eval"][r["best_model"]][TEST]["bias_hiDI"]
        ax.set_title(f"{c} [{r['best_model']}]  CTR1 hiDI {b:+.2f}", fontsize=9)
        ax.set_ylim(-0.02, 0.72)
        ax.tick_params(labelsize=7)
    for ax in axs.flat[n:]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase0/clock")
    ap.add_argument("--r_spall_csv", default="Data/DamagePattern/anylabel_r_spall.csv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load_frames(a.r_spall_csv)
    info = pd.read_csv("Data/Hysteresis/info_w_mander_big.csv", dtype={"id": str}).set_index("id")
    per_exp, statics = {}, {}
    for e in COLS:
        per_exp[e], statics[e] = loop_clocks(e, info)
    enc = encoder_clocks(COLS)
    for e in COLS:
        per_exp[e].update(enc[e])
    loop_names = list(per_exp[TEST].keys())
    k = (df["timestep"] // SEG).to_numpy()
    for c in loop_names:
        df[c] = [per_exp[e][c][kk] for e, kk in zip(df["exp"], k)]

    clocks_1d = ["max_drift", "E_raw", "E_FL", "E_FyDy", "E_c", "mu", "enc_strength", "enc_stiffness", "enc_energy",
                 "BE_spall", "BE_buckle", "BE_spall_w", "m_drift"]
    combos = [("max_drift", "E_FyDy"), ("max_drift", "E_FL"), ("max_drift", "enc_strength"), ("BE_spall", "E_FyDy"),
              ("max_drift", "E_c"), ("m_drift", "E_FyDy"), ("BE_spall", "E_c"), ("BE_spall", "m_drift"), ("max_drift", "m_drift")]
    results = {}
    for c in clocks_1d:
        results[c] = run_clock(df, [c])
        results[c]["separation"] = separation(df, c)
    for c1, c2 in combos:
        name = f"{c1}+{c2}"
        results[name] = run_clock(df, [c1, c2])

    # ---- tables
    rows = []
    for name, r in results.items():
        b = r["best_model"]
        row = {"clock": name, "model": b, "loco_rmse": r["loco"][b]["rmse"],
               **{f"loco_hiDI_{h}": v for h, v in r["loco"][b]["hiDI_bias"].items()}}
        for e in (TEST, VAL):
            for kk, v in r["eval"][b][e].items():
                row[f"{e}_{kk}"] = v
        # selection uses only train (LOCO) and val (R307); CTR1 is reported, never used to choose
        row["select_score"] = float(np.mean([r["loco"][b]["rmse"]] + [abs(r["eval"][b][VAL][kk]) for kk in ("bias_hiDI", "bias_6pct", "bias_8pct")]))
        row["pass"] = abs(r["eval"][b][TEST]["bias_hiDI"]) < PASS_TOL
        rows.append(row)
    tab = pd.DataFrame(rows).sort_values("select_score")
    tab.to_csv(os.path.join(a.out, "clock_bias.csv"), index=False)

    allm = []
    for name, r in results.items():
        for mname, lr in r["loco"].items():
            row = {"clock": name, "model": mname, "loco_rmse": lr["rmse"], **{f"loco_{h}": v for h, v in lr["per_column"].items()},
                   **{f"loco_hiDI_bias_{h}": v for h, v in lr["hiDI_bias"].items()}}
            for e in (TEST, VAL):
                for kk, v in r["eval"][mname][e].items():
                    row[f"{e}_{kk}"] = v
            allm.append(row)
    pd.DataFrame(allm).to_csv(os.path.join(a.out, "clock_bias_all_models.csv"), index=False)

    sep_rows = []
    for c in clocks_1d:
        s = results[c]["separation"]
        sep_rows.append({"clock": c, **{f"{ref}_{kk}": v for ref in ("R315", "R330") for kk, v in s[ref].items()},
                         **{f"lvl{int(L)}_{e}": s["by_level_mean"][L][e] for L in (3.0, 4.0, 6.0, 8.0) for e in COLS}})
    pd.DataFrame(sep_rows).to_csv(os.path.join(a.out, "clock_separation.csv"), index=False)

    pd.DataFrame(statics).T.to_csv(os.path.join(a.out, "column_static.csv"))
    df.to_csv(os.path.join(a.out, "frames_with_clocks.csv"), index=False)
    plot(df, clocks_1d, results, os.path.join(a.out, "clock_fits.png"))

    # references: true high-DI mean R_spall per column, and the face-to-face noise floor. A column-level clock
    # predicts the same value for both faces, so its RMSE cannot go below sqrt(mean((R_a - R_b)^2 / 4)).
    hi, floor = {}, {}
    for e in COLS:
        s = df[df["exp"] == e]
        kk = max(1, int(np.ceil(len(s) / 3)))
        hi[e] = float(s.nlargest(kk, "DI")["R_spall"].mean())
        pv = s.pivot_table(index=["level", "cycle"], columns="face", values="R_spall").dropna()
        floor[e] = float(np.sqrt(np.mean((pv.iloc[:, 0] - pv.iloc[:, 1]) ** 2 / 4)))
    json.dump({"hiDI_mean_R_spall": hi, "face_noise_floor_rmse": floor, "pass_tol": PASS_TOL},
              open(os.path.join(a.out, "reference.json"), "w"), indent=2)

    # strict check (not the survey's criterion): |bias| < 0.1 for CTR1 and R307 on hi-DI, 6 % and 8 %, any model
    strict = []
    for name, r in results.items():
        for mname, ev in r["eval"].items():
            vals = [ev[e][kk] for e in (TEST, VAL) for kk in ("bias_hiDI", "bias_6pct", "bias_8pct")]
            strict.append({"clock": name, "model": mname, "loco_rmse": r["loco"][mname]["rmse"],
                           "is_loco_best": mname == r["best_model"], "max_abs_bias": float(np.max(np.abs(vals))),
                           "strict_pass": bool(np.all(np.abs(vals) < PASS_TOL))})
    strict = pd.DataFrame(strict).sort_values("max_abs_bias")
    strict.to_csv(os.path.join(a.out, "strict_check.csv"), index=False)

    pd.set_option("display.width", 250)
    cols_show = ["clock", "model", "loco_rmse", "CTR1_bias_all", "CTR1_bias_hiDI", "CTR1_bias_6pct", "CTR1_bias_8pct",
                 "R307_bias_all", "R307_bias_hiDI", "R307_bias_6pct", "R307_bias_8pct", "select_score", "pass"]
    print(tab[cols_show].round(3).to_string(index=False))
    print()
    print(tab[["clock", "model"] + [f"loco_hiDI_{h}" for h in TRAIN]].round(3).to_string(index=False))
    print()
    print(pd.DataFrame(sep_rows)[["clock", "R315_frac_later", "R315_median_ratio", "R330_frac_later", "R330_median_ratio", "R315_mean_dR_true"]].round(3).to_string(index=False))
    print()
    print(pd.DataFrame(statics).T.round(4).to_string())
    print(json.dumps({"hiDI_mean_R_spall": hi, "face_noise_floor_rmse": floor}, indent=1))
    print()
    print(strict.head(12).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
