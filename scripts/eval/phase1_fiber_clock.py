#!/usr/bin/env python3
"""Mechanics-based damage clocks from a fiber section analysis (follow-up to docs/redesign/phase0_clock.md).

Why: phase0_clock found no clock that fixes CTR1 (under-predicted) and R307 (over-predicted) together, and
phase0_spd_onset found that CTR1 starts spalling at the same drift as R315 (2.97 vs 3.02 %): the difference
is how fast spalling grows after onset. At L/D = 3 the 6 % spall area follows rho_l (R307 0.10, R315 0.19,
CTR1 0.37, R330 0.35), i.e. the compression depth. A fiber section turns drift into concrete strain using
rho_l, axial load, section, cover and confinement from mechanics, so no coefficient of these has to be learned
from the 4 training columns (whose n, rho_t and f'c are constant).

Per column (info_w_mander_big.csv), monotonic moment-curvature:
  cover concrete   Mander unconfined (f'c, eps_co 0.002), linear descent to 0 at eps_sp,u 0.006
  core concrete    Mander confined (fcc, epscc from the csv; Popovics form), 0 beyond epscu
  steel            bilinear, E 200 GPa, hardening to f_u at 0.12; As = rho_l * b * h spread equally over
                   4 + 2 p + 2 q bar positions (corner + intermediate counts from the csv; the larger
                   intermediate count is put on the longer side)
  bending          compression face = width b, lever arm along depth h (as phase0_clock); *_w swaps them
Drift -> base curvature (Priestley, Calvi & Kowalsky 2007): L_sp = 0.022 f_y d_b, L_p = max(k L + L_sp, 2 L_sp),
k = min(0.2 (f_u / f_y - 1), 0.08), phi_y = phi'_y M_n / M'_y (M_n at eps_c = 0.004),
Delta = phi (L + L_sp)^2 / 3 (elastic) or Delta_y + (phi - phi_y) L_p L.

Clocks (per loop peak, non-decreasing):
  eps_max   extreme-fiber compression strain at the base at the largest drift so far      (primary)
  eps_cum   sum over peaks so far of max(0, eps_peak - eps_sp), eps_sp = 0.004 (repeated cycles count)
  hsp_P     spalled height (mm), plastic curvature linear over L_p: L_p (phi_b - phi_sp) / (phi_b - phi_y)
  hsp_PD    hsp_P / D
  hsp_M     spalled height from the moment gradient: L (1 - M_sp / M_b), M on the monotone envelope
  hsp_MD    hsp_M / D
  eps_max_w eps_max with b and h swapped (bending direction not verified)
Baselines reproduced from phase0_clock_check: max_drift, m_drift, BE_spall.

Same protocol as phase0_clock_check: monotone regressions, model chosen by leave-one-column-out on the 4
train columns, refit on all 4; R307 (val) and CTR1 (test) are reported. The selection score uses LOCO + R307
only; CTR1 never chooses anything.

  OMP_NUM_THREADS=2 python scripts/eval/phase1_fiber_clock.py [--out results/phase1_fiber_clock] [--sweep]

--sweep: the 27 combinations of eps_sp {0.003, 0.004, 0.005}, k cap {0.06, 0.08, 0.10} and L_sp coefficient
{0.0154, 0.022, 0.0286} (+-30 %) for hsp_P and eps_max -> sweep.csv, sweep_summary.json.
"""

import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq

_spec = importlib.util.spec_from_file_location("p0clock", os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase0_clock_check.py"))
P0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P0)
TRAIN, VAL, TEST, COLS, SEG, PASS_TOL = P0.TRAIN, P0.VAL, P0.TEST, P0.COLS, P0.SEG, P0.PASS_TOL

ES = 200e3
EPS_SP = 0.004


# ----------------------------------------------------------------------------------------------- materials
def mander(eps, fpk, epk, Ec, eps_end):
    """Popovics / Mander curve, compression positive; 0 in tension and beyond eps_end."""
    e = np.maximum(eps, 0.0)
    r = Ec / (Ec - fpk / epk)
    x = e / epk
    s = fpk * x * r / (r - 1 + x ** r)
    return np.where((eps > 0) & (eps <= eps_end), s, 0.0)


def cover_stress(eps, fc):
    eco, esp = 0.002, 0.006
    Ec = 5000 * np.sqrt(fc)
    s = mander(np.minimum(eps, 2 * eco), fc, eco, Ec, 2 * eco)
    s2c = float(mander(np.array(2 * eco), fc, eco, Ec, 2 * eco))
    tail = s2c * (esp - eps) / (esp - 2 * eco)
    return np.where(eps <= 2 * eco, s, np.where(eps < esp, tail, 0.0))


def steel_stress(eps, fy, fu):
    ey = fy / ES
    Eh = (fu - fy) / (0.12 - ey)
    a = np.abs(eps)
    s = np.where(a <= ey, ES * a, np.minimum(fy + Eh * (a - ey), fu))
    return np.sign(eps) * s


# ----------------------------------------------------------------------------------------------- section
class Section:
    def __init__(self, row, swap=False, n_layers=400):
        self.fc = float(row["concreteStrength"])
        b, h = float(row["width"]), float(row["depth"])
        if swap:
            b, h = h, b
        self.b, self.h = b, h
        cc = float(row["perpendicular_clearCover"])
        dt = float(row["transverse_diameter"])
        self.db = float(row["longitudinal_corner_diameter"])
        self.fy = float(row["longitudinal_corner_yield"])
        self.fu = float(row["longitudinal_corner_strength"])
        self.fcc, self.ecc, self.ecu = float(row["fcc"]), float(row["epscc"]), float(row["epscu"])
        self.P = float(row["axialLoad"]) * 1e3
        self.n_axial = self.P / (self.fc * b * h)
        self.Ec = 5000 * np.sqrt(self.fc)

        edges = np.linspace(-h / 2, h / 2, n_layers + 1)
        self.y = 0.5 * (edges[1:] + edges[:-1])          # + toward the compression face
        dy = np.diff(edges)
        core_y = np.abs(self.y) <= h / 2 - cc
        self.A_core = np.where(core_y, (b - 2 * cc) * dy, 0.0)
        self.A_cov = b * dy - self.A_core

        p = int(row["perpendicular_numberIntermediateBars"])
        q = int(row["parallel_numberIntermediateBars"])
        n_face, n_side = (max(p, q), min(p, q)) if b >= h else (min(p, q), max(p, q))
        ye = h / 2 - cc - dt - self.db / 2
        ys = [ye] * (2 + n_face) + [-ye] * (2 + n_face) + list(np.linspace(-ye, ye, n_side + 2)[1:-1]) * 2
        self.ys = np.array(ys)
        self.As_total = float(row["reinforcementRatio"]) * b * h
        self.As = np.full(len(ys), self.As_total / len(ys))
        self.y_bar_t = -ye

    def forces(self, e0, phi):
        ec = e0 + phi * self.y
        es = e0 + phi * self.ys
        sc = mander(ec, self.fcc, self.ecc, self.Ec, self.ecu) * self.A_core + cover_stress(ec, self.fc) * self.A_cov
        ss = steel_stress(es, self.fy, self.fu) * self.As
        return sc.sum() + ss.sum(), (sc * self.y).sum() + (ss * self.ys).sum()

    def solve(self, phi):
        f = lambda e0: self.forces(e0, phi)[0] - self.P
        lo, hi = -phi * self.h / 2 - 0.02, phi * self.h / 2 + 0.004
        e0 = brentq(f, lo, hi, xtol=1e-12)
        N, M = self.forces(e0, phi)
        return e0, M

    def moment_curvature(self, n=500, eps_sp=None):
        eps_sp = EPS_SP if eps_sp is None else eps_sp
        phis = np.geomspace(1e-8, 0.06 / self.h * 4, n)
        out = []
        for ph in phis:
            e0, M = self.solve(ph)
            e_top = e0 + ph * self.h / 2
            e_st = e0 + ph * self.y_bar_t
            out.append((ph, M, e_top, e_st))
            if e_top > self.ecu:
                break
        a = np.array(out)
        self.phi, self.M, self.e_top, self.e_st = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        assert np.all(np.diff(self.e_top) > 0), "extreme strain not monotone in curvature"
        ey = self.fy / ES
        i_y = int(np.argmax((-self.e_st >= ey) | (self.e_top >= 0.002)))
        self.phi_y1, self.M_y1 = self.phi[i_y], self.M[i_y]
        self.M_n = float(np.interp(0.004, self.e_top, self.M))
        self.phi_y = self.phi_y1 * self.M_n / self.M_y1
        self.phi_sp = float(np.interp(eps_sp, self.e_top, self.phi))
        self.M_env = np.maximum.accumulate(self.M)
        self.M_sp = float(np.interp(self.phi_sp, self.phi, self.M_env))
        return self


class Member:
    def __init__(self, sec, L, k_cap=0.08, lsp_coef=0.022):
        self.s, self.L = sec, L
        self.Lsp = lsp_coef * sec.fy * sec.db
        k = min(0.2 * (sec.fu / sec.fy - 1), k_cap)
        self.Lp = max(k * L + self.Lsp, 2 * self.Lsp)
        self.Dy = sec.phi_y * (L + self.Lsp) ** 2 / 3

    def phi_base(self, drift_pct):
        D = np.abs(np.asarray(drift_pct, float)) / 100 * self.L
        el = 3 * D / (self.L + self.Lsp) ** 2
        pl = self.s.phi_y + (D - self.Dy) / (self.Lp * self.L)
        return np.where(D <= self.Dy, el, pl)

    def eps(self, phi):
        s = self.s
        # beyond the computed range: extend with the last neutral-axis depth
        c_last = s.e_top[-1] / s.phi[-1]
        return np.where(phi <= s.phi[-1], np.interp(phi, s.phi, s.e_top), phi * c_last)

    def hsp_P(self, phi):
        s = self.s
        return np.where(phi > s.phi_sp, self.Lp * (phi - s.phi_sp) / np.maximum(phi - s.phi_y, 1e-12), 0.0)

    def hsp_M(self, phi):
        s = self.s
        Mb = np.interp(np.minimum(phi, s.phi[-1]), s.phi, s.M_env)
        return np.where(phi > s.phi_sp, self.L * np.maximum(0.0, 1 - s.M_sp / Mb), 0.0)


def fiber_clocks(exp, row, eps_sp=EPS_SP, k_cap=0.08, lsp_coef=0.022, swaps=(False, True)):
    sim = pd.read_csv(os.path.join("Data/Hysteresis/Simulate", f"{exp}.csv"))
    drift = sim["Drift(%)"].to_numpy(float)
    m = (len(drift) - 1) // SEG
    peaks = np.arange(0, m + 1) * SEG
    d_peak = np.abs(drift[peaks])
    maxd = np.maximum.accumulate(np.abs(drift))[peaks]
    L = float(row["lMeasured"])
    out, static = {}, {}
    for swap in swaps:
        suf = "_w" if swap else ""
        sec = Section(row, swap=swap).moment_curvature(eps_sp=eps_sp)
        mem = Member(sec, L, k_cap=k_cap, lsp_coef=lsp_coef)
        phi_max = mem.phi_base(maxd)
        e_max = mem.eps(phi_max)
        out[f"eps_max{suf}"] = e_max
        if swap:
            static["Mn_over_L_kN_w"] = sec.M_n / L / 1e3
            continue
        e_peak = mem.eps(mem.phi_base(d_peak))
        out["eps_cum"] = np.cumsum(np.maximum(0.0, e_peak - eps_sp))
        hP, hM = mem.hsp_P(phi_max), mem.hsp_M(phi_max)
        out.update(hsp_P=hP, hsp_PD=hP / sec.h, hsp_M=hM, hsp_MD=hM / sec.h)
        c_n = float(np.interp(0.004, sec.e_top, sec.e_top / sec.phi))
        # drift at which eps_top reaches eps_sp (invert the monotone drift -> eps map numerically)
        grid = np.linspace(0, 15, 3001)
        eg = mem.eps(mem.phi_base(grid))
        static.update(n_axial=sec.n_axial, rho_l=sec.As_total / (sec.b * sec.h), c_over_h_at_0p004=c_n / sec.h,
                      phi_y=sec.phi_y, phi_sp=sec.phi_sp, Mn_over_L_kN=sec.M_n / L / 1e3, Lp_mm=mem.Lp, Lsp_mm=mem.Lsp,
                      drift_y_pct=mem.Dy / L * 100, drift_at_eps_sp_pct=float(np.interp(eps_sp, eg, grid)),
                      eps_at_4pct=float(np.interp(4, grid, eg)), eps_at_6pct=float(np.interp(6, grid, eg)),
                      eps_at_8pct=float(np.interp(8, grid, eg)))
    return out, static


# ----------------------------------------------------------------------------------------------- sweep
SWEEP = {"eps_sp": (0.003, 0.004, 0.005), "k_cap": (0.06, 0.08, 0.10), "lsp_coef": (0.0154, 0.022, 0.0286)}
NOMINAL = {"eps_sp": 0.004, "k_cap": 0.08, "lsp_coef": 0.022}
SWEEP_CLOCKS = ("hsp_P", "eps_max")
CELLS = [(e, kk) for e in (TEST, VAL) for kk in ("bias_hiDI", "bias_6pct", "bias_8pct")]


def sweep(df0, info, out_dir):
    """Robustness of the conclusion over the uncertain mechanics constants (27 settings). Nothing is chosen
    on CTR1: per setting each clock keeps its LOCO-best model (train columns only), and a fixed model
    (the nominal LOCO-best one) is reported as well."""
    import itertools
    nominal_model = {}
    rows = []
    for vals in itertools.product(*SWEEP.values()):
        cfg = dict(zip(SWEEP.keys(), vals))
        df = df0.copy()
        per = {e: fiber_clocks(e, info.loc[e], swaps=(False,), **cfg)[0] for e in COLS}
        k = (df["timestep"] // SEG).to_numpy()
        for c in SWEEP_CLOCKS:
            df[c] = [per[e][c][kk] for e, kk in zip(df["exp"], k)]
        for c in SWEEP_CLOCKS:
            r = P0.run_clock(df, [c])
            if cfg == NOMINAL:
                nominal_model[c] = r["best_model"]
            for mname in (r["best_model"], "isotonic", "loglogistic"):
                ev = r["eval"][mname]
                v = np.array([ev[e][kk] for e, kk in CELLS])
                rows.append({**cfg, "clock": c, "model": mname, "is_loco_best": mname == r["best_model"],
                             "loco_rmse": r["loco"][mname]["rmse"], **{f"{e}_{kk}": x for (e, kk), x in zip(CELLS, v)},
                             "max_abs_bias": float(np.abs(v).max()), "strict_pass": bool((np.abs(v) < PASS_TOL).all()),
                             "both_dirs_ok": bool(abs(ev[TEST]["bias_6pct"]) < PASS_TOL and abs(ev[VAL]["bias_6pct"]) < PASS_TOL)})
        print(f"[sweep] {cfg} done", flush=True)
    tab = pd.DataFrame(rows).drop_duplicates(["eps_sp", "k_cap", "lsp_coef", "clock", "model"])
    tab.to_csv(os.path.join(out_dir, "sweep.csv"), index=False)
    pd.set_option("display.width", 250)
    show = ["eps_sp", "k_cap", "lsp_coef", "model", "loco_rmse"] + [f"{e}_{kk}" for e, kk in CELLS] + ["max_abs_bias", "strict_pass"]
    summ = {}
    for c in SWEEP_CLOCKS:
        for label, sel in (("loco_best", tab["is_loco_best"]), ("fixed_" + nominal_model[c], tab["model"] == nominal_model[c])):
            t = tab[(tab["clock"] == c) & sel]
            print(f"\n== sweep {c}, {label} (27 settings)")
            print(t[show].round(3).to_string(index=False))
            summ[f"{c}/{label}"] = {"n": int(len(t)), "strict_pass": int(t["strict_pass"].sum()),
                                     "max_abs_bias_median": float(t["max_abs_bias"].median()),
                                     "max_abs_bias_worst": float(t["max_abs_bias"].max()),
                                     **{f"{e}_{kk}_range": [float(t[f"{e}_{kk}"].min()), float(t[f"{e}_{kk}"].max())] for e, kk in CELLS}}
    print("\n== sweep summary\n" + json.dumps(summ, indent=1))
    json.dump(summ, open(os.path.join(out_dir, "sweep_summary.json"), "w"), indent=1)


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase1_fiber_clock")
    ap.add_argument("--r_spall_csv", default="Data/DamagePattern/anylabel_r_spall.csv")
    ap.add_argument("--sweep", action="store_true", help="robustness sweep over the mechanics constants only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = P0.load_frames(a.r_spall_csv)
    info = pd.read_csv("Data/Hysteresis/info_w_mander_big.csv", dtype={"id": str}).set_index("id")
    if a.sweep:
        return sweep(df, info, a.out)
    per_exp, statics = {}, {}
    for e in COLS:
        base, st0 = P0.loop_clocks(e, info)
        fib, st1 = fiber_clocks(e, info.loc[e])
        per_exp[e] = {k: base[k] for k in ("max_drift", "m_drift", "BE_spall")} | fib
        f_exp = pd.read_csv(os.path.join("Data/Hysteresis/Experiment", f"{e}.csv"))["Force(kN)"].abs().max()
        statics[e] = {"F_max_exp_kN": float(f_exp), "F_max_sim_kN": st0["F_max_kN"], "BE_spall_drift": st0["BE_spall_drift"]} | st1
    k = (df["timestep"] // SEG).to_numpy()
    names = list(per_exp[TEST].keys())
    for c in names:
        df[c] = [per_exp[e][c][kk] for e, kk in zip(df["exp"], k)]

    clocks_1d = names
    combos = [("max_drift", "eps_cum"), ("eps_max", "eps_cum"), ("max_drift", "hsp_PD")]
    results = {c: P0.run_clock(df, [c]) for c in clocks_1d}
    for c1, c2 in combos:
        results[f"{c1}+{c2}"] = P0.run_clock(df, [c1, c2])

    rows, strict = [], []
    for name, r in results.items():
        b = r["best_model"]
        row = {"clock": name, "model": b, "loco_rmse": r["loco"][b]["rmse"]}
        for e in (TEST, VAL):
            for kk, v in r["eval"][b][e].items():
                row[f"{e}_{kk}"] = v
        row["select_score"] = float(np.mean([r["loco"][b]["rmse"]] + [abs(r["eval"][b][VAL][kk]) for kk in ("bias_hiDI", "bias_6pct", "bias_8pct")]))
        row["pass_hiDI"] = abs(r["eval"][b][TEST]["bias_hiDI"]) < PASS_TOL
        vals = [r["eval"][b][e][kk] for e in (TEST, VAL) for kk in ("bias_hiDI", "bias_6pct", "bias_8pct")]
        row["max_abs_bias"] = float(np.max(np.abs(vals)))
        row["strict_pass"] = bool(np.all(np.abs(vals) < PASS_TOL))
        rows.append(row)
        for mname, ev in r["eval"].items():
            v = [ev[e][kk] for e in (TEST, VAL) for kk in ("bias_hiDI", "bias_6pct", "bias_8pct")]
            strict.append({"clock": name, "model": mname, "loco_rmse": r["loco"][mname]["rmse"], "is_loco_best": mname == b,
                           "max_abs_bias": float(np.max(np.abs(v))), "strict_pass": bool(np.all(np.abs(v) < PASS_TOL))})
    tab = pd.DataFrame(rows).sort_values("select_score")
    tab.to_csv(os.path.join(a.out, "clock_bias.csv"), index=False)
    strict = pd.DataFrame(strict).sort_values("max_abs_bias")
    strict.to_csv(os.path.join(a.out, "strict_check.csv"), index=False)
    st = pd.DataFrame(statics).T
    st.to_csv(os.path.join(a.out, "column_static.csv"))
    df.to_csv(os.path.join(a.out, "frames_with_clocks.csv"), index=False)
    lvl = df[df["level"] >= 3].groupby(["exp", "level"])[["R_spall", "eps_max", "eps_cum", "hsp_PD", "max_drift"]].mean().round(4)
    lvl.to_csv(os.path.join(a.out, "by_level.csv"))
    P0.plot(df, clocks_1d, results, os.path.join(a.out, "clock_fits.png"))

    pd.set_option("display.width", 250)
    print("== section / member (check Mn/L against the simulated F_max)")
    print(st.round(4).to_string())
    print("\n== mean per drift level (>= 3 %)")
    print(lvl.unstack("level").to_string())
    cols_show = ["clock", "model", "loco_rmse", "CTR1_bias_hiDI", "CTR1_bias_6pct", "CTR1_bias_8pct",
                 "R307_bias_hiDI", "R307_bias_6pct", "R307_bias_8pct", "select_score", "pass_hiDI", "max_abs_bias", "strict_pass"]
    print("\n== clocks (LOCO-best model each), sorted by select_score (LOCO + R307 only)")
    print(tab[cols_show].round(3).to_string(index=False))
    print("\n== strict check, any model (|bias| < 0.1 on CTR1 & R307 x hiDI/6%/8%)")
    print(strict.head(12).round(3).to_string(index=False))
    json.dump({"eps_sp": EPS_SP, "pass_tol": PASS_TOL, "primary_clock": "eps_max"}, open(os.path.join(a.out, "settings.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
