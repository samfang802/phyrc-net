#!/usr/bin/env python3
"""Can a mechanics write gate separate real memory writes from self-confirmed ones? (docs/redesign/phase1_ilf.md §12)

No training. Per column, the fiber section of phase1_fiber_clock.py turns the simulated drift history into a
curvature profile along the height z (mm from the base) at every loop peak:
  M(z)       = M_b (1 - z / L)                        cantilever moment, M_b = M_env(phi_b)
  phi_el(z)  = section M-phi inverted at M(z)          (rising branch)
  phi_pl(z)  = phi_b - (phi_b - phi_y) z / L_p         z < L_p, phi_b > phi_y (Priestley plastic hinge)
  phi_t(z)   = max(phi_el, phi_pl),  phi_b = Member.phi_base(drift)
History variable (Kuhn-Tucker / phase-field history field): kappa_t(z) = max_{s <= t} phi_s(z). Damage can grow at
frame t only where kappa grows and the section is cracked:
  gate_t(z) = [kappa_t(z) > (1 + delta) kappa_{t-1}(z)] and [kappa_t(z) > phi_cr]
phi_cr: extreme tension fiber at f_r = 0.62 sqrt(f'c). Variants of the history:
  env    kappa over |drift| at every peak (both signs)
  pos    kappa over the + peaks only (the face is in tension at + drift); a frame at a - peak gets no write
  neg    same for - peaks
  face_a pos on the Push walk, neg on the Pull walk (each face in tension on one sign; the mapping is unverified)
  face_b the other mapping
  fname  the existing filename Kaiser gate (frame suffix 0-2), every row (reference)
Rows -> z: z = (base_row - r + 0.5) * mm_per_px; rows below the base (footing) are outside every gate. The scale
is an assumption (10 cm grid -> 1 px/mm at 750 x 800, dataset_audit.md), so it is swept with base_row.

The rule memory writes a pixel once it is dark in most of the last 7 frames (temporal median), so a structure born
at frame t is written ~3-4 frames later; --window w also scores the gate opened on any of the last w frames.

Measured per gate, on the frames t >= 1 of each walk (k trajectories averaged):
  area        share of (frame, pixel) inside the gate (what a random gate of that size would keep)
  real_born   share of the real lag-1 born dark pixels (irreversibility_curve.dark_masks) inside
  real_persist  same, born pixels still dark (within 1 px) 3 and 6 frames later (no flicker / crack opening)
  real_write  share of the real-frame memory's new entries (MemoryTrack, the training memory) inside
  self_true   share of the self memory's new entries that match real structure (within tol of the real memory
              or the real frame's dark pixels at that frame) inside
  self_false  share of the self memory's other new entries inside
Worth building only if real_write >> area and self_false << real_write.

v2 (the v1 measures above are kept but are flawed, docs/redesign/phase1_ilf.md §12.1): frames alternate load sign,
so a lag-1 "born" pixel is mostly a crack reopening on the tension peak (real: 10x the first-ever appearances);
the memory's write time is the 7-frame median's delay, not the birth; and "false" = no real structure within
tol measures position mismatch to one real draw (grid phase, a different but plausible crack), not support.
  step births  dark at t, present (within 1 px) at t+2 and t+4, absent at t-2, t-4 and t-6: same-sign frames only,
               so opening / closing and one-frame flicker do not count; frames 6 .. T-5 only
  area_v       gate share on those frames
  real_step / self_step   share of the real / generated step births inside the gate (timing, no position match)
  sign test    step births per frame on + vs - peaks, per walk (a face should crack when it is in tension)

  python scripts/eval/phase1_physics_gate.py --split test \
      --npz results/eval_suite/v12memc/test/samples/v12memc_006000_grav.npz --out results/phase1_physics_gate/test.json
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.model.cfm.refiner_memory import MemoryTrack, kaiser_gate  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", "eval", fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FC = _load("fiberclock", "phase1_fiber_clock.py")
IR = _load("irrev", "irreversibility_curve.py")
SEG = FC.SEG


class Column:
    """Curvature history of one column on a height grid (mm)."""

    def __init__(self, exp, row, z, delta):
        sec = FC.Section(row).moment_curvature()
        self.mem = mem = FC.Member(sec, float(row["lMeasured"]))
        self.sec, self.L, self.z, self.delta = sec, mem.L, z, delta
        e_bot = sec.e_top - sec.phi * sec.h                      # extreme tension fiber (< 0 in tension)
        eps_cr = 0.62 * np.sqrt(sec.fc) / sec.Ec
        self.phi_cr = float(np.interp(eps_cr, -e_bot, sec.phi))
        i = int(np.argmax(sec.M_env)) + 1
        m, k = np.unique(sec.M_env[:i], return_index=True)
        self.M_rise, self.phi_rise = m, sec.phi[:i][k]
        sim = pd.read_csv(os.path.join(ROOT, "Data/Hysteresis/Simulate", f"{exp}.csv"))
        self.drift = sim["Drift(%)"].to_numpy(float)

    def profile(self, drift_abs):
        """phi(z) at a peak of |drift| (%)."""
        s, mem, z = self.sec, self.mem, np.maximum(self.z, 0.0)
        phi_b = float(mem.phi_base(drift_abs))
        M_b = float(np.interp(min(phi_b, s.phi[-1]), s.phi, s.M_env))
        phi_el = np.interp(M_b * np.clip(1 - z / self.L, 0, 1), self.M_rise, self.phi_rise)
        phi_pl = np.where((z < mem.Lp) & (phi_b > s.phi_y), phi_b - (phi_b - s.phi_y) * z / mem.Lp, 0.0)
        return np.maximum(phi_el, phi_pl)

    def gates(self, timesteps, mode):
        """(T, Z) bool for frames at these timesteps (sorted), history over every peak up to each frame."""
        k_max = int(max(timesteps)) // SEG
        peaks = self.drift[np.arange(k_max + 1) * SEG]
        kappa = np.zeros_like(self.z, dtype=float)
        per_peak = []
        for d in peaks:
            prev = kappa.copy()
            if mode == "env" or (mode == "pos" and d > 0) or (mode == "neg" and d < 0):
                kappa = np.maximum(kappa, self.profile(abs(d)))
            per_peak.append((kappa > (1 + self.delta) * prev) & (kappa > self.phi_cr))
        return np.stack([per_peak[int(t) // SEG] for t in timesteps]) & (self.z >= 0)[None]

    def band_top(self, g):
        """Highest z (mm) inside a gate row vector, 0 if empty."""
        return float(self.z[g].max()) if g.any() else 0.0


def memory_writes(frames_u8):
    """(T, H, W) uint8 -> (T, H, W) bool new entries of the rule memory at each frame, and the held entries."""
    x = torch.from_numpy(frames_u8.astype(np.float32) / 127.5 - 1.0)[:, None]
    tr = MemoryTrack()
    held_prev = torch.zeros_like(x[0], dtype=torch.bool)
    new, held = [], []
    for t in range(len(x)):
        h = tr.update(x[t:t + 1])[0] < -tr.eps
        new.append((h & ~held_prev)[0].numpy())
        held.append(h[0].numpy())
        held_prev = h
    return np.stack(new), np.stack(held)


def dilate(m, tol):
    t = torch.from_numpy(m.astype(np.float32))[:, None]
    return (F.max_pool2d(t, 2 * tol + 1, stride=1, padding=tol)[:, 0] > 0).numpy()


def step_births(dark):
    """(T, H, W) bool -> (T, H, W) bool births that persist on the next same-sign frames, (T,) valid frames."""
    pres = dilate(dark, 1)
    T = len(dark)
    out = np.zeros_like(dark)
    valid = np.zeros(T, dtype=bool)
    for t in range(6, T - 4):
        out[t] = dark[t] & pres[t + 2] & pres[t + 4] & ~(pres[t - 2] | pres[t - 4] | pres[t - 6])
        valid[t] = True
    return out, valid


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True, help="samples npz (real + fake [N, K, H, W]) of a memory refiner")
    p.add_argument("--split", default="test")
    p.add_argument("--label", default="Data/DamagePattern/label.csv")
    p.add_argument("--info", default="Data/Hysteresis/info_w_mander_big.csv")
    p.add_argument("--mm_per_px", type=float, nargs="+", default=[5.5, 6.25, 7.0])
    p.add_argument("--base_row", type=float, nargs="+", default=[127, 117, 106], help="row of the column base")
    p.add_argument("--delta", type=float, default=0.02, help="relative kappa growth that counts as a write")
    p.add_argument("--thr", type=float, default=0.05, help="dark threshold for real born (irreversibility_curve)")
    p.add_argument("--tol", type=int, default=2, help="px tolerance for a self entry to count as real")
    p.add_argument("--window", type=int, nargs="+", default=[1, 4, 7], help="gate open on any of the last w frames")
    p.add_argument("--k", type=int, default=0, help="trajectories to use (0 = all)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    z_npz = np.load(a.npz, allow_pickle=True)
    real, fake, fn = z_npz["real"], z_npz["fake"], list(z_npz["filename"])
    K = fake.shape[1] if a.k <= 0 else min(a.k, fake.shape[1])
    H = real.shape[1]
    label = pd.read_csv(a.label).set_index("filename").loc[fn].reset_index()
    info = pd.read_csv(a.info, dtype={"id": str}).set_index("id")
    modes = ("env", "pos", "neg", "face_a", "face_b", "fname")

    # per walk: rows, real born / writes, self writes split into true / false
    walks = []
    for (exp, direction), g in label.groupby(["exp", "direction"], sort=True):
        g = g.sort_values("timestep", kind="stable")
        rows = [fn.index(f) for f in g.filename]
        r_frames = real[rows]
        dark = IR.dark_masks(r_frames, a.thr, 0.5, 2.0)
        born = np.zeros_like(dark)
        born[1:] = dark[1:] & ~dilate(dark[:-1], 1)
        dd = dilate(dark, 1)
        persist = np.zeros_like(born)
        persist[1:-6] = born[1:-6] & dd[4:-3] & dd[7:]
        r_new, r_held = memory_writes(r_frames)
        r_step, valid = step_births(dark)
        s_step = [step_births(IR.dark_masks(fake[rows, k], a.thr, 0.5, 2.0))[0] for k in range(K)]
        real_ok = dilate(r_held | dark, a.tol)
        s_true, s_false = [], []
        for k in range(K):
            s_new, _ = memory_writes(fake[rows, k])
            s_true.append(s_new & real_ok)
            s_false.append(s_new & ~real_ok)
        walks.append(dict(exp=exp, direction=direction, ts=g.timestep.to_numpy(), files=g.filename.tolist(),
                          born=born, persist=persist, r_step=r_step, s_step=s_step, valid=valid, r_new=r_new, s_true=s_true, s_false=s_false))
        print(f"[gate] {exp}/{direction}: {len(rows)} frames, real writes {int(r_new[1:].sum())}, "
              f"self writes true {np.mean([s[1:].sum() for s in s_true]):.0f} / false "
              f"{np.mean([s[1:].sum() for s in s_false]):.0f} per trajectory", flush=True)

    report = {"npz": os.path.abspath(a.npz), "split": a.split, "delta": a.delta, "thr": a.thr, "tol": a.tol, "K": K,
              "results": [], "columns": {}}
    for mm in a.mm_per_px:
        for base in a.base_row:
            zrow = (base - np.arange(H) + 0.5) * mm                  # (H,) mm from the base, top row first
            cols = {e: Column(e, info.loc[e], zrow, a.delta) for e in {w["exp"] for w in walks}}
            for mode, win in [(m, w) for m in modes for w in a.window]:
                acc = {k: np.zeros(2) for k in ("area", "real_born", "real_persist", "real_write", "self_true",
                                                "self_false", "area_v", "real_step", "self_step")}
                tops = []
                for w in walks:
                    col = cols[w["exp"]]
                    if mode == "fname":
                        g = np.array([kaiser_gate(f) for f in w["files"]])[:, None] & (zrow >= 0)[None]
                    elif mode in ("face_a", "face_b"):
                        push = str(w["direction"]).lower() == "push"
                        g = col.gates(w["ts"], "pos" if push == (mode == "face_a") else "neg")
                    else:
                        g = col.gates(w["ts"], mode)
                    if win > 1:  # open if it was open on any of the last win frames
                        g = np.stack([g[max(0, t - win + 1):t + 1].any(0) for t in range(len(g))])
                    tops += [col.band_top(r) for r in g[1:]]
                    G = np.broadcast_to(g[:, :, None], w["born"].shape)[1:]  # (T-1, H, W)
                    acc["area"] += [G.sum(), G.size]
                    V = w["valid"][1:]
                    acc["area_v"] += [G[V].sum(), G[V].size]
                    acc["real_step"] += [(w["r_step"][1:] & G).sum(), w["r_step"][1:].sum()]
                    for m in w["s_step"]:
                        acc["self_step"] += [(m[1:] & G).sum(), m[1:].sum()]
                    for key, m in (("real_born", w["born"]), ("real_persist", w["persist"]), ("real_write", w["r_new"])):
                        acc[key] += [(m[1:] & G).sum(), m[1:].sum()]
                    for key, src in (("self_true", "s_true"), ("self_false", "s_false")):
                        for m in w[src]:
                            acc[key] += [(m[1:] & G).sum(), m[1:].sum()]
                res = {k: float(v[0] / max(v[1], 1)) for k, v in acc.items()}
                res.update(mode=mode, window=win, mm_per_px=mm, base_row=base, n_self_false=float(acc["self_false"][1] / K),
                           n_self_true=float(acc["self_true"][1] / K), n_real_write=float(acc["real_write"][1]),
                           band_top_mm_median=float(np.median(tops)), band_top_mm_max=float(np.max(tops)),
                           image_top_mm=float(zrow[0]))
                report["results"].append(res)
            if mm == a.mm_per_px[0] and base == a.base_row[0]:
                for e, c in cols.items():
                    report["columns"][e] = dict(L_mm=c.L, Lp_mm=c.mem.Lp, phi_cr=c.phi_cr, phi_y=c.sec.phi_y,
                                                phi_cr_over_phi_y=c.phi_cr / c.sec.phi_y)

    # per-frame counts (where in the walk the writes happen), gates at the base section
    report["per_frame"] = []
    for w in walks:
        col = Column(w["exp"], info.loc[w["exp"]], np.array([0.0]), a.delta)
        report["per_frame"].append(dict(
            exp=w["exp"], direction=w["direction"], files=w["files"],
            real_persist=w["persist"].sum((1, 2)).tolist(), real_write=w["r_new"].sum((1, 2)).tolist(),
            self_true=np.mean([m.sum((1, 2)) for m in w["s_true"]], 0).tolist(),
            self_false=np.mean([m.sum((1, 2)) for m in w["s_false"]], 0).tolist(),
            **{f"gate_{m}": col.gates(w["ts"], m)[:, 0].astype(int).tolist() for m in ("env", "pos", "neg")}))

    # sign test on the step births: per walk, births per frame on + vs - peaks
    report["sign_test"] = []
    for w in walks:
        col = Column(w["exp"], info.loc[w["exp"]], np.array([0.0]), a.delta)
        sg = np.sign(col.drift[w["ts"].astype(int)])
        row = dict(exp=w["exp"], direction=w["direction"])
        for name, m in (("real", w["r_step"].sum((1, 2))), ("self", np.mean([x.sum((1, 2)) for x in w["s_step"]], 0))):
            for sgn, lab in ((1, "pos"), (-1, "neg")):
                sel = w["valid"] & (sg == sgn)
                row[f"{name}_{lab}"] = float(m[sel].mean()) if sel.any() else 0.0
        report["sign_test"].append(row)

    # temporal only: real born per frame on frames where the |drift| envelope grows vs not (Felicity check)
    fel = {"new": np.zeros(2), "repeat": np.zeros(2)}
    for w in walks:
        col = Column(w["exp"], info.loc[w["exp"]], np.array([0.0]), a.delta)   # z = 0: the base section
        g = col.gates(w["ts"], "env")[:, 0]
        for t in range(1, len(g)):
            key = "new" if g[t] else "repeat"
            fel[key] += [w["born"][t].sum(), 1]
    report["born_per_frame"] = {k: float(v[0] / max(v[1], 1)) for k, v in fel.items()}
    report["frames"] = {k: int(v[1]) for k, v in fel.items()}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\n== {a.split}: columns {json.dumps(report['columns'], indent=None)}")
    print(f"real born per frame: envelope grows {report['born_per_frame']['new']:.0f} "
          f"({report['frames']['new']} frames) vs not {report['born_per_frame']['repeat']:.0f} "
          f"({report['frames']['repeat']} frames)")
    for r in report["sign_test"]:
        print(f"sign test {r['exp']}/{r['direction']}: step births per frame real +{r['real_pos']:.1f} / -{r['real_neg']:.1f}, "
              f"self +{r['self_pos']:.1f} / -{r['self_neg']:.1f}")
    print("v2 (step births, w = 1):")
    for r in report["results"]:
        if r["window"] == 1:
            print(f"  {r['mode']:6s} {r['mm_per_px']:5.2f} {r['base_row']:4.0f} | area {r['area_v']:.3f} real {r['real_step']:.3f} "
                  f"(lift {r['real_step'] / max(r['area_v'], 1e-9):.2f}) self {r['self_step']:.3f} "
                  f"(lift {r['self_step'] / max(r['area_v'], 1e-9):.2f})")
    print("v1 (flawed, kept for reference):")
    print(f"{'mode':6s} {'win':>3s} {'mm/px':>5s} {'base':>4s} | {'area':>5s} {'r_born':>6s} {'r_pers':>6s} "
          f"{'r_write':>7s} {'s_true':>6s} "
          f"{'s_false':>7s} | band top median / max (mm), image top")
    for r in report["results"]:
        print(f"{r['mode']:6s} {r['window']:3d} {r['mm_per_px']:5.2f} {r['base_row']:4.0f} | {r['area']:.3f} "
              f"{r['real_born']:6.3f} {r['real_persist']:6.3f} "
              f"{r['real_write']:7.3f} {r['self_true']:6.3f} {r['self_false']:7.3f} | "
              f"{r['band_top_mm_median']:.0f} / {r['band_top_mm_max']:.0f}, {r['image_top_mm']:.0f}")


if __name__ == "__main__":
    main()
