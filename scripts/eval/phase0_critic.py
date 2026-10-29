#!/usr/bin/env python3
"""Phase-0 check for OPA-Flow (docs/redesign/survey_DE_realism_condition.md, O1 day 1): can a pixel CNN
critic R(x_a, x_b) → (logit "b is later than a", progress gap m̂) work on columns it has not seen?

No DINO anywhere (the eval metrics use DINOv3; a DINO critic would be circular). The critic is a
small CNN trained from scratch on same-column, same-face real pairs. Each image of a pair gets its
own strong brightness / contrast / gamma / noise / small affine augmentation, so tone cannot tell
the order. m̂ regresses a column-independent progress gap from the damage-state encoder
(src/data/damage_condition.py, run A_info): P = (−strength_ratio + cum_energy) / 2 in the encoder's
standardized target units, Δenc(a, b) = P_b − P_a.

Architectures (--archs):
  pair      early fusion: CNN([x_a, x_b, x_b − x_a]); outputs made antisymmetric,
            out(a, b) = (f(a, b) − f(b, a)) / 2.
  siamese   per-image scores: logit = s_o(b) − s_o(a), m̂ = s_m(b) − s_m(a).

Protocol (split id_list_val_R307.json): leave-one-column-out over the 4 train columns
(R315, R330, R615, R1015), plus models trained on all 4 evaluated on R307 (val) and CTR1 (test).
Metrics per held-out column:
  (a) order accuracy on pairs ≥ 1 drift level apart (also ≥ 2 levels, same level);
  (b) Spearman(m̂, Δenc) on pairs ≥ 1 level apart (also Spearman with Δlevel for scale);
  (c) on generated samples (eval-suite npz): Σ m̂(gen_i, gen_j) / Σ m̂(real_i, real_j) over the same
      pairs (and a curve version: each level against the lowest level), jackknife over cycle index.
Also: encoder progress P per drift level for every image column (is CTR1 > R315 at the same drift?).

Outputs: <out_dir>/metrics.json, report.txt, runs/<arch>_<fold>_s<seed>.pt.

  python scripts/eval/phase0_critic.py --out_dir results/phase0/critic \
      --samples gn=results/eval_suite/v105/test/samples/gn.npz gn_shared=results/eval_suite/v105/test/samples/gn_shared.npz
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics.dino_mmd import load_real_uint8  # noqa: E402  (preprocessing only, no DINO)

TRAIN_COLS = ("R315", "R330", "R615", "R1015")
ENCODER_RUN = "results/DamageStateEncoder/A_info/2026-10-23_22-34-47"


# ----------------------------------------------------------------------------------------------- data


def load_frames(data_path, label_csv, image_dir, encoder_run):
    """All labelled frames of the split file: uint8 images + per-frame metadata + encoder progress."""
    from src.data.damage_condition import build_frame_conditions

    with open(data_path) as f:
        split = json.load(f)
    lab = pd.read_csv(label_csv)
    split_of = {n: s for s, names in split.items() for n in names}
    lab = lab[lab["filename"].isin(split_of)].reset_index(drop=True)
    parts = lab["filename"].str[:-4].str.split("_")
    lab["col"] = parts.str[0]
    lab["face"] = parts.str[1]
    lab["drift"] = parts.str[2].astype(int) / 1000.0  # 00250 → 0.25 %
    lab["cycle"] = parts.str[3].astype(int)
    lab["split"] = lab["filename"].map(split_of)
    levels = sorted(lab["drift"].unique())
    lab["level"] = lab["drift"].map({d: i for i, d in enumerate(levels)})
    lab["seq"] = lab["col"] + "_" + lab["face"]

    cond = build_frame_conditions(encoder_run, sorted(lab["exp"].unique()), root=os.path.join(_ROOT, "Data", "Hysteresis"))
    sc = np.stack([cond[e][int(t), -4:].numpy() for e, t in zip(lab["exp"], lab["timestep"])])
    lab["enc_strength"], lab["enc_stiffness"], lab["enc_energy"], lab["enc_logdrift"] = sc.T
    lab["P"] = 0.5 * (-lab["enc_strength"] + lab["enc_energy"])
    imgs = load_real_uint8(lab["filename"].tolist(), image_dir)
    return lab, imgs


def pairs_of(lab, rows=None, min_gap=None):
    """Ordered pairs (i earlier, j later) within each sequence. Returns int arrays i, j."""
    sub = lab if rows is None else lab.loc[rows]
    I, J = [], []
    for _, g in sub.groupby("seq"):
        idx = g.sort_values("timestep").index.to_numpy()
        a, b = np.triu_indices(len(idx), k=1)
        I.append(idx[a])
        J.append(idx[b])
    I, J = np.concatenate(I), np.concatenate(J)
    if min_gap is not None:
        keep = (lab["level"].to_numpy()[J] - lab["level"].to_numpy()[I]) >= min_gap
        I, J = I[keep], J[keep]
    return I, J


# ---------------------------------------------------------------------------------------------- model


def _block(cin, cout, stride):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
    )


class Trunk(nn.Module):
    def __init__(self, cin, width=32, nout=2):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            _block(cin, w, 1), _block(w, w, 2), _block(w, 2 * w, 2), _block(2 * w, 4 * w, 2),
            _block(4 * w, 4 * w, 2), _block(4 * w, 8 * w, 2),
        )
        self.head = nn.Sequential(nn.Linear(8 * w, 128), nn.SiLU(), nn.Dropout(0.2), nn.Linear(128, nout))

    def forward(self, x):
        return self.head(self.net(x).mean(dim=(2, 3)))


class Critic(nn.Module):
    """forward(a, b) → (logit that b is later, m̂ = progress(b) − progress(a)); a, b in [-1, 1], (B,1,H,W)."""

    def __init__(self, arch="pair", width=32):
        super().__init__()
        self.arch = arch
        self.trunk = Trunk(3 if arch == "pair" else 1, width, 2)

    def forward(self, a, b):
        if self.arch == "pair":
            f_ab = self.trunk(torch.cat([a, b, b - a], 1))
            f_ba = self.trunk(torch.cat([b, a, a - b], 1))
            out = 0.5 * (f_ab - f_ba)
        else:
            s = self.trunk(torch.cat([a, b], 0))
            out = s[a.shape[0]:] - s[: a.shape[0]]
        return out[:, 0], out[:, 1]


# --------------------------------------------------------------------------------------- augmentation


def photometric(x, gen):
    """Independent per-image tone change. x in [0,1], (B,1,H,W)."""
    B = x.shape[0]

    def U(lo, hi):
        return lo + (hi - lo) * torch.rand(B, 1, 1, 1, device=x.device, generator=gen)

    gamma = torch.exp(U(math.log(0.6), math.log(1.7)))
    x = x.clamp(1e-4, 1.0) ** gamma
    m = x.mean(dim=(2, 3), keepdim=True)
    x = (x - m) * U(0.55, 1.6) + m + U(-0.25, 0.25)
    x = x + U(0.0, 0.04) * torch.randn(x.shape, device=x.device, generator=gen)
    return x.clamp(0.0, 1.0)


def geometric(x, gen, max_deg=3.0, max_shift=0.06, scale=(0.94, 1.06)):
    B = x.shape[0]
    r = lambda: torch.rand(B, device=x.device, generator=gen)  # noqa: E731
    th = (2 * r() - 1) * max_deg * math.pi / 180
    s = scale[0] + (scale[1] - scale[0]) * r()
    tx, ty = (2 * r() - 1) * max_shift, (2 * r() - 1) * max_shift
    A = torch.stack([torch.stack([s * torch.cos(th), -s * torch.sin(th), tx], 1),
                     torch.stack([s * torch.sin(th), s * torch.cos(th), ty], 1)], 1)
    grid = F.affine_grid(A, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def to_model(u8_t):
    return u8_t.float().unsqueeze(1) / 255.0


# -------------------------------------------------------------------------------------------- training


def train_critic(X, lab, train_rows, arch, seed, steps, batch, lr, lam, device, log_every=500):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Critic(arch).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    seqs = [g.sort_values("timestep").index.to_numpy() for _, g in lab.loc[train_rows].groupby("seq")]
    T = torch.as_tensor(lab["timestep"].to_numpy(), device=device)
    P = torch.as_tensor(lab["P"].to_numpy(), dtype=torch.float32, device=device)
    net.train()
    t0 = time.time()
    for step in range(steps):
        s_idx = rng.integers(len(seqs), size=batch)
        a = np.empty(batch, dtype=np.int64)
        b = np.empty(batch, dtype=np.int64)
        for n, s in enumerate(s_idx):
            a[n], b[n] = rng.choice(seqs[s], size=2, replace=False)
        a_t, b_t = torch.as_tensor(a, device=device), torch.as_tensor(b, device=device)
        xa, xb = to_model(X[a_t]), to_model(X[b_t])
        flip = torch.rand(batch, device=device, generator=gen) < 0.5  # same flip for both images of a pair
        xa = torch.where(flip[:, None, None, None], xa.flip(-1), xa)
        xb = torch.where(flip[:, None, None, None], xb.flip(-1), xb)
        xa, xb = geometric(photometric(xa, gen), gen), geometric(photometric(xb, gen), gen)
        logit, m = net(xa * 2 - 1, xb * 2 - 1)
        y = (T[b_t] > T[a_t]).float()
        loss = F.binary_cross_entropy_with_logits(logit, y) + lam * F.smooth_l1_loss(m, P[b_t] - P[a_t], beta=0.5)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % log_every == 0:
            acc = ((logit > 0).float() == y).float().mean().item()
            print(f"    step {step + 1}/{steps} loss {loss.item():.4f} batch-acc {acc:.3f} ({time.time() - t0:.0f} s)", flush=True)
    net.eval()
    return net


@torch.no_grad()
def score_pairs(net, XA, XB, ia, ib, device, bs=512):
    """Critic on pairs (XA[ia], XB[ib]); XA/XB are uint8 tensors (N,H,W) on device."""
    L, M = [], []
    for s in range(0, len(ia), bs):
        a = to_model(XA[torch.as_tensor(ia[s:s + bs], device=device)]) * 2 - 1
        b = to_model(XB[torch.as_tensor(ib[s:s + bs], device=device)]) * 2 - 1
        lo, m = net(a, b)
        L.append(lo.float().cpu())
        M.append(m.float().cpu())
    return torch.cat(L).numpy(), torch.cat(M).numpy()


def eval_column(net, X, lab, col, device):
    rows = lab.index[lab["col"] == col]
    I, J = pairs_of(lab, rows)
    logit, m = score_pairs(net, X, X, I, J, device)
    gap = lab["level"].to_numpy()[J] - lab["level"].to_numpy()[I]
    d_enc = lab["P"].to_numpy()[J] - lab["P"].to_numpy()[I]
    g1 = gap >= 1
    out = {
        "n_pairs_gap1": int(g1.sum()),
        "acc_gap1": float((logit[g1] > 0).mean()),
        "acc_gap2": float((logit[gap >= 2] > 0).mean()),
        "acc_same_level": float((logit[gap == 0] > 0).mean()) if (gap == 0).any() else float("nan"),
        "spearman_m_enc_gap1": float(spearmanr(m[g1], d_enc[g1])[0]),
        "spearman_m_enc_all": float(spearmanr(m, d_enc)[0]),
        "spearman_m_energy_gap1": float(spearmanr(m[g1], (lab["enc_energy"].to_numpy()[J] - lab["enc_energy"].to_numpy()[I])[g1])[0]),
        "spearman_m_strdrop_gap1": float(spearmanr(m[g1], (lab["enc_strength"].to_numpy()[I] - lab["enc_strength"].to_numpy()[J])[g1])[0]),
        "spearman_m_level_gap1": float(spearmanr(m[g1], gap[g1])[0]),
        "spearman_enc_level_gap1": float(spearmanr(d_enc[g1], gap[g1])[0]),
        "mean_m_gap1": float(m[g1].mean()),
        "mean_enc_gap1": float(d_enc[g1].mean()),
    }
    # Same image twice (independent tone augmentation): m̂ and logit should be ≈ 0.
    gen = torch.Generator(device=device).manual_seed(123)
    xr = to_model(X[torch.as_tensor(rows.to_numpy(), device=device)])
    with torch.no_grad():
        lo, mm = net(photometric(xr, gen) * 2 - 1, photometric(xr, gen) * 2 - 1)
    out["same_image_abs_logit"] = float(lo.abs().mean())
    out["same_image_abs_m"] = float(mm.abs().mean())
    return out


# ------------------------------------------------------------------------------- generated-sample ratio


def _jackknife(fn, cycles):
    """fn(mask over pairs-or-frames by cycle) → value; delete-one-cycle jackknife SE."""
    full = fn(None)
    units = sorted(set(cycles))
    vals = np.array([fn(u) for u in units])
    n = len(units)
    se = math.sqrt((n - 1) / n * np.sum((vals - vals.mean()) ** 2))
    return full, se


@torch.no_grad()
def gen_ratio(nets, npz_path, lab, device):
    """Σ m̂(gen pair) / Σ m̂(real pair) on the npz's frames, averaged over critics."""
    d = np.load(npz_path, allow_pickle=False)
    fake, real, names = d["fake"], d["real"], [str(s) for s in d["filename"]]
    sub = lab.set_index("filename").loc[names].reset_index()
    K = fake.shape[1]
    I, J = pairs_of(sub, min_gap=1)
    cyc_i, cyc_j = sub["cycle"].to_numpy()[I], sub["cycle"].to_numpy()[J]
    lvl = sub["level"].to_numpy()
    Rt = torch.as_tensor(real, device=device)
    res = {"n_pairs": int(len(I)), "K": int(K), "per_critic": []}
    acc = {"gen": [], "anchor": [], "real": [], "curve_gen": [], "curve_real": []}
    for net in nets:
        _, m_real = score_pairs(net, Rt, Rt, I, J, device)
        m_gen, m_anc, lo_gen = [], [], []
        for k in range(K):
            Ft = torch.as_tensor(np.ascontiguousarray(fake[:, k]), device=device)
            lo, m = score_pairs(net, Ft, Ft, I, J, device)
            m_gen.append(m)
            lo_gen.append(lo)
            m_anc.append(score_pairs(net, Rt, Ft, I, J, device)[1])
        m_gen, m_anc, lo_gen = np.mean(m_gen, 0), np.mean(m_anc, 0), np.stack(lo_gen)
        acc["gen"].append(m_gen)
        acc["anchor"].append(m_anc)
        acc["real"].append(m_real)
        res["per_critic"].append({"ratio_gen": float(m_gen.sum() / m_real.sum()),
                                  "ratio_anchor": float(m_anc.sum() / m_real.sum()),
                                  "order_acc_gen": float((lo_gen > 0).mean())})
    mg, ma, mr = (np.mean(acc[k], 0) for k in ("gen", "anchor", "real"))

    def ratio(num, cyc_drop):
        keep = np.ones(len(I), bool) if cyc_drop is None else (cyc_i != cyc_drop) & (cyc_j != cyc_drop)
        return num[keep].sum() / mr[keep].sum()

    # Curve version: every frame against the frames of the lowest level of its sequence.
    first = np.array([lvl[i] == lvl[sub.index[sub["seq"] == sub["seq"][i]]].min() for i in I])
    cur_keep = first

    def curve_ratio(num, cyc_drop):
        keep = cur_keep.copy()
        if cyc_drop is not None:
            keep &= (cyc_i != cyc_drop) & (cyc_j != cyc_drop)
        num_l, den_l = [], []
        for (s, L) in sorted(set(zip(sub["seq"].to_numpy()[J][keep], lvl[J][keep]))):
            sel = keep & (sub["seq"].to_numpy()[J] == s) & (lvl[J] == L)
            num_l.append(num[sel].mean())
            den_l.append(mr[sel].mean())
        return float(np.sum(num_l) / np.sum(den_l))

    cyc_all = list(cyc_i) + list(cyc_j)
    for key, num in (("gen", mg), ("anchor", ma)):
        r, se = _jackknife(lambda c: ratio(num, c), cyc_all)
        res[f"ratio_{key}"], res[f"ratio_{key}_se"] = float(r), float(se)
        r, se = _jackknife(lambda c: curve_ratio(num, c), cyc_all)
        res[f"curve_ratio_{key}"], res[f"curve_ratio_{key}_se"] = float(r), float(se)
    res["spearman_gen_real"] = float(spearmanr(mg, mr)[0])
    res["mean_m_real"], res["mean_m_gen"] = float(mr.mean()), float(mg.mean())
    return res


@torch.no_grad()
def other_column_ratio(nets, X, lab, col, device):
    """Reference row: curve of each other real column (its own sequences) ÷ curve of `col`, on shared levels."""

    def curve(c):
        rows = lab.index[lab["col"] == c]
        I, J = pairs_of(lab, rows)
        lvl = lab["level"].to_numpy()
        lo_min = {s: lab.loc[lab["seq"] == s, "level"].min() for s in lab.loc[rows, "seq"].unique()}
        keep = np.array([lvl[i] == lo_min[lab["seq"][i]] for i in I]) & (lvl[J] > lvl[I])
        I, J = I[keep], J[keep]
        m = np.mean([score_pairs(n, X, X, I, J, device)[1] for n in nets], 0)
        df = pd.DataFrame({"drift": lab["drift"].to_numpy()[J], "m": m})
        return df.groupby("drift")["m"].mean()

    ref = curve(col)
    out = {}
    for c in sorted(lab["col"].unique()):
        if c == col:
            continue
        oc = curve(c)
        common = ref.index.intersection(oc.index)
        out[c] = float(oc[common].sum() / ref[common].sum())
    out["mean"] = float(np.mean(list(out.values())))
    return out


def encoder_table(lab):
    """Encoder progress per (column, drift level): last frame of each level, both faces averaged."""
    last = lab.sort_values("timestep").groupby(["col", "face", "drift"]).tail(1)
    t = last.groupby(["col", "drift"])[["P", "enc_strength", "enc_energy"]].mean()
    return t


# ------------------------------------------------------------------------------------------------ main


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out_dir", default="results/phase0/critic")
    p.add_argument("--data_path", default="Data/DamagePattern/id_list_val_R307.json")
    p.add_argument("--label_csv", default="Data/DamagePattern/label.csv")
    p.add_argument("--image_dir", default="Data/DamagePattern/images")
    p.add_argument("--encoder_run", default=ENCODER_RUN)
    p.add_argument("--archs", nargs="*", default=["pair", "siamese"])
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2], help="seeds of the all-4-columns models")
    p.add_argument("--loco_seeds", nargs="*", type=int, default=[0])
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=1.0, help="weight of the m̂ Huber loss")
    p.add_argument("--samples", nargs="*", default=[], metavar="NAME=NPZ", help="generated samples for (c)")
    p.add_argument("--eval_only", action="store_true", help="reuse runs/*.pt")
    cli = p.parse_args()
    os.chdir(_ROOT)
    os.makedirs(os.path.join(cli.out_dir, "runs"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lab, imgs = load_frames(cli.data_path, cli.label_csv, cli.image_dir, cli.encoder_run)
    X = torch.as_tensor(imgs, device=device)
    print(f"[critic] {len(lab)} frames, columns {sorted(lab['col'].unique())}, device {device}", flush=True)
    results = {"config": vars(cli), "loco": {}, "full": {}, "samples": {}}

    # Preprocessing check against the eval npz's real (same transform as the generator).
    for name, path in (kv.split("=", 1) for kv in cli.samples):
        d = np.load(path, allow_pickle=False)
        rows = lab.set_index("filename").index.get_indexer([str(s) for s in d["filename"]])
        diff = np.abs(imgs[rows].astype(int) - d["real"].astype(int))
        results.setdefault("preproc_check", {})[name] = {"mean_abs_diff": float(diff.mean()), "max": int(diff.max())}
        print(f"[critic] preprocessing vs {name} real: mean |diff| {diff.mean():.3f}, max {diff.max()}", flush=True)

    def get(arch, fold, seed, train_cols):
        path = os.path.join(cli.out_dir, "runs", f"{arch}_{fold}_s{seed}.pt")
        net = Critic(arch).to(device)
        if cli.eval_only or os.path.exists(path):
            net.load_state_dict(torch.load(path, map_location=device))
            net.eval()
            return net
        print(f"[critic] train {arch} fold={fold} seed={seed} on {train_cols}", flush=True)
        rows = lab.index[lab["col"].isin(train_cols)]
        net = train_critic(X, lab, rows, arch, seed, cli.steps, cli.batch, cli.lr, cli.lam, device)
        torch.save(net.state_dict(), path)
        return net

    for arch in cli.archs:
        results["loco"][arch] = {}
        for held in TRAIN_COLS:
            tr = [c for c in TRAIN_COLS if c != held]
            for seed in cli.loco_seeds:
                net = get(arch, f"loco{held}", seed, tr)
                r = {c: eval_column(net, X, lab, c, device) for c in (held, "R307", "CTR1")}
                results["loco"][arch][f"{held}_s{seed}"] = r
                print(f"  {arch} hold {held}: " + "  ".join(f"{c} acc {v['acc_gap1']:.3f} ρ {v['spearman_m_enc_gap1']:.3f}"
                                                          for c, v in r.items()), flush=True)
        nets = []
        results["full"][arch] = {}
        for seed in cli.seeds:
            net = get(arch, "all4", seed, list(TRAIN_COLS))
            nets.append(net)
            r = {c: eval_column(net, X, lab, c, device) for c in ("R307", "CTR1")}
            results["full"][arch][f"s{seed}"] = r
            print(f"  {arch} all4 s{seed}: " + "  ".join(f"{c} acc {v['acc_gap1']:.3f} ρ {v['spearman_m_enc_gap1']:.3f}"
                                                         for c, v in r.items()), flush=True)
        results["samples"][arch] = {}
        for name, path in (kv.split("=", 1) for kv in cli.samples):
            r = gen_ratio(nets, path, lab, device)
            col = lab.set_index("filename").loc[str(np.load(path)["filename"][0]), "col"]
            r["other_columns_curve_ratio"] = other_column_ratio(nets, X, lab, col, device)
            results["samples"][arch][name] = r
            print(f"  {arch} {name}: ratio gen {r['ratio_gen']:.3f}±{r['ratio_gen_se']:.3f}  anchor "
                  f"{r['ratio_anchor']:.3f}±{r['ratio_anchor_se']:.3f}  curve {r['curve_ratio_gen']:.3f}"
                  f"±{r['curve_ratio_gen_se']:.3f}  other cols {r['other_columns_curve_ratio']['mean']:.3f}", flush=True)

    enc = encoder_table(lab)
    results["encoder_progress"] = {f"{c}@{dr:g}": {k: float(v) for k, v in row.items()} for (c, dr), row in enc.iterrows()}
    with open(os.path.join(cli.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=1)
    write_report(results, enc, os.path.join(cli.out_dir, "report.txt"))
    print(f"[critic] wrote {cli.out_dir}/metrics.json, report.txt", flush=True)


def write_report(R, enc, path):
    L = []
    keys = ("acc_gap1", "acc_gap2", "acc_same_level", "spearman_m_enc_gap1", "spearman_m_enc_all",
            "spearman_m_energy_gap1", "spearman_m_strdrop_gap1", "spearman_m_level_gap1", "spearman_enc_level_gap1", "same_image_abs_m")
    for arch, folds in R["loco"].items():
        L.append(f"== {arch}: leave-one-column-out (held-out column / R307 / CTR1)")
        for fold, cols in folds.items():
            for c, v in cols.items():
                L.append(f"  {fold:12s} {c:6s} " + " ".join(f"{k}={v[k]:.3f}" for k in keys))
    for arch, seeds in R["full"].items():
        L.append(f"== {arch}: trained on all 4 columns")
        for c in ("R307", "CTR1"):
            for k in keys:
                vals = [seeds[s][c][k] for s in seeds]
                L.append(f"  {c:6s} {k:24s} mean {np.mean(vals):.3f}  sd {np.std(vals):.3f}  ({', '.join(f'{x:.3f}' for x in vals)})")
    for arch, samp in R["samples"].items():
        L.append(f"== {arch}: generated samples, Σ m̂(gen) / Σ m̂(real) on pairs ≥ 1 level apart")
        for name, r in samp.items():
            L.append(f"  {name:12s} ratio_gen {r['ratio_gen']:.3f}±{r['ratio_gen_se']:.3f}  ratio_anchor "
                     f"{r['ratio_anchor']:.3f}±{r['ratio_anchor_se']:.3f}  curve_gen {r['curve_ratio_gen']:.3f}"
                     f"±{r['curve_ratio_gen_se']:.3f}  curve_anchor {r['curve_ratio_anchor']:.3f}  "
                     f"other-columns curve {r['other_columns_curve_ratio']['mean']:.3f}  ρ(gen,real) {r['spearman_gen_real']:.3f}")
    L.append("== encoder progress P = (−strength + energy)/2 at the last frame of each drift level (faces averaged)")
    for k in ("P", "enc_energy", "enc_strength"):
        L.append(f"-- {k}")
        L.append(enc[k].unstack(0).round(3).to_string())
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
