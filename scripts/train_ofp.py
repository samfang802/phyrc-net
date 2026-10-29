#!/usr/bin/env python3
"""Train / select / evaluate OFP (src/model/ofp.py) at layout level.  CPU, a few seconds per fit.

  OMP_NUM_THREADS=2 python scripts/train_ofp.py --out_dir results/phase1_ofp

Split: train = R315 R330 R615 R1015, val = R307 (selection of hyper-parameters), test = CTR1 (reported only).
Writes sweep.json, final metrics (val + test), models/*.pt, layouts_{split}_{variant}.npz (filename, layout, method, split),
figures.  Layout of an eval frame = state of rollout k at the frame's drift level (a binary 0/1 coverage; no averaging).
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import ofp as O

torch.set_num_threads(2)
TRAIN, VAL, TEST = ("R315", "R330", "R615", "R1015"), ("R307",), ("CTR1",)
EVAL_FN = {"test": "results/eval_suite/v105/test/samples/gn.npz", "val": "results/eval_suite/v106/val/samples/gn.npz"}


def split_metrics(layouts, frames):
    """layouts (N,K,G,G) aligned with frames (DataFrame, same order).  Own implementation (shared tool not required)."""
    N, K = layouts.shape[:2]
    gt = np.stack(frames["gt"].values).astype(np.float32)          # iso>0.5
    cov = np.stack(frames["cov"].values).astype(np.float32)     # exact polygon coverage
    a_gt = cov.mean((1, 2))
    a_pr = layouts.mean((2, 3))                                 # (N,K)
    bias = (a_pr - a_gt[:, None]).mean(1)
    di = frames.DI.values
    hi = np.zeros(N, bool)
    for s, g in frames.groupby("seq").groups.items():
        g = np.array(list(g)); thr = np.quantile(di[g], 2 / 3); hi[g] = di[g] >= thr
    lev = frames.level.values
    inter = (layouts * gt[:, None]).sum((2, 3)); union = np.maximum(layouts, gt[:, None]).sum((2, 3))
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    big = a_gt > 0.1
    nc_pred, lf_pred, nc_gt, lf_gt, div = [], [], [], [], []
    for i in np.where(big)[0]:
        n, lf = O.comp_stats(gt[i]); nc_gt.append(n); lf_gt.append(lf)
        for k in range(K):
            n, lf = O.comp_stats(layouts[i, k]); nc_pred.append(n); lf_pred.append(lf)
        if K > 1:
            ious = []
            for a in range(K):
                for b in range(a + 1, K):
                    u = np.maximum(layouts[i, a], layouts[i, b]).sum()
                    ious.append(np.minimum(layouts[i, a], layouts[i, b]).sum() / u if u else np.nan)
            div.append(np.nanmean(ious))
    r = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)
    return {
        "area_bias_all": r(bias.mean()), "area_bias_highDI": r(bias[hi].mean()),
        "area_bias_6": r(bias[lev == 6].mean()), "area_bias_8": r(bias[lev == 8].mean()),
        "area_mae_all": r(np.abs(a_pr - a_gt[:, None]).mean()),
        "iou_all": r(np.nanmean(iou)), "iou_big": r(np.nanmean(iou[big])), "iou_highDI": r(np.nanmean(iou[hi])),
        "ncomp_pred": r(np.mean(nc_pred)), "ncomp_gt": r(np.mean(nc_gt)),
        "largest_frac_pred": r(np.nanmean(lf_pred)), "largest_frac_gt": r(np.nanmean(lf_gt)),
        "pairwise_iou_samples": r(np.mean(div)) if div else None,
        "gt_area_highDI": r(a_gt[hi].mean()), "n_frames": int(N), "n_big": int(big.sum()),
    }


def monotone_violations(roll):  # roll (K,J,G,G)
    return int((roll[:, 1:] < roll[:, :-1]).sum())


def rollouts(model, seqs, K, seed):
    out = {}
    for name, s in seqs.items():
        g = torch.Generator().manual_seed(seed + abs(hash(name)) % 1000 if False else seed + sum(map(ord, name)))
        out[name] = model.rollout(torch.from_numpy(s["tau"]), K, g)
    return out


def eval_model(model, seqs, K=8, seed=0):
    import pandas as pd
    rolls = rollouts(model, seqs, K, seed)
    frames = pd.concat([s["frames"].assign(seq=n) for n, s in seqs.items()], ignore_index=True)
    lay = np.concatenate([O.layout_for_frames(rolls[n], s["frames"]) for n, s in seqs.items()])
    m = split_metrics(lay, frames)
    m["monotone_violations"] = sum(monotone_violations(r) for r in rolls.values())
    return m, rolls, frames, lay


def prior_mean_layout(train, seqs_eval):
    """Baseline: P(c | tau) = mean over training sequences of their state at that tau (step in max-drift), threshold 0.5."""
    import pandas as pd
    outs = {}
    for n, s in seqs_eval.items():
        P = []
        for t in train.values():
            j = np.searchsorted(t["tau"], s["tau"], side="right") - 1
            st = np.where(j[:, None, None] >= 0, t["states"][np.clip(j, 0, None)], 0)
            P.append(st)
        outs[n] = np.mean(P, 0)  # (J,G,G)
    return outs


def val_nll(model, seqs):
    with torch.no_grad():
        n, r, e = model.nll(seqs)
    return float(n / r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/phase1_ofp")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--reuse_sweep", action="store_true", help="read sweep.json, only run the extra c configs below")
    a = ap.parse_args()
    os.makedirs(f"{a.out_dir}/models", exist_ok=True)
    allseq = O.load_sequences()
    train, val, test = (O.sequences_of(allseq, x) for x in (TRAIN, VAL, TEST))
    print("train", list(train), "val", list(val), "test", list(test))
    nt = sum(int(((np.concatenate([np.zeros((1, 32, 32)), s["states"]])[:-1] == 0) & (s["states"] > 0.5)).sum()) for s in train.values())
    print("train spall events (cells newly spalled across level steps):", nt)

    grid = []
    for v in "abc":
        for ks in ([3] if v == "a" else [3, 5]):
            if v == "c":
                for hidden in (4, 8):
                    for wd in (1e-3, 1e-2, 1e-1):
                        grid.append(dict(variant=v, ks=ks, hidden=hidden, wd=wd))
            else:
                grid.append(dict(variant=v, ks=ks, hidden=8, wd=1e-3))
    sweep = []
    if a.reuse_sweep:
        sweep = json.load(open(f"{a.out_dir}/sweep.json"))
        grid = [dict(variant="c", ks=3, hidden=8, wd=w) for w in (1e-4, 3e-3, 3e-2)]
    for cfg in grid:
        t0 = time.time()
        m = O.fit(cfg["variant"], train, ks=cfg["ks"], hidden=cfg["hidden"], wd=cfg["wd"], steps=a.steps)
        r = dict(cfg, train_nll=val_nll(m, train), val_nll=val_nll(m, val),
                 n_params=sum(p.numel() for p in m.parameters()), sec=round(time.time() - t0, 1))
        mv, *_ = eval_model(m, val, a.K)
        r["val_area_bias_all"], r["val_iou_big"] = mv["area_bias_all"], mv["iou_big"]
        sweep.append(r)
        print(r, flush=True)
    json.dump(sweep, open(f"{a.out_dir}/sweep.json", "w"), indent=1)

    best = {}
    for v in "abc":
        c = [r for r in sweep if r["variant"] == v]
        best[v] = min(c, key=lambda r: r["val_nll"])          # rule 1: R307 teacher-forced NLL
    # rule 2 (reported as 'c_iou'): R307 rollout IoU on frames with spall > 10 % (what the renderer needs)
    best["c_iou"] = max([r for r in sweep if r["variant"] == "c"], key=lambda r: r["val_iou_big"])
    print("selected (min R307 teacher-forced NLL):", best)

    # refit selected, evaluate val + test, save layouts
    import pandas as pd
    final = {}
    ref = {s: np.load(p, allow_pickle=True)["filename"] for s, p in EVAL_FN.items()}
    for v, cfg in best.items():
        m = O.fit(cfg["variant"], train, ks=cfg["ks"], hidden=cfg["hidden"], wd=cfg["wd"], steps=a.steps)
        torch.save({"state": m.state_dict(), "cfg": cfg}, f"{a.out_dir}/models/ofp_{v}.pt")
        final[v] = {"cfg": cfg, "params": {k: v_.detach().numpy().round(4).tolist() for k, v_ in m.named_parameters()
                                              if v_.numel() <= 30}}
        for split, sq in (("val", val), ("test", test)):
            met, rolls, frames, lay = eval_model(m, sq, a.K)
            final[v][split] = met
            fidx = {f: i for i, f in enumerate(frames.filename)}
            order = np.array([fidx[f] for f in ref[split]])
            np.savez_compressed(f"{a.out_dir}/layouts_{split}_{v}.npz", filename=ref[split], layout=lay[order].astype(np.float32),
                                method=np.array("ofp"), split=np.array(split))
            torch.save({n: r for n, r in rolls.items()}, f"{a.out_dir}/rollouts_{split}_{v}.pt")
        print(v, final[v]["val"], final[v]["test"], flush=True)
    # baseline: mean training field (deterministic, K=1)
    base = {}
    for split, sq in (("val", val), ("test", test)):
        P = prior_mean_layout(train, sq)
        frames = pd.concat([s["frames"].assign(seq=n) for n, s in sq.items()], ignore_index=True)
        lay = np.concatenate([(P[n][s["frames"].j.values] > 0.5).astype(np.float32) for n, s in sq.items()])[:, None]
        base[split] = split_metrics(lay, frames)
    final["prior_mean"] = base
    json.dump(final, open(f"{a.out_dir}/final_metrics.json", "w"), indent=1)
    print("baseline prior_mean", base)


if __name__ == "__main__":
    main()
