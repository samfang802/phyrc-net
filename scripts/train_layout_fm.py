#!/usr/bin/env python3
"""Train + evaluate the C2 layout generator (32x32 SDF flow matching). docs/redesign/phase1_c2.md

Split: train = R315/R330/R615/R1015 (8 seqs), val = R307, test = CTR1 (report only).
  --cond tau|g|taug   physical condition (tau = max drift reached so far; g = GRU state, PCA --g_dim)
  --use_prev 0|1      previous layout as input (rollout); 0 = every frame independent
  --proj 0|1          monotone projection new = max(prev_projected, generated) at rollout time
Outputs (--out_dir): metrics.json, gen_{val,test}.npz (filename, layout N,K,32,32 coverage, method, split).
"""
import argparse, json, os, sys, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
from src.model.layout_fm import LayoutUNet, fm_loss, sample, mask_to_sdf, sdf_to_coverage, S

D = "Data/DamagePattern"
TRAIN_COLS = ("R315", "R330", "R615", "R1015")


def load_data(g_dim):
    d = np.load(f"{D}/derived/onset_field.npz", allow_pickle=True)
    fn, seq, md, ts = d["filenames"].astype(str), d["seq"].astype(str), d["max_drift"], d["timestep"]
    iso, cov = d["iso32"], d["cov32"]
    gz = np.load(f"{D}/derived/gru_g.npz")
    gmap = {f: i for i, f in enumerate(gz["filename"].astype(str))}
    g = gz["g"][[gmap[f] for f in fn]]
    col = np.array([s.split("_")[0] for s in seq])
    face = np.array([0 if s.split("_")[1] in ("NE", "NW") else 1 for s in seq])
    split = np.where(np.isin(col, TRAIN_COLS), "train", np.where(col == "R307", "val", "test"))
    order = np.lexsort((ts, seq))  # sorted by seq then timestep
    # tertile-free; DI for high-DI third
    lab = __import__("pandas").read_csv(f"{D}/label.csv").set_index("filename")
    di = lab.loc[fn, "DI"].values
    tr = split == "train"
    mu, U = g[tr].mean(0), None
    u, s, vt = np.linalg.svd(g[tr] - mu, full_matrices=False)
    gp = (g - mu) @ vt[:g_dim].T
    gp = gp / (gp[tr].std(0) + 1e-6)
    gt_mask = (cov >= 0.5)
    tgt_mask = (iso >= 0.5)
    return dict(fn=fn, seq=seq, md=md, ts=ts, col=col, face=face, split=split, order=order, di=di,
                g=gp.astype(np.float32), gt_mask=gt_mask, tgt_sdf=np.stack([mask_to_sdf(m) for m in tgt_mask]))


def cond_vec(data, kind):
    tau = (data["md"] / 10.0).astype(np.float32)
    parts = []
    if kind in ("tau", "taug"):
        parts.append(np.stack([tau, np.sqrt(tau)], 1))
    if kind in ("g", "taug"):
        parts.append(data["g"])
    f = np.eye(2, dtype=np.float32)[data["face"]]
    parts.append(f)
    return np.concatenate(parts, 1).astype(np.float32)


def seq_index(data, split):
    """dict seq -> frame indices (sorted by timestep) for the split."""
    out = {}
    for i in data["order"]:
        if data["split"][i] == split:
            out.setdefault(data["seq"][i], []).append(i)
    return out


def blur(x, k=5, sigma=1.5):
    ax = torch.arange(k, device=x.device) - k // 2
    w = torch.exp(-ax.float() ** 2 / (2 * sigma ** 2)); w = w / w.sum()
    x = F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), w.view(1, 1, 1, k))
    return F.conv2d(x, w.view(1, 1, k, 1))


def perturb_prev(p, amp, shift):
    if amp <= 0 and shift <= 0:
        return p
    b = p.shape[0]
    n = blur(torch.randn_like(p))
    n = n / (n.std() + 1e-6)
    d = (torch.rand(b, 1, 1, 1, device=p.device) * 2 - 1) * shift  # dilate/erode (sdf units; 0.5 = 1 cell)
    return (p + amp * n + d).clamp(-1, 1)


def train(cfg, data, dev):
    cond = torch.tensor(cond_vec(data, cfg.cond), device=dev)
    sdf = torch.tensor(data["tgt_sdf"], device=dev)[:, None]
    sdf = torch.cat([sdf, -torch.ones(1, 1, S, S, device=dev)])  # index N = empty layout
    N = sdf.shape[0] - 1
    tr = [i for i in range(N) if data["split"][i] == "train"]
    pos = {}
    prev_pool = {}
    for s, idx in seq_index(data, "train").items():
        for k, i in enumerate(idx):
            pos[i] = (s, k)
    ids = torch.tensor(tr, device=dev)
    # previous index table: pv[i, j-1] = frame j steps earlier in the sequence, or N (empty)
    maxj = 12
    pv = np.full((N, maxj), N, dtype=np.int64)
    for s, idx in seq_index(data, "train").items():
        for k, i in enumerate(idx):
            for j in range(1, maxj + 1):
                if k - j >= 0:
                    pv[i, j - 1] = idx[k - j]
    pv = torch.tensor(pv, device=dev)
    gaps = torch.tensor([int(x) for x in cfg.gaps.split(",")], device=dev)
    model = LayoutUNet(cond.shape[1], use_prev=bool(cfg.use_prev), ch=cfg.ch).to(dev)
    ema = LayoutUNet(cond.shape[1], use_prev=bool(cfg.use_prev), ch=cfg.ch).to(dev)
    ema.load_state_dict(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * (0.5 * (1 + np.cos(np.pi * s / cfg.steps)) * 0.95 + 0.05))
    for step in range(cfg.steps):
        bi = ids[torch.randint(len(ids), (cfg.bs,), device=dev)]
        x1 = sdf[bi]
        c = cond[bi].clone()
        if cfg.cond_noise > 0:
            c = c + cfg.cond_noise * torch.randn_like(c) * torch.cat(
                [torch.ones(c.shape[1] - 2, device=dev), torch.zeros(2, device=dev)])
        prev = None
        if cfg.use_prev:
            j = gaps[torch.randint(len(gaps), (cfg.bs,), device=dev)]
            pj = pv[bi, j - 1]
            prev = sdf[pj]
            # in a rollout the previous frame is never above the current one's target (monotone data)
            prev = perturb_prev(prev, cfg.pert_amp, cfg.pert_shift)
        flip = torch.rand(cfg.bs, device=dev) < 0.5
        x1 = torch.where(flip[:, None, None, None], x1.flip(-1), x1)
        if prev is not None:
            prev = torch.where(flip[:, None, None, None], prev.flip(-1), prev)
        loss = fm_loss(model, x1, c, prev)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - cfg.ema)
        if step % 1000 == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)
    return ema.eval()


@torch.no_grad()
def rollout(model, data, split, cfg, dev, K, seed):
    """-> sdf (N_all, K, 32, 32) for frames of `split` (others zeros) ; projected + raw."""
    cond = torch.tensor(cond_vec(data, cfg.cond), device=dev)
    N = len(data["fn"])
    proj = np.zeros((N, K, S, S), np.float32); raw = np.zeros_like(proj)
    gen = torch.Generator(device=dev).manual_seed(seed)
    for s, idx in seq_index(data, split).items():
        prev = -torch.ones(K, 1, S, S, device=dev)  # first frame: all negative
        run = prev.clone()
        for i in idx:
            c = cond[i][None].expand(K, -1)
            x = sample(model, c, prev if cfg.use_prev else None, steps=cfg.ode_steps, generator=gen)
            raw[i] = x[:, 0].cpu().numpy()
            run = torch.maximum(run, x)           # monotone projection
            proj[i] = run[:, 0].cpu().numpy()
            # next input: projected layout (proj=1) or the raw generated layout (proj=0)
            prev = run if cfg.proj else x
    return proj, raw


def components(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    a = st[1:, cv2.CC_STAT_AREA]
    return (len(a), float(a.max() / a.sum()) if len(a) else 0.0)


def metrics(data, sdf, split, train_masks):
    """sdf (N,K,32,32) for the split's frames. Compare with the hand-polygon coverage >= 0.5."""
    idx = np.array([i for i in range(len(data["fn"])) if data["split"][i] == split])
    gm = data["gt_mask"][idx].astype(np.float32)           # (n,32,32)
    gen = (sdf[idx] > 0).astype(np.float32)                # (n,K,32,32)
    n, K = gen.shape[:2]
    ga = gm.mean((1, 2)); aa = gen.mean((2, 3))             # (n,), (n,K)
    amean = aa.mean(1)
    inter = (gen * gm[:, None]).sum((2, 3)); uni = ((gen + gm[:, None]) > 0).sum((2, 3))
    iou = np.where(uni > 0, inter / np.maximum(uni, 1), 1.0)   # empty/empty = 1
    sp = ga > 0.02
    di = data["di"][idx]; hi = di >= np.quantile(di, 2 / 3)
    lvl = np.array([int(f.split("_")[2]) for f in data["fn"][idx]])
    out = dict(n_frames=int(n), area_gt_mean=float(ga.mean()), area_gen_mean=float(amean.mean()),
               area_bias=float((amean - ga).mean()), area_mae=float(np.abs(amean - ga).mean()),
               area_bias_highDI=float((amean - ga)[hi].mean()),
               area_bias_6=float((amean - ga)[lvl == 6000].mean()), area_bias_8=float((amean - ga)[lvl == 8000].mean()),
               iou_spall_frames=float(iou[sp].mean()), iou_all=float(iou.mean()),
               iou_highDI=float(iou[hi].mean()))
    # K-sample diversity: mean pairwise IoU among the K samples on spall frames
    pw = []
    for i in np.where(sp)[0]:
        g = gen[i].reshape(K, -1); it = g @ g.T; un = g.sum(1)[:, None] + g.sum(1)[None] - it
        pw.append(((it / np.maximum(un, 1))[~np.eye(K, dtype=bool)]).mean())
    out["pairwise_iou_K"] = float(np.mean(pw)) if pw else None
    # connected components on spall frames (GT vs gen)
    gc = [components(gm[i]) for i in np.where(sp)[0]]
    nc = [components(gen[i, k]) for i in np.where(sp)[0] for k in range(K)]
    out["ncomp_gt"] = float(np.mean([c[0] for c in gc])); out["ncomp_gen"] = float(np.mean([c[0] for c in nc]))
    out["largest_frac_gt"] = float(np.mean([c[1] for c in gc])); out["largest_frac_gen"] = float(np.mean([c[1] for c in nc]))
    # copying: best IoU to any training layout (area > 0.02)
    tm = train_masks.reshape(len(train_masks), -1)
    def nn_iou(X):
        X = X.reshape(len(X), -1); it = X @ tm.T
        un = X.sum(1)[:, None] + tm.sum(1)[None] - it
        return (it / np.maximum(un, 1)).max(1)
    g_nn = nn_iou(gm[sp])
    gen_sp = gen[sp].reshape(-1, S, S); gen_big = gen_sp.mean((1, 2)) > 0.02
    gen_nn = nn_iou(gen_sp[gen_big])
    out["nnIoU_gt_mean"] = float(g_nn.mean()); out["nnIoU_gen_mean"] = float(gen_nn.mean())
    out["nnIoU_gen_frac_gt0.8"] = float((gen_nn > 0.8).mean()); out["nnIoU_gt_frac_gt0.8"] = float((g_nn > 0.8).mean())
    out["nnIoU_gen_max"] = float(gen_nn.max())
    out["score"] = out["area_mae"] + 0.5 * (1 - out["iou_spall_frames"])
    return out


def save_npz(path, data, sdf, split, ref_npz):
    fn_ref = np.load(ref_npz, allow_pickle=True)["filename"].astype(str)
    pos = {f: i for i, f in enumerate(data["fn"])}
    sel = np.array([pos[f] for f in fn_ref])
    cov = sdf_to_coverage(sdf[sel])
    np.savez_compressed(path, filename=fn_ref, layout=cov.astype(np.float32), method=np.asarray("c2"), split=np.asarray(split))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cond", default="tau", choices=("tau", "g", "taug"))
    p.add_argument("--use_prev", type=int, default=1)
    p.add_argument("--proj", type=int, default=1)
    p.add_argument("--g_dim", type=int, default=16)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--ema", type=float, default=0.995)
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--pert_amp", type=float, default=0.2)
    p.add_argument("--pert_shift", type=float, default=0.25)
    p.add_argument("--cond_noise", type=float, default=0.0)
    p.add_argument("--gaps", default="1,1,1,2,3,5")
    p.add_argument("--ode_steps", type=int, default=24)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_test", type=int, default=1, help="also evaluate + save test (CTR1; report only)")
    cfg = p.parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(cfg.g_dim)
    tm = data["gt_mask"][(data["split"] == "train") & (data["gt_mask"].mean((1, 2)) > 0.02)].astype(np.float32)
    t0 = time.time()
    model = train(cfg, data, dev)
    res = dict(cfg=vars(cfg), train_sec=time.time() - t0)
    for split, ref in (("val", "results/eval_suite/v106/val/samples/gn.npz"), ("test", "results/eval_suite/v105/test/samples/gn.npz")):
        if split == "test" and not cfg.save_test:
            continue
        proj, raw = rollout(model, data, split, cfg, dev, cfg.k, 123)
        res[split] = metrics(data, proj, split, tm)
        res[split + "_rawgen"] = metrics(data, raw, split, tm)  # un-projected last-step generation
        save_npz(os.path.join(cfg.out_dir, f"gen_{split}.npz"), data, proj, split, ref)
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model.pt"))
    json.dump(res, open(os.path.join(cfg.out_dir, "metrics.json"), "w"), indent=1)
    v = res["val"]
    print("VAL", {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()}, flush=True)


if __name__ == "__main__":
    main()
