#!/usr/bin/env python3
"""DINOv3 KID / MMD for generated damage images, with small-sample handling.

Inputs are npz files from generate_eval_samples.py (or any script using save_samples_npz).
Reference = the images in --image_dir for the npz rows (test split by default); all npz must share the rows.
Backbone: DINOv3 ViT-S/16 by default (gated on Hugging Face); --dino_model facebook/dinov2-small
is public and ranked the models nearly the same on test and val (docs/eval/distribution.md).

Per model:
  kid_global       KID on [CLS ; mean patch], block U-statistic. Every SE here is a leave-one-block-out
                   jackknife, block = one face at one drift level (neighbouring frames are near-duplicates)
  mmd_rbf_global   RBF MMD², bandwidth = median distance of the real reference (fixed per run)
  kid_patch        KID on a fixed 4×4 spatial patch grid (bag-of-patches across groups)
  kid_by_di        kid_global inside DI tertiles of the reference
  kid_soft_di      kid_global reweighted by a wide Gaussian window on reference DI
  probe            frozen DINOv3 + ridge fitted on train reals → DI / R_spall of the fakes vs the
                   condition's label (R², MAE); the same probe on the held-out reals is the ceiling
  kid_by_spall     kid_global inside R_spall bands (≤0.05, 0.05–0.3, >0.3) of the reference
  saturated_frames share of images with >1% saturated pixels (blown-out artifacts), fake vs real
  probe_spall_top  probe-predicted R_spall on the R_spall>0.3 conditions: real vs fake
  nbr_global       GT-neighbourhood contrast (higher = better): distance to the reals outside the GT's
                   neighbourhood minus distance to the k same-face closest-DI frames; `shuffled` moves the
                   fakes to random conditions (must drop to ~0); `frac_of_ceiling` divides by the same score
                   for a real photo of another specimen at matching DI (the reachable target, since the
                   generator does not know the specimen)
  nbr_patch        same contrast on position-aligned patch tokens (where on the column the damage is)
  progression_*    how far the images move away from the undamaged (lowest-drift) level as drift grows:
                   per face, unbiased distance between each drift level's feature centroid and the
                   lowest level's (base level excluded from every score). ratio = the model's summed
                   distance / GT's (1 = as much change as the real column), corr = shape agreement
                   with GT's curve, monotone = Spearman vs drift; SE = leave-one-cycle-out jackknife.
                   `shuffled_*` mixes the drift levels by moving fakes to random conditions; the
                   other-specimens row is real photos of other columns at matched DI
  interference     share of generated images with a detected fixture (inpaint_generated_samples.py);
                   a defect reported on its own, removed before every other score when the npz says so
  prdc             precision / recall / density / coverage, one fake per condition, averaged over K
  memorization     NN distance to train reals, fakes vs leave-one-column-out train reals
Reference rows:
  real_halves      reals split into two random halves (noise floor)
  other_real       another real column (default val R307) vs the reference
Comparisons:
  every model pair: D = MMD²(A,R) − MMD²(B,R) with a paired jackknife SE and z.

Example:
  python scripts/eval/eval_dinov3_kid.py \
      --samples base=results/.../samples_test_k8.npz refiner=results/.../samples_test_k8.npz \
      --out_dir results/eval_dinov3_kid/run1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from itertools import combinations

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics import dino_mmd as dm


def _parse_args():
    p = argparse.ArgumentParser(description="DINOv3 KID / MMD evaluation")
    p.add_argument("--samples", nargs="+", required=True, metavar="NAME=NPZ")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dino_model", default=None, help="Backbone id; overrides --dino_model_id")
    p.add_argument("--dino_model_id", default=dm.DEFAULT_DINO)
    p.add_argument("--n_patch", type=int, default=16, help="Patch tokens kept per image (0 = skip patch KID)")
    p.add_argument("--other_split", default="val", help="Real split used as the 'another column' reference ('' = skip)")
    p.add_argument("--memo_split", default="train", help="Split for the memorization check ('' = skip)")
    p.add_argument("--data_path", default="Data/DamagePattern/id_list_val_R307.json")
    p.add_argument("--image_dir", default="Data/DamagePattern/images")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--no_clahe", action="store_true", help="Image files are already CLAHE'd (e.g. the inpainted set)")
    p.add_argument(
        "--normalize",
        default="none",
        choices=dm.NORMALIZE_MODES,
        help="Per-image tone normalization before features, applied to every image set",
    )
    p.add_argument("--n_halves", type=int, default=50)
    p.add_argument("--probe_split", default="train", help="Split the ridge probe is fitted on ('' = skip)")
    p.add_argument("--prdc_k", type=int, default=5)
    p.add_argument("--nbr_k", type=int, default=4, help="GT neighbours: same specimen and face, closest DI")
    p.add_argument("--nbr_shuffle", type=int, default=20)
    p.add_argument("--prog_shuffle", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _load_npz(path):
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in ("fake", "real", "cond", "DI", "filename")}
    out["meta"] = json.loads(str(z["meta"])) if "meta" in z else {}
    # Written by inpaint_generated_samples.py: what was detected (and removed) before scoring.
    for k in ("interference_flag", "interference_mask"):
        if k in z:
            out[k] = z[k]
    return out


def _model_slug(model_id):
    return model_id.replace("/", "__").replace(" ", "_")


class _Cache:
    def __init__(self, extractor, out_dir, n_patch, seed, normalize="none"):
        # One directory per backbone so ViT-S and ViT-B caches cannot collide.
        # Patch indices are deterministic, so the seed is not part of the cache key.
        slug = _model_slug(extractor.model_id)
        self.fx, self.dir, self.n_patch, self.seed = extractor, os.path.join(out_dir, "feature_cache", slug), n_patch, seed
        self.normalize = normalize
        os.makedirs(self.dir, exist_ok=True)

    def __call__(self, u8):
        # Every image set (reference, fakes, other splits) goes through here, so the same tone
        # normalization is applied to all of them.
        u8 = dm.normalize_u8(u8, self.normalize)
        key = hashlib.md5(u8.tobytes() + f"{self.n_patch}".encode()).hexdigest()
        path = os.path.join(self.dir, key + ".npz")
        if os.path.isfile(path):
            z = np.load(path)
            return z["g"], (z["p"] if self.n_patch > 0 else None)
        g, p = self.fx(u8, n_patch=self.n_patch, seed=self.seed)
        np.savez(path, g=g, **({"p": p} if p is not None else {}))
        return g, p


def _r(d):
    return {k: (round(v, 8) if isinstance(v, float) else v) for k, v in d.items()}


def _labels(filenames):
    import pandas as pd

    di = pd.read_csv("Data/DamagePattern/label.csv").set_index("filename")["DI"]
    rs = pd.read_csv("Data/DamagePattern/r_spall.csv").set_index("filename")["R_spall"]
    names = list(filenames)
    return {"DI": di.loc[names].to_numpy(np.float64), "R_spall": rs.loc[names].to_numpy(np.float64)}


def _fit_probes(feat, labels):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        k: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 30))).fit(feat, y)
        for k, y in labels.items()
    }


def _score_probes(probes, feat, labels, di):
    """Score one feature row per condition (fakes must already be averaged over K)."""
    return {k: dm.regression_diagnostics(labels[k], m.predict(feat), di) for k, m in probes.items()}


def _plot_progression(curves, reference_rows, out_path):
    """One panel per face: GT, every model, distance from the lowest drift level (global features)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = curves["GT"]
    faces = sorted(gt)
    fig, axes = plt.subplots(1, len(faces), figsize=(4.2 * len(faces), 3.4), squeeze=False)
    xs = reference_rows.get("progression_global_other_specimens", {}).get("_curve") or {}
    for ax, face in zip(axes[0], faces):
        lv, d = gt[face]
        ax.plot(np.asarray(lv) / 1000, d, "k-o", lw=2.2, label="GT")
        if face in xs:
            ax.plot(np.asarray(xs[face][0]) / 1000, xs[face][1], "--", color="0.55", lw=1.4, label="other specimens (real)")
        for name, cv in curves.items():
            if name == "GT" or face not in cv:
                continue
            lv_m, d_m = cv[face]
            ax.plot(np.asarray(lv_m) / 1000, d_m, "-o", lw=1.4, ms=3.5, label=name)
        ax.set_title(face)
        ax.set_xlabel("drift code (×1000)")
        ax.set_ylabel("distance from lowest drift level")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[kid] wrote {out_path}", flush=True)


def _circular_eval(name, path, meta, model_id):
    """oracle_dino is conditioned on DINOv3 features, so only a DINOv3 backbone measures it circularly.
    Another backbone (e.g. DINOv2) is a related self-supervised model, not an independent judge."""
    blob = f"{name} {path} {json.dumps(meta, default=str)}"
    return "oracle_dino" in blob and "dinov3" in model_id.lower()


def main():
    cli = _parse_args()
    os.makedirs(cli.out_dir, exist_ok=True)
    rng = np.random.default_rng(cli.seed)
    clahe = not cli.no_clahe

    runs = {}
    for item in cli.samples:
        name, path = item.split("=", 1)
        runs[name] = _load_npz(path)
        runs[name]["path"] = os.path.abspath(path)
    ref = next(iter(runs.values()))
    for name, r in runs.items():
        if not np.array_equal(r["filename"], ref["filename"]):
            raise ValueError(f"{name}: npz rows (filenames) differ from the first npz")

    # The reference is always read from --image_dir, so every model is scored against the same
    # images whatever its npz carries in `real` (original GT for a fresh sample file, the inpainted
    # GT for older ones). The npz copy is only compared, as a check of the preprocessing.
    DI = ref["DI"]
    real_u8 = dm.load_real_uint8(list(ref["filename"]), cli.image_dir, cli.image_size, clahe)
    G = len(real_u8)
    gr = np.arange(G)
    with open(cli.data_path) as f:
        split_ids = json.load(f)
    pipeline_mad = float(np.abs(ref["real"].astype(np.int16) - real_u8.astype(np.int16)).mean())
    print(f"[kid] reference n={G} from {cli.image_dir}; first npz's `real` vs files mean |Δ| = {pipeline_mad:.3f} "
          "(0 when the npz already carries the same reference; ~7 for original GT vs the inpainted set)", flush=True)

    model_id = cli.dino_model or cli.dino_model_id
    fx = dm.DinoFeatures(model_id, device=cli.device)
    feats = _Cache(fx, cli.out_dir, cli.n_patch, cli.seed, cli.normalize)
    real_g, real_p = feats(real_u8)
    sigma = dm.median_distance(real_g)
    tert = np.quantile(DI, [1 / 3, 2 / 3])
    di_bins = {"low": DI <= tert[0], "mid": (DI > tert[0]) & (DI <= tert[1]), "high": DI > tert[1]}
    ref_labels = _labels(ref["filename"])
    R_sp = ref_labels["R_spall"]
    # Jackknife unit: one face at one drift level (~6 near-duplicate frames), not one frame.
    blocks = np.array([dm.frame_block(n) for n in ref["filename"]])
    spall_bins = {"le0.05": R_sp <= 0.05, "0.05_0.3": (R_sp > 0.05) & (R_sp <= 0.3), "gt0.3": R_sp > 0.3}

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dino_model_id": model_id,
        "feature_cache": os.path.join(cli.out_dir, "feature_cache", _model_slug(model_id)),
        "prdc_degraded": True,
        "prdc_note": (
            "PRDC is kept for compatibility but excluded from the comparison table: "
            "a single specimen makes 5-NN radii too small, so the scores degenerate."
        ),
        "circular_eval_warning": False,
        "normalize": cli.normalize,
        "image_dir": cli.image_dir,
        "n_reference": G,
        "reference_files_check_mean_abs_uint8": pipeline_mad,
        "rbf_sigma": sigma,
        "di_tertile_edges": tert.tolist(),
        "n_patch": cli.n_patch,
        "models": {},
        "reference_rows": {},
        "comparisons": {},
    }

    # ---- reference rows
    S_rr = dm.group_sums(real_g, gr, real_g, gr, G, dm.poly_kernel, cli.device)
    rr = dm.GroupMMD(S_rr, S_rr, S_rr, np.ones(G), np.ones(G))
    halves = []
    for _ in range(cli.n_halves):
        perm = rng.permutation(G)
        a = np.zeros(G, bool)
        a[perm[: G // 2]] = True
        halves.append(rr.value(a, ~a))
    report["reference_rows"]["real_halves_kid"] = {"mean": float(np.mean(halves)), "std": float(np.std(halves))}

    if cli.other_split:
        other_u8 = dm.load_real_uint8(split_ids[cli.other_split], cli.image_dir, cli.image_size, clahe)
        og, _ = feats(other_u8)
        n_o = len(og)
        go, gt = np.arange(n_o), np.arange(n_o, n_o + G)
        ngr = n_o + G
        om = dm.GroupMMD(
            dm.group_sums(og, go, og, go, ngr, dm.poly_kernel, cli.device),
            dm.group_sums(real_g, gt, real_g, gt, ngr, dm.poly_kernel, cli.device),
            dm.group_sums(og, go, real_g, gt, ngr, dm.poly_kernel, cli.device),
            dm.counts(go, ngr),
            dm.counts(gt, ngr),
        )
        report["reference_rows"][f"other_real_{cli.other_split}_kid"] = {"value": om.value(), "n": n_o}
        # The reference is one column of near-duplicate consecutive frames, so k-NN radii are tiny.
        # PRDC of another real column shows how degenerate PRDC is on this reference.
        report["reference_rows"][f"other_real_{cli.other_split}_prdc"] = dm.prdc(real_g, og, cli.prdc_k)

    split_feats = {}

    def _split_feats(split):
        if split not in split_feats:
            u8 = dm.load_real_uint8(split_ids[split], cli.image_dir, cli.image_size, clahe)
            split_feats[split] = feats(u8)
        return split_feats[split]

    def _split_g(split):
        return _split_feats(split)[0]

    train_g = None
    if cli.memo_split:
        # Copying baseline = how close a *new real column* gets to the train set: leave-one-column-out
        # inside train. Held-out test reals are not used, since the test column itself may sit far
        # from the train columns (specimen shift), which would flag every fake as "copying".
        train_g = _split_g(cli.memo_split)
        train_names = split_ids[cli.memo_split]
        col = np.array([n.split("_")[0] for n in train_names])
        D = np.sqrt(np.maximum(dm.sq_dist(train_g, train_g), 0.0))
        D[col[:, None] == col[None, :]] = np.inf
        d_ref = D.min(1)
        d_test, _ = dm.nn_distance(real_g, train_g)

    probes = None
    if cli.probe_split:
        probes = _fit_probes(_split_g(cli.probe_split), _labels(split_ids[cli.probe_split]))
        report["reference_rows"]["probe_on_heldout_reals"] = _score_probes(probes, real_g, ref_labels, DI)

    # ---- GT-neighbourhood contrast: neighbours and the cross-specimen ceiling
    nbrs = dm.gt_neighbors(ref["filename"], DI, cli.nbr_k)
    spaces = {"global": (real_g, "global")}
    if real_p is not None:
        spaces["patch"] = (real_p, "patch")
    ref_specs = {dm.frame_key(n)[0] for n in ref["filename"]}
    pool_names, pool_g, pool_p = [], [], []
    for split in split_ids:
        keep = [i for i, n in enumerate(split_ids[split]) if dm.frame_key(n)[0] not in ref_specs]
        if not keep:
            continue
        g_s, p_s = _split_feats(split)
        pool_names += [split_ids[split][i] for i in keep]
        pool_g.append(g_s[keep])
        if p_s is not None:
            pool_p.append(p_s[keep])
    nbr_ceiling = {}
    if pool_names:
        pool_DI = _labels(pool_names)["DI"]
        pool_spec = np.array([dm.frame_key(n)[0] for n in pool_names])
        pool_feat = {"global": np.concatenate(pool_g)}
        if pool_p:
            pool_feat["patch"] = np.concatenate(pool_p)
        specs = sorted(set(pool_spec))
        spec_pick = {}
        for sp in specs:
            idx = np.flatnonzero(pool_spec == sp)
            spec_pick[sp] = idx[np.argmin(np.abs(pool_DI[idx][None, :] - DI[:, None]), axis=1)]
        for space, (real_f, kind) in spaces.items():
            per_spec = []
            for sp in specs:
                pick = spec_pick[sp]
                c, _ = dm.nbr_contrast_from_dist(dm.pair_distance(pool_feat[space][pick], real_f, kind), nbrs)
                per_spec.append(c)
            per_cond = np.mean(per_spec, axis=0)
            res, _ = dm.mean_jackknife(per_cond, blocks)
            nbr_ceiling[space] = res["value"]
            report["reference_rows"][f"nbr_{space}_ceiling_other_specimens"] = {
                **_r(res),
                "specimens": specs,
                "per_specimen": {sp: float(c.mean()) for sp, c in zip(specs, per_spec)},
            }

    # ---- drift progression: GT curve and the other-specimens reference
    prog_groups, prog_levels, prog_cycles = dm.drift_levels(ref["filename"])
    prog_ref = {"global": real_g[:, None]}
    if real_p is not None:
        prog_ref["patch"] = real_p.reshape(G, 1, -1)
    prog_xspec = {}
    if pool_names:
        for space in prog_ref:
            pf = pool_feat[space] if space == "global" else pool_feat[space].reshape(len(pool_feat[space]), -1)
            prog_xspec[space] = np.stack([pf[spec_pick[sp]] for sp in specs], axis=1)
            pr = dm.progression(prog_xspec[space], prog_ref[space], prog_groups, prog_levels, prog_cycles, cli.prog_shuffle, cli.seed)
            report["reference_rows"][f"progression_{space}_other_specimens"] = {
                k: (_r(v) if isinstance(v, dict) and "value" in v else v)
                for k, v in pr.items() if not k.startswith("_loo") and k != "_curve_ref"
            }
    progression_plot = {"GT": None}

    # ---- models
    jk = {}
    for name, r in runs.items():
        fake = r["fake"]
        K = fake.shape[1]
        fake_g, fake_p = feats(fake.reshape(-1, *fake.shape[2:]))
        gf = np.repeat(gr, K)
        mm = dm.global_mmds(fake_g, gf, real_g, gr, G, sigma, cli.device)
        out = {"npz": r["path"], "k": int(K), "meta": r["meta"]}
        if "interference_flag" in r:
            flag, mask = r["interference_flag"], r["interference_mask"]
            out["interference"] = {
                "frac_images_flagged": float(flag.mean()),
                "mean_area_px_when_flagged": float(mask.reshape(flag.size, -1).sum(1)[flag.ravel()].mean()) if flag.any() else 0.0,
                "removed_before_scoring": bool(r["meta"].get("interference_removed", False)),
            }
        jk[name] = {}
        for key in ("kid", "mmd_rbf"):
            res, loo = mm[key].jackknife(blocks=blocks)
            out[f"{key}_global"] = _r(res)
            jk[name][f"{key}_global"] = (res, loo)
        out["kid_by_di"] = {b: _r(mm["kid"].jackknife(m, blocks)[0]) for b, m in di_bins.items()}
        soft, soft_loo = dm.kid_soft_di(mm["kid"], DI).jackknife(blocks)
        out["kid_soft_di"] = _r(soft)
        jk[name]["kid_soft_di"] = (soft, soft_loo)
        out["kid_by_spall"] = {b: _r(mm["kid"].jackknife(m, blocks)[0]) for b, m in spall_bins.items() if m.sum() >= 4}
        # Artifact rate: images with >1% saturated pixels (≤5 or ≥250). Real images almost never have any.
        sat = lambda u: ((u <= 5) | (u >= 250)).reshape(len(u), -1).mean(1) > 0.01
        out["saturated_frames"] = {
            "fake_frac": float(sat(fake.reshape(-1, *fake.shape[2:])).mean()),
            "real_frac": float(sat(real_u8).mean()),
        }

        if fake_p is not None:
            res, loo = dm.patch_kid(fake_p.reshape(G * K, *fake_p.shape[1:]), gf, real_p, gr, G, cli.device).jackknife(blocks=blocks)
            out["kid_patch"] = _r(res)
            jk[name]["kid_patch"] = (res, loo)

        fg = fake_g.reshape(G, K, -1)
        if probes is not None:
            pred_mean, pred_std = {}, {}
            for target, model in probes.items():
                per_k = np.stack([model.predict(fg[:, k]) for k in range(K)], axis=1)
                pred_mean[target] = per_k.mean(axis=1)
                pred_std[target] = per_k.std(axis=1)
            out["probe"] = {t: dm.regression_diagnostics(ref_labels[t], pred_mean[t], DI) for t in probes}
            out["_probe_draws"] = {"mean": pred_mean, "std": pred_std}
            top = spall_bins["gt0.3"]
            if top.sum() >= 3:  # does the model draw spalling where the condition has a lot of it?
                pr = lambda f: float(np.mean(probes["R_spall"].predict(f[top])))
                out["probe_spall_top"] = {
                    "n": int(top.sum()),
                    "true_mean": float(R_sp[top].mean()),
                    "probe_on_real": pr(real_g),
                    "probe_on_fake": float(pred_mean["R_spall"][top].mean()),
                }

        for space, (real_f, kind) in spaces.items():
            fake_f = fg if space == "global" else fake_p.reshape(G, K, *fake_p.shape[1:])
            con, near, shuf = dm.nbr_contrast(fake_f, real_f, nbrs, kind, cli.nbr_shuffle, cli.seed)
            res, loo = dm.mean_jackknife(con, blocks)
            ceil = nbr_ceiling.get(space)
            out[f"nbr_{space}"] = {
                **_r(res),
                "near_dist": float(near.mean()),
                "shuffled_mean": float(shuf.mean()),
                "shuffled_std": float(shuf.std()),
                "frac_of_ceiling": float(res["value"] / ceil) if ceil else None,
            }
            jk[name][f"nbr_{space}"] = (res, loo)

        for space in prog_ref:
            ff = fg if space == "global" else fake_p.reshape(G, K, -1)
            pr = dm.progression(ff, prog_ref[space], prog_groups, prog_levels, prog_cycles, cli.prog_shuffle, cli.seed)
            jk[name][f"progression_{space}"] = (pr["ratio"], pr.pop("_loo_ratio"))
            jk[name][f"progression_{space}_corr"] = (pr["corr"], pr.pop("_loo_corr"))
            curves = {"model": pr.pop("_curve"), "gt": pr.pop("_curve_ref")}
            if space == "global":
                progression_plot[name] = curves["model"]
                progression_plot["GT"] = curves["gt"]
            out[f"progression_{space}"] = {k: (_r(v) if isinstance(v, dict) else v) for k, v in pr.items()}
            out[f"progression_{space}"]["curve"] = curves["model"]

        prs = [dm.prdc(real_g, fg[:, k], cli.prdc_k) for k in range(K)]
        out["prdc"] = {m: float(np.mean([p[m] for p in prs])) for m in prs[0]}
        out["prdc_degraded"] = True
        out["prdc_note"] = report["prdc_note"]
        if _circular_eval(name, r["path"], r["meta"], model_id):
            print(
                "WARNING: Circular evaluation detected (oracle_dino uses DINOv3 features as condition)",
                flush=True,
            )
            out["circular_eval_warning"] = True
            report["circular_eval_warning"] = True

        if train_g is not None:
            from scipy.stats import mannwhitneyu

            d_fake, nn_idx = dm.nn_distance(fake_g, train_g)
            q05 = float(np.quantile(d_ref, 0.05))
            order = np.argsort(d_fake)[:16]
            out["memorization"] = {
                "median_nn_fake_to_train": float(np.median(d_fake)),
                "median_nn_loco_train": float(np.median(d_ref)),
                "median_nn_test_real_to_train": float(np.median(d_test)),
                "frac_fake_below_loco_q05": float(np.mean(d_fake < q05)),
                "mannwhitney_p_fake_closer": float(mannwhitneyu(d_fake, d_ref, alternative="less").pvalue),
                "closest_pairs": [
                    {
                        "fake": f"{ref['filename'][i // K]}#k{i % K}",
                        "train": split_ids[cli.memo_split][int(nn_idx[i])],
                        "dist": float(d_fake[i]),
                    }
                    for i in order
                ],
            }
        report["models"][name] = out
        probe_txt = ""
        if "probe" in out:
            bits = []
            for t, v in out["probe"].items():
                bits.append(
                    f"{t} R²={v['r2']:.3f} slope={v['calib_slope']:.3f} bias={v['mean_bias']:+.4f} "
                    f"highDI={v['high_di_bias']:+.4f}"
                )
            probe_txt = "  probe " + " | ".join(bits)
        print(
            f"[kid] {name:>16s}  KID={out['kid_global']['value']:.5f}±{out['kid_global']['se']:.5f}  "
            f"patchKID={out.get('kid_patch', {}).get('value', float('nan')):.5f}  "
            f"softDI={out['kid_soft_di']['value']:.5f}±{out['kid_soft_di']['se']:.5f}{probe_txt}",
            flush=True,
        )
        for space in spaces:
            nb = out[f"nbr_{space}"]
            frac = nb["frac_of_ceiling"]
            print(
                f"[kid] {'':>16s}  nbr_{space}={nb['value']:+.4f}±{nb['se']:.4f} "
                f"(shuffled {nb['shuffled_mean']:+.4f})  of ceiling={frac if frac is None else round(frac, 3)}",
                flush=True,
            )
        for space in prog_ref:
            pg = out[f"progression_{space}"]
            print(
                f"[kid] {'':>16s}  progression_{space}: ratio={pg['ratio']['value']:.3f}±{pg['ratio']['se']:.3f} "
                f"corr={pg['corr']['value']:+.3f} monotone={pg['monotone']['value']:+.2f} "
                f"(shuffled ratio {pg['shuffled_ratio']:.3f}, corr {pg['shuffled_corr']:+.2f}, monotone {pg['shuffled_monotone']:+.2f})",
                flush=True,
            )
        if "interference" in out:
            it = out["interference"]
            print(f"[kid] {'':>16s}  interference flagged={it['frac_images_flagged']:.3f} "
                  f"(mean {it['mean_area_px_when_flagged']:.0f} px, removed={it['removed_before_scoring']})", flush=True)
        if "memorization" in out:
            mo = out["memorization"]
            print(
                f"[kid] {'':>16s}  NN→train median fake={mo['median_nn_fake_to_train']:.3f} "
                f"loco={mo['median_nn_loco_train']:.3f}  frac<loco q05={mo['frac_fake_below_loco_q05']:.3f}",
                flush=True,
            )

    for a, b in combinations(runs, 2):
        # PRDC stays out of this table (prdc_degraded).
        report["comparisons"][f"{a}_vs_{b}"] = {m: _r(dm.compare(jk[a][m], jk[b][m])) for m in jk[a] if m in jk[b]}

    if probes is not None:
        real_pred = {t: probes[t].predict(real_g) for t in probes}
        names = list(runs)
        payload = {
            "gt_di": ref_labels["DI"],
            "gt_rspall": ref_labels["R_spall"],
            "real_pred_di": real_pred["DI"],
            "real_pred_rspall": real_pred["R_spall"],
            "model_names": np.asarray(names),
            "fake_pred_di_mean": np.stack([report["models"][n]["_probe_draws"]["mean"]["DI"] for n in names]),
            "fake_pred_di_std": np.stack([report["models"][n]["_probe_draws"]["std"]["DI"] for n in names]),
            "fake_pred_rspall_mean": np.stack([report["models"][n]["_probe_draws"]["mean"]["R_spall"] for n in names]),
            "fake_pred_rspall_std": np.stack([report["models"][n]["_probe_draws"]["std"]["R_spall"] for n in names]),
        }
        np.savez(os.path.join(cli.out_dir, "probe_predictions.npz"), **payload)
        for n in names:
            del report["models"][n]["_probe_draws"]

    for space in prog_ref:
        row = report["reference_rows"].get(f"progression_{space}_other_specimens")
        if row:
            print(
                f"[kid] progression_{space} other specimens (real, matched DI): ratio={row['ratio']['value']:.3f} "
                f"corr={row['corr']['value']:+.3f}",
                flush=True,
            )
    if progression_plot.get("GT"):
        _plot_progression(progression_plot, report["reference_rows"], os.path.join(cli.out_dir, "progression.png"))

    print(
        "[kid] PRDC excluded from the comparison table "
        "(single-specimen 5-NN radii are too small; scores are still stored).",
        flush=True,
    )
    out_path = os.path.join(cli.out_dir, "dinov3_kid.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[kid] real halves KID = {report['reference_rows']['real_halves_kid']}", flush=True)
    for k, v in report["reference_rows"].items():
        if k.startswith("other_real") and k.endswith("_kid"):
            print(f"[kid] {k} = {v['value']:.5f}", flush=True)
    if "probe_on_heldout_reals" in report["reference_rows"]:
        print(f"[kid] probe on held-out reals (ceiling) = {report['reference_rows']['probe_on_heldout_reals']}", flush=True)
    for space in spaces:
        row = report["reference_rows"].get(f"nbr_{space}_ceiling_other_specimens")
        if row:
            print(f"[kid] nbr_{space} ceiling (other specimens, matched DI) = {row['value']:+.4f}±{row['se']:.4f}", flush=True)
    for pair, ms in report["comparisons"].items():
        print(f"[kid] {pair}: " + "  ".join(f"{m} Δ={v['diff']:+.4f} z={v['z']:+.1f}" for m, v in ms.items()), flush=True)
    print(f"[kid] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
