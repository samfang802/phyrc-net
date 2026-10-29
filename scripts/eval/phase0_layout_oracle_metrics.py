#!/usr/bin/env python3
"""Phase-0 layout-oracle: spall area and fragmentation of generated images (docs/redesign/phase0_layout_oracle.md).

Reads the eval-suite sample files (samples/<name>.npz, src/metrics/samples_io.py) and measures spall on
every generated image with two measuring tools:
  seg   the pixel CNN of phase0_layout_oracle_spallseg.py (trained on train-split AnyLabel masks);
  mstar the coarse m* brightness rule (src/model/cfm/spall_mask.py::extent_mask, 16×16). Known flaws:
        misses bright spall faces, counts the dark pipe at the bottom.
GT is the AnyLabel polygon mask at 128 (Data/DamagePattern/anylabel_layout_sdf128.npz), and for scale
also the same tool applied to the real frame (`real` in the npz), so the tool's own bias is visible.

Per model and split, on the high-DI tertile (split DI ≥ 2/3 quantile, as eval_damage_texture.py) and on
all frames:
  area      mean spall fraction (mean over K, then over conditions)
  bias      area − AnyLabel area, paired per condition; SE = block jackknife (face × drift level)
  n_comp    connected components (8-conn, ≥ min_comp px) per image
  largest   largest component / total spall area, on images with ≥ min_area px of spall
  iou       IoU of the generated mask with the GT layout (does it follow the layout?)
  paired z  vs the baseline model (difference of per-condition values, block jackknife)

  python scripts/eval/phase0_layout_oracle_metrics.py --split test \
      --model gn=results/phase0_layout_oracle/eval/samples/test/gn.npz --model ...  --out out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import ndimage

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics.dino_mmd import frame_block  # noqa: E402

_spec_path = os.path.join(_ROOT, "scripts", "eval", "phase0_layout_oracle_spallseg.py")
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("_p0seg", _spec_path)
_seg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_seg)

_EIGHT = np.ones((3, 3), dtype=int)


def _components(mask, min_comp):
    lab, n = ndimage.label(mask, structure=_EIGHT)
    if n == 0:
        return 0, 0.0, 0
    sizes = np.bincount(lab.ravel())[1:]
    keep = sizes[sizes >= min_comp]
    total = int(sizes.sum())
    return int(len(keep)), float(sizes.max() / total) if total else float("nan"), total


def _shape_stats(masks, min_comp, min_area):
    """masks (..., H, W) bool → n_comp, largest ratio (nan when < min_area px), area."""
    flat = masks.reshape((-1,) + masks.shape[-2:])
    ncomp = np.empty(len(flat))
    largest = np.full(len(flat), np.nan)
    for i, m in enumerate(flat):
        n, lr, total = _components(m, min_comp)
        ncomp[i] = n
        if total >= min_area:
            largest[i] = lr
    shp = masks.shape[:-2]
    return ncomp.reshape(shp), largest.reshape(shp), masks.reshape(shp + (-1,)).mean(-1)


def _mstar(u8, device, grid=16):
    from src.model.cfm.spall_mask import extent_mask

    out = []
    for i in range(0, len(u8), 256):
        x = torch.from_numpy(np.asarray(u8[i : i + 256])).float().to(device)[:, None] / 127.5 - 1.0
        out.append(extent_mask(x.expand(-1, 3, -1, -1).contiguous(), grid).squeeze(1).cpu().numpy() > 0.5)
    return np.concatenate(out)


def _jk(x, blocks):
    """Mean with block-jackknife SE, ignoring nan conditions."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    x, blocks = x[ok], np.asarray(blocks)[ok]
    if len(x) == 0:
        return {"value": float("nan"), "se": float("nan"), "n": 0}
    labels = np.unique(blocks)
    if len(labels) < 2:
        return {"value": float(x.mean()), "se": float("nan"), "n": int(len(x))}
    loo = np.array([x[blocks != b].mean() for b in labels])
    n = len(labels)
    se = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))
    return {"value": float(x.mean()), "se": se, "n": int(len(x)), "n_blocks": int(n)}


def _nanmean_k(a):
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(a, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--model", action="append", required=True, help="name=path/to/samples.npz")
    p.add_argument("--baseline", default="gn")
    p.add_argument("--segmenter", default="results/phase0_layout_oracle/spallseg/best.pt")
    p.add_argument("--layout_cache", default="Data/DamagePattern/anylabel_layout_sdf128.npz")
    p.add_argument("--min_comp", type=int, default=16, help="component size counted in n_comp (px at 128)")
    p.add_argument("--min_area", type=int, default=50, help="largest-ratio needs this much spall (px)")
    p.add_argument("--out", required=True)
    cli = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, thr = _seg.load_segmenter(cli.segmenter, dev)

    lay = np.load(cli.layout_cache)
    lidx = {str(f): i for i, f in enumerate(lay["filename"])}

    models = dict(kv.split("=", 1) for kv in cli.model)
    first = np.load(next(iter(models.values())))
    filenames = [str(f) for f in first["filename"]]
    di = first["DI"].astype(np.float64)
    hi = di >= np.quantile(di, 2 / 3)
    blocks = np.array([frame_block(f) for f in filenames])
    lay_mask = np.asarray(lay["mask"])  # read once: NpzFile re-decompresses on every lay["mask"]
    gt_mask = np.stack([lay_mask[lidx[f]] for f in filenames]).astype(bool)
    real = first["real"]

    gt_nc, gt_lr, gt_area = _shape_stats(gt_mask, cli.min_comp, cli.min_area)
    seg_real = _seg.predict_masks(net, real, dev, thr)
    sr_nc, sr_lr, sr_area = _shape_stats(seg_real, cli.min_comp, cli.min_area)
    ms_real = _mstar(real, dev).reshape(len(real), -1).mean(1)

    per = {}  # name -> dict of per-condition arrays
    for name, path in models.items():
        z = np.load(path)
        if [str(f) for f in z["filename"]] != filenames:
            raise ValueError(f"{name}: row order differs from the first npz")
        fake = z["fake"]  # (N, K, H, W)
        n, k = fake.shape[:2]
        flat = fake.reshape(n * k, *fake.shape[2:])
        seg = _seg.predict_masks(net, flat, dev, thr).reshape(n, k, *fake.shape[2:])
        nc, lr, area = _shape_stats(seg, cli.min_comp, cli.min_area)
        inter = (seg & gt_mask[:, None]).reshape(n, k, -1).sum(-1)
        union = (seg | gt_mask[:, None]).reshape(n, k, -1).sum(-1)
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        ms = _mstar(flat, dev).reshape(n, k, -1).mean(-1)
        per[name] = {
            "seg_area": area.mean(1),
            "seg_bias": area.mean(1) - gt_area,
            "n_comp": nc.mean(1),
            "largest": _nanmean_k(lr),
            "iou": _nanmean_k(np.where(gt_mask.reshape(n, -1).any(1)[:, None], iou, np.nan)),
            "mstar_area": ms.mean(1),
            "mstar_bias_vs_real": ms.mean(1) - ms_real,
        }

    def summarize(mask):
        out = {
            "n_conditions": int(mask.sum()),
            "gt": {
                "anylabel_area": _jk(gt_area[mask], blocks[mask]),
                "anylabel_n_comp": _jk(gt_nc[mask], blocks[mask]),
                "anylabel_largest": _jk(gt_lr[mask], blocks[mask]),
                "seg_on_real_area": _jk(sr_area[mask], blocks[mask]),
                "seg_on_real_bias": _jk(sr_area[mask] - gt_area[mask], blocks[mask]),
                "seg_on_real_n_comp": _jk(sr_nc[mask], blocks[mask]),
                "seg_on_real_largest": _jk(sr_lr[mask], blocks[mask]),
                "mstar_on_real_area": _jk(ms_real[mask], blocks[mask]),
            },
            "models": {},
        }
        for name, d in per.items():
            row = {key: _jk(v[mask], blocks[mask]) for key, v in d.items()}
            row["largest_minus_gt"] = _jk(d["largest"][mask] - gt_lr[mask], blocks[mask])
            row["n_comp_minus_gt"] = _jk(d["n_comp"][mask] - gt_nc[mask], blocks[mask])
            if cli.baseline in per and name != cli.baseline:
                b = per[cli.baseline]
                for key in ("seg_area", "n_comp", "largest", "iou", "mstar_area"):
                    diff = _jk(d[key][mask] - b[key][mask], blocks[mask])
                    diff["z"] = diff["value"] / diff["se"] if diff["se"] and diff["se"] > 0 else float("nan")
                    row[f"diff_vs_{cli.baseline}_{key}"] = diff
            out["models"][name] = row
        return out

    # Per drift level (filename code, e.g. 06000 = 6 %): where along the loading the bias sits.
    drift = np.array([os.path.basename(f).split("_")[2] for f in filenames])
    by_drift = {}
    for d in sorted(set(drift)):
        m = drift == d
        by_drift[d] = {
            "n": int(m.sum()),
            "gt_area": float(gt_area[m].mean()),
            "seg_on_real_area": float(sr_area[m].mean()),
            **{f"{name}_area": float(v["seg_area"][m].mean()) for name, v in per.items()},
            **{f"{name}_largest": float(np.nanmean(v["largest"][m])) if np.isfinite(v["largest"][m]).any() else float("nan")
               for name, v in per.items()},
            "gt_largest": float(np.nanmean(gt_lr[m])) if np.isfinite(gt_lr[m]).any() else float("nan"),
        }

    result = {
        "split": cli.split,
        "by_drift": by_drift,
        "segmenter": os.path.abspath(cli.segmenter),
        "threshold": thr,
        "min_comp": cli.min_comp,
        "min_area": cli.min_area,
        "di_high_edge": float(np.quantile(di, 2 / 3)),
        "high_di": summarize(hi),
        "all": summarize(np.ones_like(hi)),
        "models": models,
    }
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    with open(cli.out, "w") as f:
        json.dump(result, f, indent=2)

    h = result["high_di"]
    g = h["gt"]
    print(f"[{cli.split}] high-DI n={h['n_conditions']}  GT anylabel area {g['anylabel_area']['value']:.3f}  "
          f"n_comp {g['anylabel_n_comp']['value']:.2f}  largest {g['anylabel_largest']['value']:.3f}  | "
          f"seg on real: area {g['seg_on_real_area']['value']:.3f} (bias {g['seg_on_real_bias']['value']:+.3f}) "
          f"largest {g['seg_on_real_largest']['value']:.3f}")
    for name, r in h["models"].items():
        print(f"  {name:18s} area {r['seg_area']['value']:.3f}  bias {r['seg_bias']['value']:+.3f}±{r['seg_bias']['se']:.3f}  "
              f"n_comp {r['n_comp']['value']:.2f}  largest {r['largest']['value']:.3f} (Δgt {r['largest_minus_gt']['value']:+.3f})  "
              f"IoU {r['iou']['value']:.3f}  m* {r['mstar_area']['value']:.3f}")
    print("  drift  n   GT    " + "  ".join(f"{n[:12]:>12s}" for n in per))
    for d, r in by_drift.items():
        print(f"  {d}  {r['n']:3d}  {r['gt_area']:.3f}  " + "  ".join(f"{r[n + '_area']:12.3f}" for n in per))
    print(f"→ {cli.out}")


if __name__ == "__main__":
    main()
