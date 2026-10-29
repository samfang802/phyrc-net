#!/usr/bin/env python3
"""Print val/test metric table for results/phase1_c2/runs/*/metrics.json."""
import json, glob, os, sys
keys = ["area_bias", "area_mae", "area_bias_highDI", "area_bias_6", "area_bias_8", "iou_spall_frames", "iou_highDI",
        "pairwise_iou_K", "ncomp_gen", "largest_frac_gen", "nnIoU_gen_mean", "nnIoU_gen_frac_gt0.8", "score"]
pat = sys.argv[1] if len(sys.argv) > 1 else "*"
splits = sys.argv[2].split(",") if len(sys.argv) > 2 else ["val", "test"]
for sp in splits:
    print(f"== {sp}")
    print("run".ljust(24) + " ".join(k[:11].rjust(11) for k in keys))
    for f in sorted(glob.glob(f"results/phase1_c2/runs/{pat}/metrics.json")):
        r = json.load(open(f)); m = r.get(sp)
        if not m: continue
        print(os.path.basename(os.path.dirname(f)).ljust(24) + " ".join(f"{(m[k] if m[k] is not None else float('nan')):11.3f}" for k in keys))
