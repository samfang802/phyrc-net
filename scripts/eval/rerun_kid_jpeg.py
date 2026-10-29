#!/usr/bin/env python3
"""Re-score finished eval suites with the JPEG-aligned DINOv3 metrics, reusing their samples.

The reference set is JPEG q75 and our samples are not (docs/eval/protocol.md). For every
<suite>/<split>/kid/dinov3_kid.json this reruns eval_dinov3_kid.py on the same npz files with the
same reference, R_spall labels and normalization, plus --jpeg NAME=Q for every model that is not
already JPEG (cGAN archives are skipped). Output goes next to the original:

  <suite>/<split>/kid_jpeg<Q>/dinov3_kid.json   (the original kid/ is left untouched)
  <suite>/<split>/summary_jpeg<Q>.md            (texture / temporal reused, they do not see JPEG)

  python scripts/eval/rerun_kid_jpeg.py results/eval_suite/v105 results/eval_suite/v11 ... [--dry_run]
"""
import argparse
import json
import os
import subprocess
import sys

OTHER_SPLIT = {"test": "val"}


def _jobs(suite, quality, skip):
    cfg_path = os.path.join(suite, "suite_config.json")
    cfg = json.load(open(cfg_path)) if os.path.isfile(cfg_path) else {}
    for split in ("test", "val"):
        sd = os.path.join(suite, split)
        src = os.path.join(sd, "kid", "dinov3_kid.json")
        if not os.path.isfile(src):
            continue
        d = json.load(open(src))
        models = d["models"]
        missing = [m["npz"] for m in models.values() if not os.path.isfile(m["npz"])]
        jpeg = [f"{n}={quality}" for n, m in models.items()
                if not m.get("jpeg_quality") and not any(s in n.lower() for s in skip)]
        out = os.path.join(sd, f"kid_jpeg{quality}")
        cmd = [sys.executable, "-u", "scripts/eval/eval_dinov3_kid.py",
               "--samples", *[f"{n}={m['npz']}" for n, m in models.items()],
               "--image_dir", d["image_dir"], "--no_clahe",
               "--data_path", cfg.get("data_path", "Data/DamagePattern/id_list_val_R307.json"),
               "--dino_model", d["dino_model_id"], "--normalize", d.get("normalize", "none"),
               # suites before 2026-10-25 have no r_spall_csv key and used the old labels
               "--r_spall_csv", d.get("r_spall_csv") or "old",
               "--other_split", OTHER_SPLIT.get(split, ""), "--out_dir", out]
        if jpeg:
            cmd[cmd.index("--out_dir"):cmd.index("--out_dir")] = ["--jpeg", *jpeg]
        summ = [sys.executable, "-u", "scripts/eval/summarize_eval.py", "--kid_dir", out,
                "--title", f"{split}（JPEG q{quality}）", "--out", os.path.join(sd, f"summary_jpeg{quality}.md")]
        for flag, sub in (("--texture_dir", "texture"), ("--temporal_dir", "temporal")):
            if os.path.isfile(os.path.join(sd, sub, f"{'damage_texture' if sub == 'texture' else sub}.json")):
                summ += [flag, os.path.join(sd, sub)]
        base = cfg.get("baseline")
        if base in models:
            summ += ["--baseline", base]
        yield f"{suite}:{split}", missing, os.path.join(out, "dinov3_kid.json"), cmd, summ


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("suites", nargs="+")
    p.add_argument("--quality", type=int, default=75)
    p.add_argument("--skip", nargs="*", default=["cgan"], help="name substrings that are already JPEG")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()
    failed = []
    for tag, missing, done, cmd, summ in (j for s in a.suites for j in _jobs(s, a.quality, a.skip)):
        if missing:
            print(f"[jpeg] {tag}: skip, missing npz {missing}", flush=True)
            failed.append(tag)
            continue
        steps = ([] if os.path.isfile(done) and not a.force else [cmd]) + [summ]
        for c in steps:
            print(f"[jpeg] {tag}: {' '.join(c[1:])}", flush=True)
            if not a.dry_run and subprocess.run(c).returncode != 0:
                failed.append(tag)
                break
    print(f"[jpeg] done; failed/skipped: {failed or 'none'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
