#!/usr/bin/env python3
"""Writes the eval-suite config of the OPA-Flow experiment (docs/redesign/phase1_opaflow.md).
Models: gn (and gn_facfg) reuse the phase-0 guidance samples; A / B / C from their result dirs, each also as
a FA-CFG variant whose samples come from scripts/eval/phase0_facfg_samples.py.
  python scripts/eval/make_opaflow_suite.py A=<result_dir> B=<result_dir> [C=<result_dir>] > cfg.json"""
import glob, json, sys

OUT = "results/eval_suite/phase1_opaflow"
G = "results/eval_suite/phase0_guidance"
GN_DIR = "results/DamagePatternGeneration/CFMADM/1024-dadaGNsplit"
models = [
    {"name": "gn", "npz": {s: f"{G}/{s}/samples/gn.npz" for s in ("test", "val")}},
    {"name": "gn_facfg", "npz": {s: f"{G}/{s}/samples/gn_facfg.npz" for s in ("test", "val")}},
]
for kv in sys.argv[1:]:
    name, d = kv.split("=", 1)
    ck = sorted(glob.glob(f"{d}/*/*/checkpoints/last.ckpt"))[-1]
    models.append({"name": name, "result_dir": d, "checkpoint": ck})
    models.append({"name": f"{name}_facfg", "npz": {s: f"{OUT}/{s}/samples/{name}_facfg.npz" for s in ("test", "val")}})
print(json.dumps({"out_dir": OUT, "splits": ["test", "val"], "k": 8, "baseline": "gn",
                  "dino_model": "facebook/dinov3-vits16-pretrain-lvd1689m", "extra_dino_models": ["facebook/dinov2-small"],
                  "remove_interference": False, "models": models}, indent=1))
