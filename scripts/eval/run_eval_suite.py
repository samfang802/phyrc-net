#!/usr/bin/env python3
"""The whole evaluation in one command: samples → interference removal → three evaluations → summary.

Driven by a JSON config (example: job/DamagePatternGeneration/eval_suite_models.json):

  {
    "out_dir": "results/eval_suite/my_run",
    "splits": ["test", "val"],
    "k": 8,
    "baseline": "gn",
    "dino_model": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "extra_dino_models": ["facebook/dinov2-small"],
    "models": [
      {"name": "gn", "result_dir": "results/.../1024-dadaGNsplit", "checkpoint": "results/.../last.ckpt"},
      {"name": "gn_shared", "result_dir": "...", "checkpoint": "...", "shared_noise": true},
      {"name": "gn_apg2", "result_dir": "...", "checkpoint": "...", "set": ["apg_guidance_scale=2.0"]},
      {"name": "cgan", "images": {"test": {"dir": "cGANckpt/CTR1", "suffix": "_test.jpg"}}},
      {"name": "other", "npz": {"test": "path/to/samples_test.npz"}}
    ]
  }

A model is one of
  - a CFM run of this repo (result_dir + checkpoint): generate_eval_samples.py; residual refiners
    are detected automatically; "shared_noise" and "set" are passed through;
  - a folder of saved images per split ("images"): images_to_npz.py;
  - a ready npz per split ("npz"), in the src/metrics/samples_io.py layout.
A model with no source for a split (e.g. no val images) is left out of that split.

Backbone: "dino_model" (default DINOv3 ViT-S/16, gated on Hugging Face: accept its licence and
`huggingface-cli login`) feeds the realism / condition metrics and the white-vs-DINO test. Set it to
"facebook/dinov2-small" (public) to run without DINOv3 access; the interference detector carries its
own backbone weights and needs no access either. "extra_dino_models" repeats the realism / condition
metrics with each listed backbone (kid_<name>/), shown as a robustness table in the summary.

Per split, under <out_dir>/<split>/:
  samples/<name>.npz         raw samples
  samples/<name>_clean.npz   interference removed           → damage texture, temporal
  samples/<name>_eq.npz      … plus the GT-mask placebo      → DINOv3 metrics
  kid/ texture/ temporal/    the three evaluations (kid/ = DINOv3; another backbone → kid_<name>/)
  summary.md                 the report (scripts/eval/summarize_eval.py)
and <out_dir>/summary.md links the splits. Every step is skipped when its output exists, so an
interrupted run continues where it stopped; --force redoes everything. Logs: <out_dir>/logs/.
GPU: whatever CUDA_VISIBLE_DEVICES is set to. Protocol and the meaning of each number: docs/eval/.

Examples:
  python scripts/eval/run_eval_suite.py --config job/DamagePatternGeneration/eval_suite_models.json
  python scripts/eval/run_eval_suite.py --config cfg.json --splits test --only summary
  python scripts/eval/run_eval_suite.py --config cfg.json --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL_VERSION = "eval-v1"
DEFAULT_DINO = "facebook/dinov3-vits16-pretrain-lvd1689m"
INPAINTER_CKPT = ("results/DamagePatternGeneration/CFMADM/selfcondnoDI/Opensees Based damage Patterns Generation/"
                  "vq9dirur/checkpoints/last.ckpt")
STEPS = ("samples", "clean", "kid", "texture", "temporal", "summary")
DEFAULTS = {
    "splits": ["test", "val"],
    "k": 8,
    "data_path": "Data/DamagePattern/id_list_val_R307.json",
    "image_dir": "Data/DamagePattern_inpaint_selfcondnoDI/images",
    "detector": "results/InterferenceDetector/v1/best.pt",
    "dino_model": DEFAULT_DINO,
    "extra_dino_models": [],
    "remove_interference": True,
    "batch_size": 32,
    "seed": 0,
}
# The other real column used as a reference row on each split.
OTHER_SPLIT = {"test": "val", "val": "test"}


def _parse_args():
    p = argparse.ArgumentParser(description="Run the full evaluation suite")
    p.add_argument("--config", required=True)
    p.add_argument("--splits", nargs="*", default=None, help="Override the config's splits")
    p.add_argument("--only", nargs="*", default=None, choices=STEPS, help="Run only these steps")
    p.add_argument("--force", action="store_true", help="Redo steps whose output already exists")
    p.add_argument("--dry_run", action="store_true", help="Print the commands without running them")
    return p.parse_args()


class Runner:
    def __init__(self, cfg, cli):
        self.cfg, self.cli = cfg, cli
        self.py = sys.executable
        self.log_dir = os.path.join(cfg["out_dir"], "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def want(self, step):
        return self.cli.only is None or step in self.cli.only

    def run(self, name, args, outputs):
        """Run `python args` unless every path in outputs exists (or --force)."""
        if not self.cli.force and outputs and all(os.path.exists(o) for o in outputs):
            print(f"[suite] skip {name} (done)", flush=True)
            return
        cmd = [self.py, "-u"] + [str(a) for a in args]
        print(f"[suite] {name}\n        {shlex.join(cmd)}", flush=True)
        if self.cli.dry_run:
            return
        log = os.path.join(self.log_dir, f"{name}.log")
        t0 = time.time()
        with open(log, "w") as f:
            ret = subprocess.run(cmd, cwd=_ROOT, stdout=f, stderr=subprocess.STDOUT)
        if ret.returncode != 0:
            with open(log) as f:
                tail = "".join(f.readlines()[-15:])
            raise SystemExit(f"[suite] {name} failed (exit {ret.returncode}); log {log}:\n{tail}")
        print(f"        done in {time.time() - t0:.0f} s", flush=True)


def _kid_dir(sd, model_id):
    return os.path.join(sd, "kid" if model_id == DEFAULT_DINO else f"kid_{model_id.split('/')[-1]}")


def _detector_offline(path):
    """True when the detector can be built without Hugging Face (src/metrics/interference.py)."""
    if os.path.isfile(os.path.join(os.path.dirname(path), "backbone_config.json")):
        return True
    import torch

    return "backbone_config" in torch.load(path, map_location="cpu", weights_only=False)


def _check_assets(cfg):
    problems = []
    for key in ("data_path", "image_dir"):
        if not os.path.exists(os.path.join(_ROOT, cfg[key])):
            problems.append(f"{key}: {cfg[key]} not found (see docs/eval/README.md, 安裝)")
    if cfg["remove_interference"]:
        # detector, the CFM inpainter (inpaint_generated_samples.py defaults) and the GRU extractor it conditions on
        for what, path in (("detector", cfg["detector"]), ("inpainter", INPAINTER_CKPT),
                           ("GRU extractor", "phase1_extractor/best.ckpt")):
            if not os.path.isfile(os.path.join(_ROOT, path)):
                problems.append(f"{what}: {path} not found (eval_assets_v1.tar.gz, see docs/eval/README.md, 安裝; "
                                "or set \"remove_interference\": false)")
    from transformers import AutoConfig

    needed = [cfg["dino_model"], *cfg["extra_dino_models"]]
    det = os.path.join(_ROOT, cfg["detector"])
    if cfg["remove_interference"] and os.path.isfile(det) and not _detector_offline(det):
        needed.append(DEFAULT_DINO)
    for model_id in dict.fromkeys(needed):
        try:
            AutoConfig.from_pretrained(model_id)
        except Exception as e:  # gated model without access, no network and no cache, ...
            hint = ("accept the licence on its Hugging Face page and run `huggingface-cli login`, or set "
                    "\"dino_model\": \"facebook/dinov2-small\"" if "dinov3" in model_id else
                    "first run needs the network (HF_HUB_OFFLINE=0); an expired login token also blocks "
                    "public models (`huggingface-cli logout`)")
            problems.append(f"backbone {model_id} not loadable ({type(e).__name__}: {str(e)[:160]}); {hint}")
    for m in cfg["models"]:
        if "checkpoint" in m and not os.path.isfile(os.path.join(_ROOT, m["checkpoint"])):
            problems.append(f"model {m['name']}: checkpoint {m['checkpoint']} not found")
    if problems:
        raise SystemExit("[suite] cannot start:\n  - " + "\n  - ".join(problems))


def _sample_step(r, cfg, m, split, raw):
    if "checkpoint" in m:
        args = ["scripts/eval/generate_eval_samples.py", "--cfm_result_dir", m["result_dir"], "--cfm_checkpoint",
                m["checkpoint"], "--split", split, "--k", m.get("k", cfg["k"]), "--batch_size", cfg["batch_size"],
                "--seed", cfg["seed"], "--out", raw]
        if m.get("shared_noise"):
            args.append("--shared_noise")
        if m.get("set"):
            args += ["--set", *m["set"]]
        r.run(f"{split}_samples_{m['name']}", args, [raw])
        return raw
    if "images" in m and split in m["images"]:
        src = m["images"][split]
        args = ["scripts/eval/images_to_npz.py", "--image_dir", src["dir"], "--suffix", src.get("suffix", ".png"),
                "--k", src.get("k", 1), "--split", split, "--data_path", cfg["data_path"], "--out", raw]
        if src.get("clahe"):
            args.append("--clahe")
        r.run(f"{split}_images_{m['name']}", args, [raw])
        return raw
    if "npz" in m and split in m["npz"]:
        return m["npz"][split]
    return None


def main():
    cli = _parse_args()
    with open(cli.config) as f:
        cfg = {**DEFAULTS, **json.load(f)}
    if cli.splits:
        cfg["splits"] = cli.splits
    names = [m["name"] for m in cfg["models"]]
    if len(set(names)) != len(names):
        raise SystemExit("model names must be unique")
    os.chdir(_ROOT)
    if not cli.dry_run:
        _check_assets(cfg)
    r = Runner(cfg, cli)
    with open(os.path.join(cfg["out_dir"], "suite_config.json"), "w") as f:
        json.dump({"eval_version": EVAL_VERSION, **cfg}, f, indent=2)

    index = [f"# 評估結果（{EVAL_VERSION}）", "", f"設定檔：`{os.path.abspath(cli.config)}`", ""]
    for split in cfg["splits"]:
        sd = os.path.join(cfg["out_dir"], split)
        samples = {}
        for m in cfg["models"]:
            raw = os.path.join(sd, "samples", f"{m['name']}.npz")
            src = _sample_step(r, cfg, m, split, raw) if (r.want("samples") or not os.path.exists(raw)) else raw
            if src is None:
                print(f"[suite] {split}: no samples for {m['name']}, left out", flush=True)
                continue
            clean = os.path.join(sd, "samples", f"{m['name']}_clean.npz")
            eq = os.path.join(sd, "samples", f"{m['name']}_eq.npz")
            if cfg["remove_interference"] and r.want("clean"):
                base = ["scripts/eval/inpaint_generated_samples.py", "--samples", src, "--detector", cfg["detector"],
                        "--split", split]
                r.run(f"{split}_clean_{m['name']}", base + ["--out", clean], [clean])
                r.run(f"{split}_eq_{m['name']}", base + ["--add_gt_mask", "--out", eq], [eq])
            if not cfg["remove_interference"]:
                clean = eq = src
            samples[m["name"]] = (clean, eq)
        if not samples:
            continue
        common = ["--image_dir", cfg["image_dir"], "--no_clahe", "--data_path", cfg["data_path"]]
        eq_list = [f"{n}={p[1]}" for n, p in samples.items()]
        clean_list = [f"{n}={p[0]}" for n, p in samples.items()]
        tex_dir, tmp_dir = (os.path.join(sd, d) for d in ("texture", "temporal"))
        kid_dirs = {b: _kid_dir(sd, b) for b in dict.fromkeys([cfg["dino_model"], *cfg["extra_dino_models"]])}
        kid_dir = kid_dirs[cfg["dino_model"]]
        other = OTHER_SPLIT.get(split, "")
        if r.want("kid"):
            for b, d in kid_dirs.items():
                r.run(f"{split}_{os.path.basename(d)}", ["scripts/eval/eval_dinov3_kid.py", "--samples", *eq_list, *common,
                                                        "--dino_model", b, "--other_split", other if split == "test" else "",
                                                        "--out_dir", d], [os.path.join(d, "dinov3_kid.json")])
        if r.want("texture"):
            dino = ["--dino", "--dino_model_id", cfg["dino_model"]] if max(int(m.get("k", cfg["k"])) for m in cfg["models"]) > 1 else []
            r.run(f"{split}_texture", ["scripts/eval/eval_damage_texture.py", "--samples", *clean_list, *common,
                                       "--other_split", other, *dino, "--out_dir", tex_dir],
                  [os.path.join(tex_dir, "damage_texture.json")])
        if r.want("temporal"):
            r.run(f"{split}_temporal", ["scripts/eval/eval_temporal.py", "--samples", *clean_list, *common,
                                        "--out_dir", tmp_dir], [os.path.join(tmp_dir, "temporal.json")])
        if r.want("summary"):
            base = cfg.get("baseline") or next(iter(samples))
            extra = [d for b, d in kid_dirs.items() if d != kid_dir]
            r.run(f"{split}_summary", ["scripts/eval/summarize_eval.py", "--kid_dir", kid_dir,
                                       *(["--kid_extra_dir", *extra] if extra else []), "--texture_dir", tex_dir,
                                       "--temporal_dir", tmp_dir, "--baseline", base, "--title", split,
                                       "--out", os.path.join(sd, "summary.md")], [])
        index.append(f"- [{split}]({split}/summary.md)：{', '.join(samples)}")
    if not cli.dry_run:
        with open(os.path.join(cfg["out_dir"], "summary.md"), "w") as f:
            f.write("\n".join(index) + "\n")
        print(f"[suite] done → {os.path.join(cfg['out_dir'], 'summary.md')}", flush=True)


if __name__ == "__main__":
    main()
