"""
Check Phase-1 (HystereticGRU) extractor bundle for Stage-3 CFM training.

Verifies:
  - args.json + best.ckpt presence
  - checkpoint hyper_parameters (if saved by Lightning)
  - architecture dims inferred from state_dict
  - optional dry-run model load

Usage:
  python scripts/eval/check_phase1_extractor.py
  python scripts/eval/check_phase1_extractor.py --extractor_path ./phase1_extractor
  python scripts/eval/check_phase1_extractor.py --write_args_json  # create args.json from ckpt if missing
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys
from typing import Any

import torch


DEFAULT_CANDIDATES = [
    "phase1_extractor",
    ".",
    "results/HystereticPrediction/HystereticGRU/2024-06-12_18-13-47",
    "results/HystereticPrediction/HystereticGRU/2025-08-01_13-28-08",
]

# job/HysteresisPrediction/train_hysteresis.sh defaults (formerly train_GRU.sh) (used when ckpt has no hyper_parameters)
AUTHOR_DEFAULTS = {
    "architecture": "HystereticGRU",
    "hidden_state_dim": 512,
    "layers": 4,
    "MLP_activation": "softsign",
    "bidirectional": True,
    "simulate": True,
    "scale": True,
    "data_size": "big",
    "lr": 1e-4,
    "epoch": 5000,
    "batch_size": 8,
    "num_workers": 2,
    "maximum": 1.0,
    "data_type": {},
    "data_version": 1,
}


def _first_match(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern, recursive=True))
    return matches[0] if matches else None


def find_extractor_files(extractor_path: str) -> dict[str, str | None]:
    return {
        "dir": extractor_path,
        "args_json": _first_match(os.path.join(extractor_path, "**", "args.json")),
        "best_ckpt": _first_match(os.path.join(extractor_path, "**", "best.ckpt")),
    }


def load_checkpoint(ckpt_path: str) -> dict[str, Any]:
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def infer_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    inferred: dict[str, Any] = {}

    gru_keys = [k for k in state_dict if k.startswith("L.gru.weight_ih_l")]
    lstm_keys = [k for k in state_dict if k.startswith("L.lstm.weight_ih_l")]
    if gru_keys:
        inferred["architecture"] = "HystereticGRU"
        prefix = "L.gru."
    elif lstm_keys:
        inferred["architecture"] = "HystereticLSTM"
        prefix = "L.lstm."
    else:
        inferred["architecture"] = "unknown"
        return inferred

    layer_ids = sorted(
        {
            int(k.split("weight_ih_l")[1].split("_")[0])
            for k in state_dict
            if k.startswith(f"{prefix}weight_ih_l") and "_reverse" not in k
        }
    )
    inferred["layers"] = len(layer_ids)

    w_ih = state_dict[f"{prefix}weight_ih_l0"]
    # GRU/LSTM input-hidden: [gate_factor * hidden, input_size]
    gate_factor = 3 if "gru" in prefix else 4
    inferred["hidden_state_dim"] = int(w_ih.shape[0] // gate_factor)
    inferred["signal_feat"] = int(w_ih.shape[1])

    inferred["bidirectional"] = any("_reverse" in k for k in state_dict if k.startswith(prefix))

    p0 = state_dict.get("P.fcn.0.weight")
    if p0 is not None:
        in_dim = int(p0.shape[1])
        expected = inferred["hidden_state_dim"] * (2 if inferred["bidirectional"] else 1)
        if in_dim == expected:
            inferred["MLP_activation"] = "(unknown — check args.json)"
        else:
            inferred["MLP_activation"] = f"(mismatch: P expects in_dim={in_dim}, expected {expected})"

    return inferred


def merge_args(
    args_json: dict[str, Any] | None,
    hyper_parameters: dict[str, Any] | None,
    inferred: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(AUTHOR_DEFAULTS)
    if hyper_parameters:
        merged.update({k: v for k, v in hyper_parameters.items() if v is not None})
    merged.update(inferred)
    if args_json:
        merged.update(args_json)
    return merged


def try_load_model(args_dict: dict[str, Any], ckpt_path: str) -> tuple[bool, str]:
    architecture = args_dict.get("architecture")
    if architecture not in ("HystereticGRU", "HystereticLSTM"):
        return False, f"Unsupported architecture: {architecture}"

    try:
        model_cls = getattr(importlib.import_module("src.model.HystereticPrediction"), architecture)
        model = model_cls(**args_dict)
        ckpt = load_checkpoint(ckpt_path)
        status = model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 32, int(args_dict["signal_feat"]))
            y = model(x)
        return True, (
            f"load_state_dict OK ({status}); "
            f"forward OK output_shape={tuple(y.shape)}"
        )
    except Exception as exc:
        return False, str(exc)


def write_args_json(path: str, args_dict: dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {k: v for k, v in args_dict.items() if k != "result_dir"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase-1 extractor bundle")
    parser.add_argument(
        "--extractor_path",
        type=str,
        default=None,
        help="Directory containing args.json and best.ckpt (searched recursively)",
    )
    parser.add_argument(
        "--write_args_json",
        action="store_true",
        help="If args.json is missing, write one next to best.ckpt using ckpt metadata",
    )
    args = parser.parse_args()

    candidates = [args.extractor_path] if args.extractor_path else DEFAULT_CANDIDATES
    candidates = [c for c in candidates if c]

    print_section("1) Search extractor bundle")
    chosen = None
    for cand in candidates:
        files = find_extractor_files(cand)
        print(f"\n[{cand}]")
        print(f"  args.json : {files['args_json'] or 'MISSING'}")
        print(f"  best.ckpt : {files['best_ckpt'] or 'MISSING'}")
        if files["best_ckpt"]:
            size_mb = os.path.getsize(files["best_ckpt"]) / 1024 / 1024
            print(f"  ckpt size : {size_mb:.2f} MB")
        if files["best_ckpt"] and not chosen:
            chosen = files

    if not chosen or not chosen["best_ckpt"]:
        print("\nFAIL: no best.ckpt found in candidates.")
        return 1

    ckpt_path = chosen["best_ckpt"]
    args_path = chosen["args_json"]

    print_section("2) Inspect checkpoint")
    ckpt = load_checkpoint(ckpt_path)
    print("top-level keys:", list(ckpt.keys()))
    epoch = ckpt.get("epoch", "?")
    global_step = ckpt.get("global_step", "?")
    print(f"epoch={epoch}, global_step={global_step}")

    hyper_parameters = ckpt.get("hyper_parameters") or {}
    if hyper_parameters:
        print("\nhyper_parameters from ckpt:")
        for key in sorted(hyper_parameters):
            print(f"  {key}: {hyper_parameters[key]}")
    else:
        print("\nhyper_parameters: (not stored in this checkpoint)")

    state_dict = ckpt["state_dict"]
    print(f"\nstate_dict tensors: {len(state_dict)}")
    sample_keys = list(state_dict.keys())[:8]
    for key in sample_keys:
        print(f"  {key}: {tuple(state_dict[key].shape)}")

    print_section("3) Infer architecture from weights")
    inferred = infer_from_state_dict(state_dict)
    for key, value in inferred.items():
        print(f"  {key}: {value}")

    args_json_data = None
    if args_path:
        with open(args_path, "r", encoding="utf-8") as f:
            args_json_data = json.load(f)
        print_section("4) args.json contents")
        for key in sorted(args_json_data):
            print(f"  {key}: {args_json_data[key]}")
    else:
        print_section("4) args.json")
        print("  MISSING — Stage 3 train_mixed.py will fail without it.")

    merged = merge_args(args_json_data, hyper_parameters, inferred)
    print_section("5) Merged config used for load test")
    important = [
        "architecture",
        "signal_feat",
        "hidden_state_dim",
        "layers",
        "MLP_activation",
        "bidirectional",
        "simulate",
        "scale",
        "data_size",
    ]
    for key in important:
        if key in merged:
            print(f"  {key}: {merged[key]}")

    print_section("6) Dry-run model load")
    ok, message = try_load_model(merged, ckpt_path)
    print("PASS" if ok else "FAIL", ":", message)

    if not args_path and args.write_args_json:
        out_path = os.path.join(os.path.dirname(ckpt_path), "args.json")
        write_args_json(out_path, merged)
        print(f"\nWrote inferred args.json -> {out_path}")
    elif not args_path:
        print(
            "\nTip: re-run with --write_args_json to generate args.json "
            "next to best.ckpt (verify values before training)."
        )

    print_section("Summary")
    has_args = args_path is not None
    print(f"extractor dir : {chosen['dir']}")
    print(f"best.ckpt     : {ckpt_path}")
    print(f"args.json     : {'OK' if has_args else 'MISSING'}")
    print(f"model load    : {'OK' if ok else 'FAILED'}")
    if has_args and ok:
        print("\nReady for Stage 3: pass --extractor_path", chosen["dir"])
        return 0
    if ok and not has_args:
        print("\nAlmost ready: generate args.json with --write_args_json")
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
