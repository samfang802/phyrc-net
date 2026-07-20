#!/bin/bash
set -euo pipefail

cd "${1:-$PWD}"

set +u
source ~/.bashrc
set -u
conda activate internship

EXTRACTOR_PATH="${EXTRACTOR_PATH:-./phase1_extractor}"
HF_REPO_ID="${HF_REPO_ID:-samfang802/rc-column-HysGRU-feature-extractor}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"

mkdir -p "${EXTRACTOR_PATH}"
rm -f "${EXTRACTOR_PATH}/best.ckpt" "${EXTRACTOR_PATH}/args.json"

if [[ -z "${HF_TOKEN:-}" && -f "${HF_TOKEN_FILE}" ]]; then
    HF_TOKEN="$(tr -d '\r\n' < "${HF_TOKEN_FILE}")"
fi

echo "[prepare_extractor] Downloading best.ckpt from ${HF_REPO_ID}"
HF_REPO_ID="${HF_REPO_ID}" EXTRACTOR_PATH="${EXTRACTOR_PATH}" HF_TOKEN="${HF_TOKEN:-}" python3 - <<'PY'
import json
import os
import shutil
import torch
from huggingface_hub import hf_hub_download

repo_id = os.environ["HF_REPO_ID"]
extractor_path = os.environ["EXTRACTOR_PATH"]
token = os.environ.get("HF_TOKEN", "").strip() or None

def download_best_ckpt(use_token):
    return hf_hub_download(
        repo_id=repo_id,
        filename="best.ckpt",
        repo_type="model",
        token=use_token,
    )

try:
    ckpt_src = download_best_ckpt(token)
except Exception as first_exc:
    if token:
        print(f"[prepare_extractor] Token download failed, retrying anonymously: {first_exc}")
        ckpt_src = download_best_ckpt(None)
    else:
        raise

best_ckpt_path = os.path.join(extractor_path, "best.ckpt")
shutil.copy2(ckpt_src, best_ckpt_path)
print(f"[prepare_extractor] Downloaded best.ckpt -> {best_ckpt_path}")

ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
hp = ckpt.get("hyper_parameters") or {}
if not hp:
    raise RuntimeError("Checkpoint has no hyper_parameters; cannot generate args.json")

args_payload = {
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
args_payload.update(hp)

sd = ckpt["state_dict"]
if "signal_feat" not in args_payload:
    if "L.gru.weight_ih_l0" in sd:
        args_payload["signal_feat"] = int(sd["L.gru.weight_ih_l0"].shape[1])
    elif "L.lstm.weight_ih_l0" in sd:
        args_payload["signal_feat"] = int(sd["L.lstm.weight_ih_l0"].shape[1])
    else:
        raise RuntimeError("Cannot infer signal_feat from checkpoint weights")

args_path = os.path.join(extractor_path, "args.json")
with open(args_path, "w", encoding="utf-8") as f:
    json.dump(args_payload, f, indent=2)
print(f"[prepare_extractor] Generated args.json -> {args_path}")
PY

echo "[prepare_extractor] Done. Files ready in ${EXTRACTOR_PATH}"
