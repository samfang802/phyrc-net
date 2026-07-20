#!/bin/bash
set -euo pipefail

cd "${PBS_O_WORKDIR:-$HOME/intership-project}"

set +u
source ~/.bashrc
set -u
conda activate internship

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"
HF_TOKEN_VALUE="${HF_TOKEN:-}"

if [[ -z "${HF_TOKEN_VALUE}" && -f "${HF_TOKEN_FILE}" ]]; then
    HF_TOKEN_VALUE="$(tr -d '\r\n' < "${HF_TOKEN_FILE}")"
fi

if [[ -z "${HF_TOKEN_VALUE}" ]]; then
    echo "[hf_login] Missing HF token."
    echo "[hf_login] Export token first:"
    echo "  export HF_TOKEN=hf_xxx"
    echo "[hf_login] Or place it at: ${HF_TOKEN_FILE}"
    exit 1
fi

HF_TOKEN="${HF_TOKEN_VALUE}" python3 - <<'PY'
import os
from huggingface_hub import login

token = os.environ["HF_TOKEN"].strip()
login(token=token, add_to_git_credential=False)
print("[hf_login] Hugging Face login successful.")
PY
