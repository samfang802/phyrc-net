#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=2
#PBS -l place=shared
#PBS -o log/out.out
#PBS -e log/err.out
#PBS -N TEST_DINOV3_VAE_REC

set -euo pipefail

cd "${PBS_O_WORKDIR:-$HOME/intership-project}"

set +u
source ~/.bashrc
set -u
conda activate internship

module load cuda-12.4

echo "[test_dinov3_vae_rec] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Re-login Hugging Face when token exists.
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"
HF_TOKEN_VALUE="${HF_TOKEN:-}"
if [[ -z "${HF_TOKEN_VALUE}" && -f "${HF_TOKEN_FILE}" ]]; then
    HF_TOKEN_VALUE="$(tr -d '\r\n' < "${HF_TOKEN_FILE}")"
fi
if [[ -n "${HF_TOKEN_VALUE}" ]]; then
    HF_TOKEN="${HF_TOKEN_VALUE}" python3 - <<'PY'
import os
from huggingface_hub import login

token = os.environ["HF_TOKEN"].strip()
login(token=token, add_to_git_credential=False)
print("[test_dinov3_vae_rec] Hugging Face login refreshed.")
PY
else
    echo "[test_dinov3_vae_rec] HF token not found; continuing without explicit login."
fi

# Download/check DINOv3 and SD3 VAE from HF before reconstruction test.
DINO_MODEL_ID="${DINO_MODEL_ID:-facebook/dinov3-vits16-pretrain-lvd1689m}"
SD3_MODEL_ID="${SD3_MODEL_ID:-stabilityai/stable-diffusion-3-medium-diffusers}"
SD3_VAE_SUBFOLDER="${SD3_VAE_SUBFOLDER:-vae}"

DINO_MODEL_ID="${DINO_MODEL_ID}" SD3_MODEL_ID="${SD3_MODEL_ID}" SD3_VAE_SUBFOLDER="${SD3_VAE_SUBFOLDER}" python3 - <<'PY'
import os
from transformers import AutoImageProcessor, AutoModel
from diffusers import AutoencoderKL

dino_model_id = os.environ["DINO_MODEL_ID"]
sd3_model_id = os.environ["SD3_MODEL_ID"]
sd3_vae_subfolder = os.environ["SD3_VAE_SUBFOLDER"]

print(f"[test_dinov3_vae_rec] Checking DINOv3: {dino_model_id}")
_ = AutoImageProcessor.from_pretrained(dino_model_id)
_ = AutoModel.from_pretrained(dino_model_id)
print("[test_dinov3_vae_rec] DINOv3 load OK")

print(f"[test_dinov3_vae_rec] Checking SD3 VAE: {sd3_model_id}/{sd3_vae_subfolder}")
_ = AutoencoderKL.from_pretrained(sd3_model_id, subfolder=sd3_vae_subfolder)
print("[test_dinov3_vae_rec] SD3 VAE load OK")
PY

# DINOv3 + VAE crack reconstruction test script.
IMAGE_PATH="${IMAGE_PATH:-Data/DamagePattern/images/R307_NW_01000_4.jpg}"
OUTPUT_PATH="${OUTPUT_PATH:-results/ImageReconstruction/dinov3_vae_reconstruction_result.png}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"

python3 -u test_dinov3_vae_cracks.py \
    --image_path "${IMAGE_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --dino_model_id "${DINO_MODEL_ID}" \
    --sd3_model_id "${SD3_MODEL_ID}" \
    --vae_subfolder "${SD3_VAE_SUBFOLDER}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}"
