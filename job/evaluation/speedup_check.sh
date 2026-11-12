#!/bin/bash
#PBS -l select=1:ncpus=2:mem=15gb:gpu_id=2
#PBS -l place=shared
#PBS -o log/speedup_check.out
#PBS -e log/speedup_check.err
#PBS -N spdchk

# Checks the batched sampler / probe against the unbatched ones on rfmv2 6k (val), and times them.
#   1. generate_eval_samples: walks batched (b) vs one walk at a time (nb), same process -> same noise, ~same pixels
#   2. a second process (b2) -> identical to b (walk seeds no longer depend on Python's salted hash())
#   3. rollout probe batched over positions: roll_* rows vs the unbatched run (results/rollout_probe/rfmv2_6k_val.json)
#   qsub job/evaluation/speedup_check.sh
cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
source job/DamagePatternGeneration/common.sh
adm_env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
set -e
R=results/DamagePatternGeneration/CFMADM/2026-11-10_03-47-29_rfmv2
CK="${R}/checkpoint_step_006000.pt"
O=results/speedup_check
mkdir -p "${O}"
GEN=(python3 -u scripts/eval/generate_eval_samples.py --cfm_result_dir "${R}" --cfm_checkpoint "${CK}"
     --split val --k 8 --batch_size 32 --seed 0 --app_code mean
     --layouts "results/phase1_fiber_layout/layouts/grav_hspP_{split}.npz" --out "${O}/{split}_{name}.npz")

s=$(date +%s)
"${GEN[@]}" --variant "b:refiner_state_conf_value=0.1" --variant "nb:refiner_state_conf_value=0.1,refiner_walk_batch=0"
echo "[check] process 1 (setup + b + nb): $(( $(date +%s) - s )) s"
s=$(date +%s)
"${GEN[@]}" --variant "b2:refiner_state_conf_value=0.1"
echo "[check] process 2 (setup + b2): $(( $(date +%s) - s )) s"
python3 - "${O}" <<'PY'
import sys
import numpy as np
o = sys.argv[1]
f = {k: np.load(f"{o}/val_{k}.npz", allow_pickle=True)["fake"].astype(np.int16) for k in ("b", "nb", "b2")}
for a, b in (("b", "nb"), ("b", "b2")):
    d = np.abs(f[a] - f[b])
    print(f"[check] {a} vs {b}: identical {bool((d == 0).all())}, mean |diff| {d.mean():.4f}, "
          f"max {d.max()}, share of pixels off by > 2: {(d > 2).mean():.5f}")
PY

s=$(date +%s)
python3 -u scripts/eval/rollout_vs_resample_probe.py --model "rfmv2=${R}@${CK}" --split val --t_every 3 \
    --chunk 12 --out "${O}/probe_rfmv2_6k_val.json"
echo "[check] batched probe: $(( $(date +%s) - s )) s"
python3 - "${O}/probe_rfmv2_6k_val.json" results/rollout_probe/rfmv2_6k_val.json <<'PY'
import json, sys
new, old = (json.load(open(p)) for p in sys.argv[1:])
print("[check] roll rows, batched vs unbatched probe (keep / false / rmse_lo / dmem):")
for v in ("roll_1", "roll_2", "roll_4", "roll_8"):
    a, b = new["summary"][v], old["summary"][v]
    print(f"  {v}: " + "  ".join(f"{k} {a[k]:.4f} vs {b[k]:.4f}" for k in ("keep", "false", "rmse_lo", "dmem")))
print("[check] rf / ar rows now use", new["resample_solver"], "and ar memory from",
      "resampled" if new["ar_mem_resampled"] else "real", "frames (old: euler, real)")
for v in ("rf_0.1", "rf_0.5", "ar0.1_8", "ar0.3_8"):
    a, b = new["summary"][v], old["summary"][v]
    print(f"  {v}: " + "  ".join(f"{k} {a[k]:.4f} vs {b[k]:.4f}" for k in ("keep", "false", "rmse_lo", "d_mean")))
PY
