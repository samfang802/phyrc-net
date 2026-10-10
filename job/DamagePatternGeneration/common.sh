# Shared cluster setup for DamagePatternGeneration jobs.
# Source this after the PBS headers, once the shell is in the repo:
#   cd "${PBS_O_WORKDIR:-$HOME/intership-project}"
#   source job/DamagePatternGeneration/common.sh
#   adm_env

adm_env() {
    set +u
    source ~/.bashrc
    set -u
    conda activate internship
    module load cuda-12.4
    USE_NICE=False
    NICE=10
    if [[ "${USE_NICE}" == "True" || "${USE_NICE}" == "true" || "${USE_NICE}" == "1" ]]; then
        NICE_CMD=(nice -n "${NICE}")
    else
        NICE_CMD=()
    fi
}
