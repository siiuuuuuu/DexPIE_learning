#!/usr/bin/env bash
set -euo pipefail

# Examples:
# RTC policy expert-intervention collection script.

# bash scripts/RTC_expert_interve_collect.sh RTC_dp_224x224_r3m two-image 0310 DATASET OUTPUT_DIR
# bash scripts/RTC_expert_interve_collect.sh RTC_Recap Recap-image 0312 DATASET OUTPUT_DIR
# bash scripts/RTC_expert_interve_collect.sh DexPIE Recap-image 0507_task1_iter1 DATASET OUTPUT_DIR
#
# Optional diagnostics:
# RTC_OBS_LATENCY_STEPS=1 bash scripts/RTC_expert_interve_collect.sh DexPIE Recap-image 0507_task1_iter1 DATASET OUTPUT_DIR
# RTC_POLICY_MAX_TASK_LENGTH=300 bash scripts/RTC_expert_interve_collect.sh DexPIE Recap-image 0507_task1_iter1 DATASET OUTPUT_DIR

if (( $# < 3 )); then
    echo "Usage: $0 ALGORITHM TASK TAG [DATASET] [OUTPUT_DIR]" >&2
    exit 2
fi

dataset_path=${4:-${DATASET_PATH:-}}
output_dir=${5:-${DEXPIE_DATA_DIR:-}}
if [[ -z "${dataset_path}" ]]; then
    echo "Error: provide DATASET or set DATASET_PATH." >&2
    exit 2
fi
if [[ -z "${output_dir}" ]]; then
    echo "Error: provide OUTPUT_DIR or set DEXPIE_DATA_DIR." >&2
    exit 2
fi

wandb_mode=${WANDB_MODE:-offline}
debug=${DEBUG:-False}
save_ckpt=True
python_bin=${PYTHON:-python}

alg_name=${1}
task_name=${2}
addition_info=${3}
config_name=${alg_name}
if [ "${alg_name}" = "RTC_sigRecap" ]; then
    # RTC_sigRecap is the predecessor of DexPIE. Keep the old run_dir name so
    # existing checkpoints are found, but instantiate the current DexPIE code.
    config_name=DexPIE
fi
seed=0
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

gpu_id=${GPU_ID:-0}
config_file="DexPIE/dexpie/config/${config_name}.yaml"
horizon=$(awk -F: '/^horizon:/ {gsub(/#.*/, "", $2); gsub(/[[:space:]]/, "", $2); print $2; exit}' "${config_file}")
n_action_steps=$(awk -F: '/^n_action_steps:/ {gsub(/#.*/, "", $2); gsub(/[[:space:]]/, "", $2); print $2; exit}' "${config_file}")
max_latency_steps=$(awk -F: '/^[[:space:]]+max_latency_steps:/ {gsub(/#.*/, "", $2); gsub(/[[:space:]]/, "", $2); print $2; exit}' "${config_file}")


echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mconfig: ${config_name}, ckpt run: ${run_dir}\033[0m"
echo -e "\033[33mhorizon: ${horizon}, n_action_steps: ${n_action_steps}\033[0m"
echo -e "\033[33mmax_latency_steps: ${max_latency_steps}\033[0m"
echo -e "\033[33mRTC_OBS_LATENCY_STEPS: ${RTC_OBS_LATENCY_STEPS:-1}\033[0m"
echo -e "\033[33mRTC_ACTION_OFFSET_STEPS: ${RTC_ACTION_OFFSET_STEPS:-dataset_attr_or_1}\033[0m"
echo -e "\033[33mrollout output: ${output_dir}\033[0m"


cd DexPIE

if [[ -e /dev/ttyUSB0 ]]; then
    sudo chmod 666 /dev/ttyUSB0 || true
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
export DEXPIE_DATA_DIR=${output_dir}

"${python_bin}" RTC_expert_interve_collect.py --config-name="${config_name}.yaml" \
    "task=${task_name}" \
    "hydra.run.dir=${run_dir}" \
    "training.debug=${debug}" \
    "training.seed=${seed}" \
    "training.device=cuda:0" \
    "exp_name=${exp_name}" \
    "logging.mode=${wandb_mode}" \
    "checkpoint.save_ckpt=${save_ckpt}" \
    "task.dataset.zarr_path=${dataset_path}"
