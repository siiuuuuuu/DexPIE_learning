#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_critic.sh critic value_image TAG DATASET
# DATASET may also be supplied through DATASET_PATH.

if (( $# < 3 )); then
    echo "Usage: $0 ALGORITHM TASK TAG [DATASET]" >&2
    exit 2
fi

dataset_path=${4:-${DATASET_PATH:-}}
if [[ -z "${dataset_path}" ]]; then
    echo "Error: provide DATASET or set DATASET_PATH." >&2
    exit 2
fi

debug=${DEBUG:-False}
wandb_mode=${WANDB_MODE:-offline}
python_bin=${PYTHON:-python}

alg_name=$1
task_name=$2
config_name=${alg_name}
addition_info=$3
seed=0
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

gpu_id=${GPU_ID:-0}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

if [[ "${debug}" == "True" ]]; then
    save_ckpt=False
    echo -e "\033[33mDebug mode!\033[0m"
else
    save_ckpt=True
    echo -e "\033[33mTrain mode\033[0m"
fi


cd DexPIE

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

"${python_bin}" train.py --config-name="${config_name}.yaml" \
    "task=${task_name}" \
    "hydra.run.dir=${run_dir}" \
    "training.debug=${debug}" \
    "training.seed=${seed}" \
    "training.device=cuda:0" \
    "exp_name=${exp_name}" \
    "logging.mode=${wandb_mode}" \
    "checkpoint.save_ckpt=${save_ckpt}" \
    "task.dataset.zarr_path=${dataset_path}"
