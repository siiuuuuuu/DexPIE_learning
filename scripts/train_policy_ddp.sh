#!/usr/bin/env bash
set -euo pipefail

# Examples:
#   bash scripts/train_policy_ddp.sh idp3 gr1_dex-3d 0323_ddp 0,1
#   bash scripts/train_policy_ddp.sh dp_224x224_r3m gr1_dex-image 0323_ddp 0,1,2,3
#   DATASET_PATH=/home/lrz/dp_data/zarr_task1 NPROC_PER_NODE=4 \
#     bash scripts/train_policy_ddp.sh RTC_sigRecap Recap-image 0323_ddp 0,1,2,3

if [ "$#" -lt 3 ]; then
    echo "Usage: bash scripts/train_policy_ddp.sh <alg_name> <task_name> <tag> [gpu_ids] [extra hydra overrides ...]"
    echo "Example: bash scripts/train_policy_ddp.sh idp3 gr1_dex-3d 0323_ddp 0,1"
    exit 1
fi

# dataset_path=/home/ze/projects/Improved-3D-Diffusion-Policy/training_data_example
# dataset_path=/home/lrz/dp_data/train_data
dataset_path=${DATASET_PATH:-/home/lrz/dp_data/zarr_task1}
DEBUG=${DEBUG:-False}
wandb_mode=${WANDB_MODE:-online}
seed=${SEED:-0}

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
gpu_ids=${4:-0,1}

# pass-through extra hydra overrides from the 5th arg onward
extra_args=()
if [ "$#" -ge 5 ]; then
    extra_args=("${@:5}")
fi

exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
default_nproc=${#gpu_array[@]}
nproc_per_node=${NPROC_PER_NODE:-${default_nproc}}

if [ "${nproc_per_node}" -lt 1 ]; then
    echo "[Error] nproc_per_node must be >= 1, got ${nproc_per_node}"
    exit 1
fi

if [ "${DEBUG}" = "True" ]; then
    save_ckpt=False
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    save_ckpt=True
    echo -e "\033[33mTrain mode\033[0m"
fi

# single-node defaults; can be overridden by env for multi-node
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
RDZV_BACKEND=${RDZV_BACKEND:-c10d}
RDZV_ENDPOINT=${RDZV_ENDPOINT:-127.0.0.1:${MASTER_PORT}}

echo -e "\033[33mGPU ids (to use): ${gpu_ids}\033[0m"
echo -e "\033[33mNPROC_PER_NODE: ${nproc_per_node}\033[0m"
echo -e "\033[33mRun dir: ${run_dir}\033[0m"

cd Improved-3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_ids}

if [ "${NNODES}" -eq 1 ]; then
    torchrun --standalone \
        --nnodes=1 \
        --nproc_per_node=${nproc_per_node} \
        --master_port=${MASTER_PORT} \
        train_ddp.py --config-name=${config_name}.yaml \
        task=${task_name} \
        hydra.run.dir=${run_dir} \
        training.debug=${DEBUG} \
        training.seed=${seed} \
        exp_name=${exp_name} \
        logging.mode=${wandb_mode} \
        checkpoint.save_ckpt=${save_ckpt} \
        task.dataset.zarr_path=${dataset_path} \
        "${extra_args[@]}"
else
    torchrun \
        --nnodes=${NNODES} \
        --node_rank=${NODE_RANK} \
        --nproc_per_node=${nproc_per_node} \
        --rdzv_backend=${RDZV_BACKEND} \
        --rdzv_endpoint=${RDZV_ENDPOINT} \
        train_ddp.py --config-name=${config_name}.yaml \
        task=${task_name} \
        hydra.run.dir=${run_dir} \
        training.debug=${DEBUG} \
        training.seed=${seed} \
        exp_name=${exp_name} \
        logging.mode=${wandb_mode} \
        checkpoint.save_ckpt=${save_ckpt} \
        task.dataset.zarr_path=${dataset_path} \
        "${extra_args[@]}"
fi
