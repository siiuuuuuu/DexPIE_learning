#!/usr/bin/env bash
set -euo pipefail

# Examples:
# DATASET may be either a legacy Zarr directory or a converted memmap dataset root.
#   bash scripts/compute_GAE.sh /path/to/latest.ckpt DATASET
#   bash scripts/compute_GAE.sh /path/to/latest.ckpt DATASET DexPIE Recap-image 0 "[30,40,10,50,20]" advantage_quantiles.json advantage 0.99 1.2 1.0

if (( $# < 2 )); then
    echo "Usage: $0 CRITIC_CKPT DATASET [CONFIG] [TASK] [GPU_ID] [TOP_PERCENTAGES] [OUTPUT_FILENAME] [ADVANTAGE_KEY] [GAE_LAMBDA] [WINDOW_MULTIPLIER] [GAE_GAMMA]" >&2
    exit 2
fi

critic_ckpt=$1
dataset_path=$2
config_name=${3:-DexPIE}
task_name=${4:-Recap-image}
gpu_id=${5:-${GPU_ID:-0}}
top_percentages=${6:-'[30,40,10,50,20]'}
output_filename=${7:-advantage_quantiles.json}
advantage_key=${8:-advantage}
gae_lambda=${9:-0.99}
gae_window_multiplier=${10:-1.2}
gae_gamma=${11:-1.0}
python_bin=${PYTHON:-python}

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mconfig: ${config_name}, task: ${task_name}\033[0m"
echo -e "\033[33mcritic ckpt: ${critic_ckpt}\033[0m"
echo -e "\033[33mdataset: ${dataset_path}\033[0m"
echo -e "\033[33mtop percentages: ${top_percentages}\033[0m"
echo -e "\033[33madvantage output: data/${advantage_key}\033[0m"
echo -e "\033[33mquantile output: ${output_filename}\033[0m"
echo -e "\033[33mGAE gamma: ${gae_gamma}, lambda: ${gae_lambda}\033[0m"
echo -e "\033[33mGAE window multiplier: ${gae_window_multiplier}\033[0m"

cd DexPIE

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

"${python_bin}" compute_GAE.py --config-name="${config_name}" \
    "task=${task_name}" \
    "training.device=cuda:0" \
    "value_critic_ckpt_path=${critic_ckpt}" \
    "task.dataset.zarr_path=${dataset_path}" \
    "+advantage_top_percentages=${top_percentages}" \
    "+advantage_quantile_filename=${output_filename}" \
    "+advantage_key=${advantage_key}" \
    "+gae_lambda=${gae_lambda}" \
    "+gae_window_multiplier=${gae_window_multiplier}" \
    "+gae_gamma=${gae_gamma}"
