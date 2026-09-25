#!/usr/bin/env bash
set -euo pipefail

# Two-way out-of-fold GAE labeling and read-only source merge.
#
# Critic A must have been trained on dataset A and labels dataset B.
# Critic B must have been trained on dataset B and labels dataset A.
# The output is a new memmap-format dataset; neither input is modified.
#
# Usage:
#   bash scripts/compute_crossfit_GAE_merge.sh \
#     CRITIC_A DATASET_A CRITIC_B DATASET_B OUTPUT_DATASET
#
# Optional positional arguments:
#   6  config name                (default: DexPIE)
#   7  task name                  (default: Recap-image)
#   8  GPU id                     (default: 0)
#   9  top percentages            (default: [30,40,10,50,20])
#   10 advantage key              (default: advantage_gae)
#   11 quantile filename          (default: advantage_quantiles_gae.json)
#   12 GAE lambda                 (default: 0.99)
#   13 GAE window multiplier      (default: 1.2)
#   14 GAE gamma                  (default: 1.0)
#   15 merge copy batch size      (default: 64)

if (( $# < 5 )); then
    echo "Usage: $0 CRITIC_A DATASET_A CRITIC_B DATASET_B OUTPUT_DATASET [CONFIG] [TASK] [GPU_ID] [TOP_PERCENTAGES] [ADVANTAGE_KEY] [QUANTILE_FILENAME] [GAE_LAMBDA] [GAE_WINDOW_MULTIPLIER] [GAE_GAMMA] [COPY_BATCH_SIZE]" >&2
    exit 2
fi

critic_a_ckpt=$1
dataset_a_path=$2
critic_b_ckpt=$3
dataset_b_path=$4
output_dataset_path=$5
config_name=${6:-DexPIE}
task_name=${7:-Recap-image}
gpu_id=${8:-0}
top_percentages=${9:-'[30,40,10,50,20]'}
advantage_key=${10:-advantage_gae}
output_filename=${11:-advantage_quantiles_gae.json}
gae_lambda=${12:-0.99}
gae_window_multiplier=${13:-1.2}
gae_gamma=${14:-1.0}
copy_batch_size=${15:-64}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)

echo -e "\033[33mcritic A trained on dataset A: ${critic_a_ckpt}\033[0m"
echo -e "\033[33mdataset A will be labeled by critic B: ${dataset_a_path}\033[0m"
echo -e "\033[33mcritic B trained on dataset B: ${critic_b_ckpt}\033[0m"
echo -e "\033[33mdataset B will be labeled by critic A: ${dataset_b_path}\033[0m"
echo -e "\033[33mnew merged memmap dataset: ${output_dataset_path}\033[0m"
echo -e "\033[33mGAE gamma=${gae_gamma}, lambda=${gae_lambda}, window multiplier=${gae_window_multiplier}\033[0m"

cd "${repo_root}/DexPIE"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

python compute_crossfit_GAE_merge.py --config-name="${config_name}" \
    "task=${task_name}" \
    "training.device=cuda:0" \
    "+crossfit.critic_a_ckpt_path=${critic_a_ckpt}" \
    "+crossfit.dataset_a_path=${dataset_a_path}" \
    "+crossfit.critic_b_ckpt_path=${critic_b_ckpt}" \
    "+crossfit.dataset_b_path=${dataset_b_path}" \
    "+crossfit.output_dataset_path=${output_dataset_path}" \
    "+advantage_top_percentages=${top_percentages}" \
    "+advantage_quantile_filename=${output_filename}" \
    "+advantage_key=${advantage_key}" \
    "+gae_lambda=${gae_lambda}" \
    "+gae_window_multiplier=${gae_window_multiplier}" \
    "+gae_gamma=${gae_gamma}" \
    "+crossfit_copy_batch_size=${copy_batch_size}"
