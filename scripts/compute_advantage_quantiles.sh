# Examples:
#
#   bash scripts/compute_advantage_quantiles.sh
#   bash scripts/compute_advantage_quantiles.sh /path/to/latest.ckpt /path/to/dataset.zarr
#   bash scripts/compute_advantage_quantiles.sh /path/to/latest.ckpt /path/to/dataset.zarr DexPIE Recap-image 0 "[30,40,10,50,20]" advantage_quantiles.json

critic_ckpt=${1:-/home/lrz/project/DexPIE/DexPIE/data/outputs/value_image-critic-0507_task1_iter1_seed0/checkpoints/latest.ckpt}
dataset_path=${2:-/home/lrz/dp_data/task1_Recap_iter1}
config_name=${3:-DexPIE}
task_name=${4:-Recap-image}
gpu_id=${5:-0}
top_percentages=${6:-'[30,40,10,50,20]'}
output_filename=${7:-advantage_quantiles.json}

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mconfig: ${config_name}, task: ${task_name}\033[0m"
echo -e "\033[33mcritic ckpt: ${critic_ckpt}\033[0m"
echo -e "\033[33mdataset: ${dataset_path}\033[0m"
echo -e "\033[33mtop percentages: ${top_percentages}\033[0m"

cd DexPIE

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

python compute_advantage_quantiles.py --config-name="${config_name}" \
    "task=${task_name}" \
    "training.device=cuda:0" \
    "value_critic_ckpt_path=${critic_ckpt}" \
    "task.dataset.zarr_path=${dataset_path}" \
    "+advantage_top_percentages=${top_percentages}" \
    "+advantage_quantile_filename=${output_filename}"
