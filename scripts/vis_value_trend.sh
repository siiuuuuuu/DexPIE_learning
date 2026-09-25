# Examples:
#
#   bash scripts/vis_value_trend.sh
#   bash scripts/vis_value_trend.sh /path/to/latest.ckpt
# DATASET may be either a legacy Zarr directory or a converted memmap dataset root.
#   bash scripts/vis_value_trend.sh /path/to/latest.ckpt DATASET 5 42 visualizations/value_trend.png 64
#   bash scripts/vis_value_trend.sh /path/to/latest.ckpt DATASET 5 42 visualizations/value_trend.png 64 30 50
#/home/lrz/project/DexPIE/DexPIE/data/outputs/value_image-critic-0322_seed0/checkpoints/latest.ckpt
critic_ckpt=${1:-/home/lrz/project/DexPIE/DexPIE/data/outputs/value_image-critic-0802_task3_iter2_seed0/checkpoints/latest.ckpt}
dataset_path=${2:-/home/lrz/dp_data/test_task3_dataset}
num_trajectories=${3:-2}
seed=${4:-8}
output_path=${5:-visualizations/critic_value_random_trajectories_w_staged_dagger_0.png}
chunk_size=${6:-128}
episode_start=${7:-10}
episode_end=${8:-15}


cd DexPIE

cmd=(
    python visualize_critic_values.py
    --critic_ckpt "${critic_ckpt}"
    --num_trajectories "${num_trajectories}"
    --seed "${seed}"
    --output "${output_path}"
    --chunk_size "${chunk_size}"
)

if [ -n "${dataset_path}" ]; then
    cmd+=(--dataset_path "${dataset_path}")
fi

if [ -n "${episode_start}" ]; then
    cmd+=(--episode_start "${episode_start}")
fi

if [ -n "${episode_end}" ]; then
    cmd+=(--episode_end "${episode_end}")
fi

"${cmd[@]}"
