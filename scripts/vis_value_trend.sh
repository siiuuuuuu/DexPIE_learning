# Examples:
#
#   bash scripts/vis_value_trend.sh
#   bash scripts/vis_value_trend.sh /path/to/latest.ckpt
#   bash scripts/vis_value_trend.sh /path/to/latest.ckpt /path/to/dataset.zarr 5 42 visualizations/value_trend.png 64
#   bash scripts/vis_value_trend.sh /path/to/latest.ckpt /path/to/dataset.zarr 5 42 visualizations/value_trend.png 64 30 50
#/home/lrz/project/DexPIE/DexPIE/data/outputs/value_image-critic-0322_seed0/checkpoints/latest.ckpt
critic_ckpt=${1:-/home/lrz/project/DexPIE/DexPIE/data/outputs/value_image-critic-0410_task2_iter1_seed0/checkpoints/latest.ckpt}
zarr_path=${2:-/home/lrz/dp_data/task2_Recap_iter1}
num_trajectories=${3:-2}
seed=${4:-8}
output_path=${5:-visualizations/critic_value_random_trajectories.png}
chunk_size=${6:-128}
episode_start=${7:-54}
episode_end=${8:-58}


cd DexPIE

cmd=(
    python visualize_critic_values.py
    --critic_ckpt "${critic_ckpt}"
    --num_trajectories "${num_trajectories}"
    --seed "${seed}"
    --output "${output_path}"
    --chunk_size "${chunk_size}"
)

if [ -n "${zarr_path}" ]; then
    cmd+=(--zarr_path "${zarr_path}")
fi

if [ -n "${episode_start}" ]; then
    cmd+=(--episode_start "${episode_start}")
fi

if [ -n "${episode_end}" ]; then
    cmd+=(--episode_end "${episode_end}")
fi

"${cmd[@]}"
