# Examples:
#
#   bash scripts/vis_value_trend.sh
#   bash scripts/vis_value_trend.sh /path/to/latest.ckpt
#   bash scripts/vis_value_trend.sh /path/to/latest.ckpt /path/to/dataset.zarr 5 42 visualizations/value_trend.png 64

critic_ckpt=${1:-/home/lrz/project/Improved-3D-Diffusion-Policy/Improved-3D-Diffusion-Policy/data/outputs/value_image-critic-0311_seed0/checkpoints/latest.ckpt}
zarr_path=${2:-}
num_trajectories=${3:-5}
seed=${4:-0}
output_path=${5:-visualizations/critic_value_random_trajectories.png}
chunk_size=${6:-64}

cd Improved-3D-Diffusion-Policy

if [ -n "${zarr_path}" ]; then
    python visualize_critic_values.py \
        --critic_ckpt "${critic_ckpt}" \
        --zarr_path "${zarr_path}" \
        --num_trajectories "${num_trajectories}" \
        --seed "${seed}" \
        --output "${output_path}" \
        --chunk_size "${chunk_size}"
else
    python visualize_critic_values.py \
        --critic_ckpt "${critic_ckpt}" \
        --num_trajectories "${num_trajectories}" \
        --seed "${seed}" \
        --output "${output_path}" \
        --chunk_size "${chunk_size}"
fi
