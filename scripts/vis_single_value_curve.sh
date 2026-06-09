#!/usr/bin/env bash
set -euo pipefail

# Examples:
#
#   bash scripts/vis_single_value_curve.sh
#   bash scripts/vis_single_value_curve.sh /path/to/latest.ckpt /path/to/dataset.zarr 12
#   bash scripts/vis_single_value_curve.sh /path/to/latest.ckpt /path/to/dataset.zarr 12 visualizations/traj12_value.png 64 120:180:red
#   bash scripts/vis_single_value_curve.sh /path/to/latest.ckpt /path/to/dataset.zarr 12 visualizations/traj12_value.png 64 120:180:red:wrong_action 250:290:orange:recover
#   LINE_COLOR="#4f97c7" HIGHLIGHT_ALPHA="0.12" bash scripts/vis_single_value_curve.sh "" "" "" "" "" 240:315:red:wrong_action 15:180:lightskyblue:stable_improve
#   FIG_DPI=320 bash scripts/vis_single_value_curve.sh
#0410_task2_iter1
critic_ckpt=${1:-/home/lrz/project/Improved-3D-Diffusion-Policy/Improved-3D-Diffusion-Policy/data/outputs/value_image-critic-0410_task2_iter1_seed0/checkpoints/latest.ckpt}
zarr_path=${2:-/home/lrz/dp_data/task2_Recap_iter1}
traj_idx=${3:-41}
output_path=${4:-visualizations/traj_spiecal_case_3_task2.png}
chunk_size=${5:-128}

# Optional style controls via environment variables (keep args backward-compatible)
# Example:
#   LINE_COLOR="#4f97c7" HIGHLIGHT_ALPHA="0.12" bash scripts/vis_single_value_curve.sh ...
line_color=${LINE_COLOR:-}
line_alpha=${LINE_ALPHA:-}
highlight_alpha=${HIGHLIGHT_ALPHA:-}
recolor_highlight_segment=${RECOLOR_HIGHLIGHT_SEGMENT:-}
fig_dpi=${FIG_DPI:-${DPI:-500}}
label_fontsize=${LABEL_FONTSIZE:-18}
tick_fontsize=${TICK_FONTSIZE:-15}
enable_stochastic_aug=${ENABLE_STOCHASTIC_AUG:-0}

# Remaining args from position 6 are highlight specs:
#   start:end[:color[:label]]
# e.g. 120:180:red:wrong_action
highlight_specs=("${@:6}")

cd Improved-3D-Diffusion-Policy

cmd=(
    python visualize_single_value_curve.py
    --critic_ckpt "${critic_ckpt}"
    --traj_idx "${traj_idx}"
    --output "${output_path}"
    --chunk_size "${chunk_size}"
    --label_fontsize "${label_fontsize}"
    --tick_fontsize "${tick_fontsize}"
)

if [ -n "${zarr_path}" ]; then
    cmd+=(--zarr_path "${zarr_path}")
fi

if [ -n "${line_color}" ]; then
    cmd+=(--line_color "${line_color}")
fi

if [ -n "${line_alpha}" ]; then
    cmd+=(--line_alpha "${line_alpha}")
fi

if [ -n "${highlight_alpha}" ]; then
    cmd+=(--highlight_alpha "${highlight_alpha}")
fi

if [ -n "${fig_dpi}" ]; then
    cmd+=(--dpi "${fig_dpi}")
fi

if [ "${recolor_highlight_segment}" = "1" ] || [ "${recolor_highlight_segment}" = "true" ]; then
    cmd+=(--recolor_highlight_segment)
fi

if [ "${enable_stochastic_aug}" = "1" ] || [ "${enable_stochastic_aug}" = "true" ]; then
    cmd+=(--enable_stochastic_aug)
fi

for spec in "${highlight_specs[@]}"; do
    if [ -n "${spec}" ]; then
        cmd+=(--highlight "${spec}")
    fi
done

"${cmd[@]}"
