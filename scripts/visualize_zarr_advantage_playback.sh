#!/usr/bin/env bash
set -euo pipefail

# Examples:
#
#   bash scripts/visualize_zarr_advantage_playback.sh
#   bash scripts/visualize_zarr_advantage_playback.sh /path/to/dataset.zarr
#   bash scripts/visualize_zarr_advantage_playback.sh /path/to/dataset.zarr 25 /path/to/advantage_quantiles.json 3
#   bash scripts/visualize_zarr_advantage_playback.sh /path/to/dataset.zarr 25 none 0 --start-frame 100

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"

python_bin="${PYTHON:-/home/lrz/miniconda3/envs/idp3/bin/python}"
dataset_path="${1:-/home/lrz/dp_data/task1_Recap_iter1}"
fps="${2:-25}"
quantiles_path="${3:-${dataset_path}/advantage_quantiles.json}"
episode="${4:-0}"

echo -e "\033[33mpython: ${python_bin}\033[0m"
echo -e "\033[33mdataset: ${dataset_path}\033[0m"
echo -e "\033[33mfps: ${fps}\033[0m"
echo -e "\033[33mepisode: ${episode}\033[0m"
if [[ -n "${quantiles_path}" && "${quantiles_path}" != "none" ]]; then
    echo -e "\033[33madvantage quantiles: ${quantiles_path}\033[0m"
else
    echo -e "\033[33madvantage quantiles: disabled\033[0m"
fi

cmd=(
    "${python_bin}"
    "${repo_dir}/visualize_zarr_advantage_playback.py"
    "${dataset_path}"
    --fps "${fps}"
    --episode "${episode}"
)

if [[ -n "${quantiles_path}" && "${quantiles_path}" != "none" ]]; then
    cmd+=(--advantage-quantiles-json "${quantiles_path}")
fi

if [[ "$#" -gt 4 ]]; then
    cmd+=("${@:5}")
fi

exec "${cmd[@]}"
