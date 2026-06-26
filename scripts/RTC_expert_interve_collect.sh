# Examples:
# RTC policy expert-intervention collection script.

# bash scripts/RTC_expert_interve_collect.sh RTC_dp_224x224_r3m two-image 0310
# bash scripts/RTC_expert_interve_collect.sh RTC_Recap Recap-image 0312
# bash scripts/RTC_expert_interve_collect.sh RTC_sigRecap Recap-image 0507_task1_iter1

wandb_mode=offline
#dataset_path=/home/lrz/dp_data/train_data
dataset_path=/home/lrz/dp_data/zarr_task1


DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
addition_info=${3}
config_name=${alg_name}
if [ "${alg_name}" = "RTC_sigRecap" ]; then
    # RTC_sigRecap is the predecessor of DexPIE. Keep the old run_dir name so
    # existing checkpoints are found, but instantiate the current DexPIE code.
    config_name=DexPIE
fi
seed=0
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

gpu_id=0
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mconfig: ${config_name}, ckpt run: ${run_dir}\033[0m"


cd DexPIE

sudo chmod 666 /dev/ttyUSB0

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python RTC_expert_interve_collect.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            task.dataset.zarr_path=$dataset_path 
