# Examples:

#   bash scripts/train_policy.sh RTC_dp_224x224_r3m two-image 0310
#   bash scripts/train_policy.sh RTC_Recap Recap-image 0313
#   bash scripts/train_policy.sh DexPIE Recap-image 0319

#dataset_path=/home/ze/projects/DexPIE/training_data_example
#dataset_path=/home/lrz/dp_data/train_data
dataset_path=/home/lrz/dp_data/zarr_task_airpod_expertdata
DEBUG=False
wandb_mode=offline


alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=0
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

gpu_id=0
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    save_ckpt=False
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    save_ckpt=True
    echo -e "\033[33mTrain mode\033[0m"
fi


cd DexPIE

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            task.dataset.zarr_path=$dataset_path 
