# Examples:

#   bash scripts/deploy_policy.sh idp3 gr1_dex-3d 0913_example
#   bash scripts/deploy_policy.sh dp_224x224_r3m gr1_dex-image 0913_example
#   bash scripts/deploy_policy.sh flow_224x224_Dino gr1_dex-image 0114_example
#   bash scripts/deploy_policy.sh flow_224x224_transformer gr1_dex-image 0115

wandb_mode=online
#dataset_path=/home/ze/projects/Improved-3D-Diffusion-Policy/training_data_example
#dataset_path=/home/lrz/dp_data/train_data
dataset_path=/home/lrz/dp_data/train_yangfang_1_data


DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=0
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

gpu_id=0
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


cd Improved-3D-Diffusion-Policy

sudo chmod 666 /dev/ttyUSB0

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python expert_intervention_infer_collect.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            task.dataset.zarr_path=$dataset_path 
