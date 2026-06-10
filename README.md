# DexPIE

This repo is for training and deployment of image-based diffusion policies for humanoid manipulation.


https://github.com/user-attachments/assets/97f6ff8c-45b3-497a-bb66-dd8b24e973b4

# Training

We provide a training data example in this [Google Drive](https://drive.google.com/file/d/1c-rDOe1CcJM8iUuT1ecXKjDYAn-afy2e/view?usp=sharing), so that you can try to train the model without collecting data.

More info:
- For the training machine, we use a local computer with an Nvidia RTX 4090 (24G memory). 
- For the deployment machine, we use the cpu of the onboard computer in Fourier GR1.
- Deployment examples use RGB observations from RealSense cameras.




## Installation

Install conda env and packages for both learning and deployment machines:

    conda remove -n dp --all
    conda create -n dp python=3.8
    conda activate dp
    
    # for cuda >= 12.1
    pip3 install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121
    # else, 
    # just install the torch version that matches your cuda version
    
    

    # install my visualizer
    cd third_party
    cd visualizer && pip install -e . && cd ..
    pip install kaleido plotly tyro termcolor h5py
    cd ..


    # install diffusion policy dependencies
    pip install --no-cache-dir wandb ipdb gpustat visdom notebook mediapy natsort scikit-video easydict pandas moviepy imageio imageio-ffmpeg termcolor av dm_control dill==0.3.5.1 hydra-core==1.2.0 einops==0.4.1 diffusers==0.11.1 zarr==2.12.0 numba==0.56.4 pygame==2.1.2 shapely==1.8.4 tensorboard==2.10.1 tensorboardx==2.5.1 absl-py==0.13.0 pyparsing==2.4.7 jupyterlab==3.0.14 scikit-image yapf==0.31.0 opencv-python==4.5.3.56 psutil av matplotlib setuptools==59.5.0

    cd DexPIE
    pip install -e .
    cd ..

    # install for diffusion policy if you want to use image-based policy
    pip install timm==0.9.7

    # install for r3m if you want to use image-based policy
    cd third_party/r3m
    pip install -e .
    cd ../..


[Install on Deployment Machine] Install realsense package for deploy:

    # first, install realsense driver
    # check this version for RealSenseL515: https://github.com/IntelRealSense/librealsense/releases/tag/v2.54.2

    # also install python api
    pip install pyrealsense2==2.54.2.5684

## Usage

We provide the training data example in [Google Drive](https://drive.google.com/file/d/1c-rDOe1CcJM8iUuT1ecXKjDYAn-afy2e/view?usp=sharing), so that you could try to train the model without collecting data. Download it and unzip it. Then specify the dataset path in `scripts/train_policy.sh`.

For example,  I put the dataset in `/home/ze/projects/DexPIE/training_data_example`, and I set `dataset_path=/home/ze/projects/DexPIE/training_data_example` in `scripts/train_policy.sh`.

Then you can train the policy.

**Train.** The script to train policy:

    bash scripts/train_policy.sh DexPIE Recap-image 0913_example

## BibTeX

Please consider citing our paper if you find this repo useful:
```
@article{ze2024humanoid_manipulation,
  title   = {Generalizable Humanoid Manipulation with Diffusion Policies},
  author  = {Yanjie Ze and Zixuan Chen and Wenhao Wang and Tianyi Chen and Xialin He and Ying Yuan and Xue Bin Peng and Jiajun Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv:2410.10803}
}
```

## Acknowledgement

We thank the authors of the following repos for their great work: [Diffusion Policy](https://github.com/columbia-ai-robotics/diffusion_policy), [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop), [Open-TeleVision](https://github.com/OpenTeleVision/TeleVision). 
