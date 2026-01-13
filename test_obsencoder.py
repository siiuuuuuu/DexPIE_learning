import torch
import timm
from typing import List
from torchvision import transforms
import numpy as np
import time
def load_dinov2_multiscale(model_name: str = "vit_large_patch14_dinov2", 
                           out_indices: tuple = (1, 5, 9, 11)):
    """
    使用 timm 加载 DINOv2 并启用多尺度特征输出
    
    Args:
        model_name: timm 模型ID，如 'dinov2_vitl14_reg'
        out_indices: 要提取的特征层索引，DINOv2 共12层
    """
    # 关键：features_only=True 启用多尺度输出
    model = timm.create_model(
        model_name,
        pretrained=True,
        features_only=False,
        num_classes=0  
    )
    model.to("cuda")
    model.eval()
    return model

print("可用的 DINOv2 模型:", timm.list_models("*dino*"))
# 使用示例
model = load_dinov2_multiscale()

# 输入图像 [B, 3, 518, 518]
dummy_array = (np.random.rand(518, 518, 3) * 255).astype(np.uint8)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(dummy_array).unsqueeze(0).to("cuda")
with torch.no_grad():
    # n=4 表示提取最后4层，也可以指定具体层索引如 n=[8, 16, 20]
    # reshape=True 将token重排为空间特征图
    start_time = time.time()
    features = model.get_intermediate_layers(
        input_tensor, 
        n=[7,11,15,23],  # 提取4层
        reshape=True,  # 重塑为 [B, C, H, W] 格式
        return_class_token=False  # 是否返回cls token
    )
    end_time = time.time()
    print(f"提取特征时间: {end_time - start_time} 秒")
    
# features是元组，每个元素是一个特征层
for i, feat in enumerate(features):
    print(f"中间层 {i}: {feat.shape}") 