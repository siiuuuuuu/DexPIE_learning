import torch
import timm
from typing import List
from torchvision import transforms
import numpy as np
import time

class MultiLayerFeatureFusion(torch.nn.Module):
    """
    多层特征融合模块
    将多个中间层特征进行融合
    """
    def __init__(self, num_layers=4, feature_dim=1024):
        super().__init__()
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        
        # 权重参数用于对不同层的特征进行加权
        self.layer_weights = torch.nn.Parameter(torch.ones(num_layers))
        self.attn = torch.nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        # 1x1卷积用于融合特征
        self.fusion_conv = torch.nn.Conv1d(feature_dim * num_layers, feature_dim, kernel_size=1)
        
        # Layer Normalization
        self.norm = torch.nn.LayerNorm(feature_dim)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        特征融合
        Args:
            features: 包含多个特征层的列表，每个特征形状为 [batch_size, num_tokens, feature_dim]
        Returns:
            融合后的特征，形状为 [batch_size, num_tokens, feature_dim]
        """
        batch_size, num_tokens, _ = features[0].shape
        
        # 对每一层特征应用权重
        weighted_features = []
        for i, feat in enumerate(features):
            # 应用层权重
            weighted_feat = feat * self.layer_weights[i]
            weighted_features.append(weighted_feat)
        
        # 拼接所有层的特征
        concatenated_features = torch.cat(weighted_features, dim=-1)  # [B, N, F*num_layers]
        
        # 通过1x1卷积融合
        # 需要先转置维度以适应Conv1d
        fused_features = concatenated_features.transpose(1, 2)  # [B, F*num_layers, N]
        fused_features = self.fusion_conv(fused_features)  # [B, F, N]
        fused_features = fused_features.transpose(1, 2)     # [B, N, F]
        attn_features, _ = self.attn(fused_features, fused_features, fused_features)
        fused_features = fused_features + attn_features 
        # 应用层归一化
        fused_features = self.norm(fused_features)
        
        return fused_features
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
        #features_only=True,
        #out_indices=out_indices,
        img_size=224,
        num_classes=0  
    )
    model.to("cuda")
    model.eval()
    return model

print("可用的 DINOv2 模型:", timm.list_models("*dino*"))
# 使用示例
model = load_dinov2_multiscale()
print(model.embed_dim)
print(type(model).__module__)  
fused_module = MultiLayerFeatureFusion(num_layers=4, feature_dim=1024).to("cuda")
# 输入图像 [B, 3, 518, 518]
dummy_array = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
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
        return_class_token=False  # 是否返回cls token
    )
    end_time = time.time()
    print(f"提取特征时间: {end_time - start_time} 秒")
    
# features是元组，每个元素是一个特征层
for i, feat in enumerate(features):
    print(f"中间层 {i}: {feat.shape}") 

fused_features = fused_module(features)
print(f"参数量: {sum(p.numel() for p in fused_module.parameters())/1e6:.2f}M")
print(f"融合特征: {fused_features.shape}")