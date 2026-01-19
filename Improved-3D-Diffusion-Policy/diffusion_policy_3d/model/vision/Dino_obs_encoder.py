import timm
from typing import List
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from diffusion_policy_3d.model.common.module_attr_mixin import ModuleAttrMixin
import copy

def load_dinov2_multiscale(model_name: str = "vit_large_patch14_dinov2",img_size: int = 518,):
    """
    使用 timm 加载 DINOv2 并启用多尺度特征输出
    
    Args:
        model_name: timm 模型ID，如 'dinov2_vitl14_reg',
        img_size: 输入图像大小，默认518 or 224
    """

    model = timm.create_model(
        model_name,
        pretrained=True,
        features_only=False,
        num_classes=0,
        img_size=img_size,
    )
    model.eval()
    return model

class MultiLayerFeatureFusion(torch.nn.Module):
    def __init__(self, num_layers=4, feature_dim=1024):
        super().__init__()
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        
        # 权重参数用于对不同层的特征进行加权
        self.layer_weights = torch.nn.Parameter(torch.ones(num_layers))
        #自注意力交互融合图像特征
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
        fused_features = fused_features.transpose(1, 2)   # [B, N, F]
        attn_features, _ = self.attn(fused_features, fused_features, fused_features)
        fused_features = fused_features + attn_features     # [B, N, F]
        
        # 应用层归一化
        fused_features = self.norm(fused_features)
        
        return fused_features
#
class DinoObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str="vit_large_patch14_dinov2",
            out_indices: List[int]=[7, 11, 15, 23],
            transforms: list=None,
        ):
        super().__init__()

        self.out_indices = out_indices
        self.model_name = model_name
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()#存放需要训练的融合网络
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()
        self.img_size = shape_meta['obs']['image']['shape'][1]
        #加载DINOv2模型
        self.Dino_model = load_dinov2_multiscale(self.model_name, img_size=self.img_size)
        self.embed_dim = self.Dino_model.embed_dim

        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)#224 or 518
            ] + transforms[1:]+[torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            #Dino需要imagenet的归一化
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                key_model_map[key] = MultiLayerFeatureFusion(num_layers=len(self.out_indices), 
                                                             feature_dim=self.embed_dim)
                #不同相机只创建自己的特征融合模块，DINO特征共用
                this_transform = transform
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                if not attr.get('ignore_by_policy', False):
                    low_dim_keys.append(key)
                    input_dim = shape[-1]
                    mlp = nn.Sequential(
                        nn.Linear(input_dim, self.embed_dim//2),
                        nn.ReLU(),
                        nn.Linear(self.embed_dim//2, self.embed_dim),
                        nn.LayerNorm(self.embed_dim),
                    )
                    key_model_map[key] = mlp
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print('rgb keys:         ', rgb_keys)
        print('low_dim_keys keys:', low_dim_keys)


    def forward(self, obs_dict):
        """
        前向传播方法：处理多模态观察数据并提取特征
        
        Args:
            obs_dict (Dict[str, torch.Tensor]): 观察数据字典，包含RGB图像和低维状态数据
                - RGB图像: shape [B, T, C, H, W] 其中B为批次大小，T为时间步数
                - 低维数据: shape [B, T, D] 其中D为特征维度
        
        Returns:
            torch.Tensor: 融合后的特征张量，shape [B*T, N, embed_dim]
                        其中N为特征序列长度，通常为257（224*224/（14*14）+1），embed_dim为嵌入维度
        """
        features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # process rgb input 处理多相机输入
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            img = img.reshape(B*T, *img.shape[2:])

            if img.shape[2:] != self.key_shape_map[key]:
                target_H, target_W = self.key_shape_map[key][1], self.key_shape_map[key][2]#自动将输入插值到模型要求的大小
                img = F.interpolate(img, size=(target_H, target_W), mode='bilinear', align_corners=False)
            img = self.key_transform_map[key](img)#预处理
            # 提取DINO特征
            raw_feature = self.Dino_model.get_intermediate_layers(
                img, 
                n=self.out_indices,  # 提取4层
                return_class_token=False  # 是否返回cls token
            )
            feature = self.key_model_map[key](raw_feature)#融合多层DINO特征
            features.append(feature.reshape(B*T, -1, self.embed_dim))

        # process lowdim input 如agent_pos
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            data = data.reshape(B,T, -1)
            feature = self.key_model_map[key](data)
            features.append(feature)
        
        # concatenate all features
        result = torch.cat(features, dim=-2)
        return result
    
def test_dino_encoder():
    print("开始测试 DinoObsEncoder...")
    
    # 定义形状元数据
    shape_meta = {
        'obs': {
            'img': {
                'shape': [3, 224, 224],  # RGB图像形状 (C, H, W)
                'type': 'rgb'
            },
            'agent_pos': {
                'shape': [7],  # 机器人关节位置
                'type': 'low_dim'
            }
        }
    }
    
    # 创建编码器实例
    encoder = DinoObsEncoder(
        shape_meta=shape_meta,
        model_name="vit_large_patch14_dinov2",  # 使用较小的模型以节省资源
        out_indices=(7, 9, 10, 11),  # 减少层数以加快测试
        transforms=None
    )
    encoder.to('cuda')
    
    print("编码器创建成功!")
    print(f"RGB keys: {encoder.rgb_keys}")
    print(f"Low-dim keys: {encoder.low_dim_keys}")
    print(f"Embedding dimension: {encoder.embed_dim}")
    
    # 准备测试数据
    batch_size = 2
    seq_len = 1  # 时间步数
    
    obs_dict = {
        'img': torch.randn(batch_size, seq_len, 3, 518, 518).to('cuda'),  # RGB图像
        'agent_pos': torch.randn(batch_size, seq_len, 7).to('cuda')      # 机器人状态
    }
    
    print(f"\n输入数据形状:")
    for key, tensor in obs_dict.items():
        print(f"  {key}: {tensor.shape}")
    
    # 执行前向传播
    print("\n执行前向传播...")
    try:
        start_time = time.time()
        output = encoder(obs_dict)
        end_time = time.time()
        print(f"前向传播耗时: {end_time - start_time:.4f} 秒")
        
        print(f"输出形状: {output.shape}")
        print("测试成功! DinoObsEncoder工作正常。")
        
        # 验证输出的一些属性
        print(f"输出张量的最大值: {output.max().item():.4f}")
        print(f"输出张量的最小值: {output.min().item():.4f}")
        print(f"输出张量的平均值: {output.mean().item():.4f}")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dino_encoder()