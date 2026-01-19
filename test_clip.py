import timm
import torch
# 列出所有包含 "clip" 且包含 "small" 的模型名称
clip_small_models = [name for name in timm.list_models() if "dino" in name ]
print("timm 中支持的 CLIP Small 模型：", clip_small_models)

# 尝试直接加载模型（验证是否可用）
try:
    model = timm.create_model("vit_small_patch14_dinov2", pretrained=True, img_size=224)
    print("模型加载成功！参数量：", sum(p.numel() for p in model.parameters())/1e6, "M")
    forward_features = model.forward_features
    x = torch.randn(1, 3, 224, 224)
    features = model(x)
    print("特征图形状：", features.shape)
except Exception as e:
    print("模型加载失败：", e)