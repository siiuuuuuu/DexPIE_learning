import copy

import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
from termcolor import cprint
from dexpie.common.import_util import load_r3m_model
from dexpie.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)


class TrainOnlyTransform(nn.Module):
    def __init__(self, transform: nn.Module):
        super().__init__()
        self.transform = transform

    def forward(self, x):
        if self.training and torch.is_grad_enabled():
            return self.transform(x)
        return x


# Used by critic_head.
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.relu = nn.ReLU()
        # If input and output dimensions differ, we need a projection shortcut
        self.shortcut=nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.linear(x)
        out = self.layer_norm(out)
        out = self.relu(out)
        # Add residual connection
        out = out + residual
        return out
    
# Resize images inside the model so dataset preprocessing does not need to.
class ValueNet(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str,
            pretrained: bool,
            frozen: bool,
            num_bins: int,
            transforms: list,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=True,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            downsample_ratio: int=32,
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()
        
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()
        self.num_bins = num_bins
        self.output_feat_dim = 0
        
        if model_name == "r3m":
            model = load_r3m_model("resnet18", pretrained=pretrained) # resnet18, resnet34
            model.eval()# Critical for BatchNorm behavior; the model otherwise fails to work correctly.
            cprint(f"Loaded R3M model using {model_name}. pretrained={pretrained}", 'green')
        else:
            raise NotImplementedError(f"Unsupported model_name: {model_name}")


        if frozen:
            cprint(f"Frozen model {model_name}", 'green')
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False
        
        feature_dim = 512 # R3M output dimension.
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
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + transforms[1:]
        if transforms is None:
            transform = nn.Identity()
        else:
            # torchvision random transforms ignore module eval() by default.
            # Keep critic augmentation for training, but make no-grad inference deterministic.
            transform = TrainOnlyTransform(torch.nn.Sequential(*transforms))

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                this_model = model if share_rgb_model else copy.deepcopy(model) # Share obs encoder by treating multi-view inputs as batch items.
                key_model_map[key] = this_model

                this_transform = transform
                key_transform_map[key] = this_transform
                self.output_feat_dim += feature_dim # Concatenate feature dimensions from each rgb key.
            elif type == 'low_dim':
                if not attr.get('ignore_by_policy', False):
                    low_dim_keys.append(key)
                    self.output_feat_dim += shape[0] # Concatenate feature dimensions from each low_dim key.
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print('rgb keys:         ', rgb_keys)
        print('low_dim_keys keys:', low_dim_keys)

        self.model_name = model_name
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.critic_head =nn.Sequential(
            ResidualBlock(self.output_feat_dim, self.output_feat_dim*2),
            ResidualBlock(self.output_feat_dim*2, self.output_feat_dim*2),
            ResidualBlock(self.output_feat_dim*2, self.output_feat_dim//2),
            nn.Linear(self.output_feat_dim//2, self.num_bins),
        )
        
        # model.to(self.device)
        logger.info(
            "number of parameters: %.2f M", sum(p.numel() for p in self.parameters()) / 1e6
        )

        
    def forward(self, obs_dict):
        features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        img_list=[]
        bt = None
        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            if bt is None:
                bt = B * T
            else:
                assert bt == B * T
            img = img.reshape(B*T, *img.shape[2:])

            if img.shape[2:] != self.key_shape_map[key]:
                target_H, target_W = self.key_shape_map[key][1], self.key_shape_map[key][2]# Interpolate input to the model-required size.
                # do torchvision resize
                # img shape: Bx3xHxW
                # new size: Bx3xnHxnW
                img = F.interpolate(img, size=(target_H, target_W), mode='bilinear', align_corners=False)
            img = self.key_transform_map[key](img)
            img_list.append(img)
        if len(self.rgb_keys) > 0:
            key = self.rgb_keys[0]# Use the first key's model for shared encoding.
            img = torch.cat(img_list, dim=0) # Run all images through the model in parallel.
            feature = self.key_model_map[key](img).to(self.device)
            assert len(feature.shape) == 2 and feature.shape[0] == bt * len(self.rgb_keys)
            tuple_feature = torch.split(feature, bt, dim=0) # Split by key; each block is [B*T, feat].
            features.extend(tuple_feature)
            #raw_feature = self.key_model_map[key](img).to(self.device)
            #feature = self.aggregate_feature(raw_feature)
            #assert len(feature.shape) == 2 and feature.shape[0] == B * T
            #features.append(feature.reshape(B, -1))
            # print("feat:", feature.device)

        # Process lowdim input, e.g. agent_pos.
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            if bt is None:
                bt = B * T
            else:
                assert bt == B * T
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B*T, -1))
            # print("data:", data.device)
        
        # concatenate all features
        cat_feature = torch.cat(features, dim=-1)
        bins_prob = self.critic_head(cat_feature) #shape [B*T,num_bins]

        return bins_prob
    

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        
        return example_output.shape


if __name__=='__main__':
    value_net = ValueNet(
        shape_meta=None,
        model_name='resnet18.a1_in1k',
        pretrained=False,
        global_pool='',
        transforms=None
    )
