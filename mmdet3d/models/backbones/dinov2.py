

import torch.utils.checkpoint as checkpoint
from torch import nn
import torch
from mmdet.models import BACKBONES
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck


@BACKBONES.register_module()
class Dinov2Vit(nn.Module):

    def __init__(
            self,
            model_type: str = 'ViT-g',
            checkpoint: str = None,
    ):
        super(Dinov2Vit, self).__init__()
        # build backbone
        if model_type == 'ViT-g':
            # self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            self.model = torch.hub.load(repo_or_dir='/home/wangxinhao/dinov2', model = 'dinov2_vitg14', source='local', checkpoint=checkpoint)
        elif model_type == 'ViT-S': # embed_dim = 384
            # self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.model = torch.hub.load(repo_or_dir='/home/wangxinhao/dinov2', model = 'dinov2_vits14', source='local', checkpoint=checkpoint)
        elif model_type == 'ViT-B':
            # self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.model = torch.hub.load(repo_or_dir='/home/wangxinhao/dinov2', model = 'dinov2_vitb14', source='local', checkpoint=checkpoint)
        elif model_type == 'ViT-L':
            # self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.model = torch.hub.load(repo_or_dir='/home/wangxinhao/dinov2', model = 'dinov2_vitl14', source='local', checkpoint=checkpoint)
        else:
            raise ValueError('model_type should be valid!')

    def forward(self, x):
        assert(len(x.shape) == 4)
        H, W = x.shape[2], x.shape[3]
        x = self.model(x, is_training=True)
        x = x['x_norm_patchtokens']
        x = x.permute(0, 2, 1)
        x = torch.reshape(x, (x.shape[0], x.shape[1], H // self.model.patch_size, W // self.model.patch_size))
        return [x]
