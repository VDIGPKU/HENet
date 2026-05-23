# Copyright (c) Phigent Robotics. All rights reserved.

import torch.utils.checkpoint as checkpoint

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

from mmdet.models import BACKBONES
from mmdet.models.backbones.resnet import Bottleneck

from .convblock_watermark import ConvBlock
from tools.watermark_cache import GlobalBEVCache


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_cfg='bn', passport_kwargs: dict=None):
        super(BasicBlock, self).__init__()
        if passport_kwargs is None:
            passport_kwargs = {'flag': True}
        self.convbnrelu_1 = ConvBlock(in_planes, planes, 3, stride, 1, bn=norm_cfg, passport_kwargs=passport_kwargs)
        self.convbn_2 = ConvBlock(planes, planes, 3, 1, 1, bn=norm_cfg, passport_kwargs=passport_kwargs)
        self.shortcut = nn.Sequential()
        if downsample is not None:
            self.shortcut = downsample
        elif stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = ConvBlock(in_planes, self.expansion * planes,
                                      1, stride, 0, bn=norm_cfg, passport_kwargs=passport_kwargs)
        # self.first_run = True

    def set_intermediate_keys(self, pre_trained, beta, gamma):
        with torch.no_grad():
            self.convbnrelu_1.set_private_keys(beta, gamma)
            out_beta, out_gamma = pre_trained.convbnrelu_1(beta), pre_trained.convbnrelu_1(gamma)
            self.convbn_2.set_private_keys(out_beta, out_gamma)
            out_beta, out_gamma = pre_trained.convbn_2(out_beta), pre_trained.convbn_2(out_gamma)
            if isinstance(self.shortcut, ConvBlock):
                self.shortcut.set_private_keys(beta, gamma)
            out_beta += pre_trained.shortcut(beta)
            out_gamma += pre_trained.shortcut(gamma)

            return F.relu(out_beta), F.relu(out_gamma)

    def forward(self, x):
        if GlobalBEVCache.force_initialize:
            print('init', self.__class__.__name__, 'with the watermark data')
            self.convbnrelu_1.set_private_keys(x, x)
        out = self.convbnrelu_1(x, GlobalBEVCache.forward_ind % 2)
        if GlobalBEVCache.force_initialize:
            self.convbn_2.set_private_keys(out, out)
        out = self.convbn_2(out, GlobalBEVCache.forward_ind % 2)
        if isinstance(self.shortcut, ConvBlock):
            if GlobalBEVCache.force_initialize:
                self.shortcut.set_private_keys(out, out)
            out = out + self.shortcut(x, GlobalBEVCache.forward_ind % 2)
        else:
            out = out + self.shortcut(x)
        out = F.relu(out)
        # self.first_run = False
        return out


@BACKBONES.register_module()
class CustomResNet_watermark(nn.Module):

    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg='bn',
            with_cp=False,
            block_type='Basic',
    ):
        super(CustomResNet_watermark, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        assert block_type=='Basic'
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    Bottleneck(
                        curr_numC,
                        num_channels[i] // 4,
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    Bottleneck(curr_numC, curr_numC // 4, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    BasicBlock(
                        curr_numC,
                        num_channels[i],
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


if __name__ == '__main__':
    test_model = CustomResNet_watermark(numC_input=16)
    print(test_model)
