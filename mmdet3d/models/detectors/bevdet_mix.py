
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule, xavier_init

from torch import nn as nn
import math

from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from mmdet.models.backbones.resnet import ResNet
from mmdet3d.models.backbones import VovNetFPN, SwinTransformer
from mmdet3d.ops import locatt_ops
from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2

from typing import Any, Dict, List, Optional, Tuple, Union

from .bevdet import BEVDepth4D, BEVStereo4D
from .cb_bevdet import CBBEVStereo4D

import pickle
import time

class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)


class BEVGridTransform(nn.Module):
    def __init__(
        self,
        *,
        input_scope: List[Tuple[float, float, float]],
        output_scope: List[Tuple[float, float, float]],
        prescale_factor: float = 1,
    ) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prescale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.prescale_factor,
                mode="bilinear",
                align_corners=False,
            )

        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
            self.input_scope, self.output_scope
        ):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))

        u, v = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)

        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return x


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, bias='auto',
                 inplace=True, affine=True):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.use_norm = norm_layer is not None
        self.use_activation = activation_layer is not None
        if bias == 'auto':
            bias = not self.use_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.use_norm:
            self.bn = norm_layer(out_channels, affine=affine)
        if self.use_activation:
            self.activation = activation_layer(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x


class similarFunction(torch.autograd.Function):
    """ credit: https://github.com/zzd1992/Image-Local-Attention """

    @staticmethod
    def forward(ctx, x_ori, x_loc, kH, kW):
        ctx.save_for_backward(x_ori, x_loc)
        ctx.kHW = (kH, kW)
        output = locatt_ops.localattention.similar_forward(
            x_ori, x_loc, kH, kW)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ori, x_loc = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = locatt_ops.localattention.similar_backward(
            x_loc, grad_outputs, kH, kW, True)
        grad_loc = locatt_ops.localattention.similar_backward(
            x_ori, grad_outputs, kH, kW, False)

        return grad_ori, grad_loc, None, None


class weightingFunction(torch.autograd.Function):
    """ credit: https://github.com/zzd1992/Image-Local-Attention """

    @staticmethod
    def forward(ctx, x_ori, x_weight, kH, kW):
        ctx.save_for_backward(x_ori, x_weight)
        ctx.kHW = (kH, kW)
        output = locatt_ops.localattention.weighting_forward(
            x_ori, x_weight, kH, kW)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x_ori, x_weight = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = locatt_ops.localattention.weighting_backward_ori(
            x_weight, grad_outputs, kH, kW)
        grad_weight = locatt_ops.localattention.weighting_backward_weight(
            x_ori, grad_outputs, kH, kW)

        return grad_ori, grad_weight, None, None


class LocalContextAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, last_affine=True):
        super().__init__()

        self.f_similar = similarFunction.apply
        self.f_weighting = weightingFunction.apply

        self.kernel_size = kernel_size
        self.query_project = nn.Sequential(ConvBNReLU(in_channels,
                                                      out_channels,
                                                      kernel_size=1,
                                                      norm_layer=nn.BatchNorm2d,
                                                      activation_layer=nn.ReLU),
                                           ConvBNReLU(out_channels,
                                                      out_channels,
                                                      kernel_size=1,
                                                      norm_layer=nn.BatchNorm2d,
                                                      activation_layer=nn.ReLU))
        self.key_project = nn.Sequential(ConvBNReLU(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    norm_layer=nn.BatchNorm2d,
                                                    activation_layer=nn.ReLU),
                                         ConvBNReLU(out_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    norm_layer=nn.BatchNorm2d,
                                                    activation_layer=nn.ReLU))
        self.value_project = ConvBNReLU(in_channels,
                                        out_channels,
                                        kernel_size=1,
                                        norm_layer=nn.BatchNorm2d,
                                        activation_layer=nn.ReLU,
                                        affine=last_affine)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, target_feats, source_feats, **kwargs):
        query = self.query_project(target_feats)
        key = self.key_project(source_feats)
        value = self.value_project(source_feats)

        weight = self.f_similar(query, key, self.kernel_size, self.kernel_size)
        weight = nn.functional.softmax(weight / math.sqrt(key.size(1)), -1)
        out = self.f_weighting(value, weight, self.kernel_size, self.kernel_size)
        return out


@DETECTORS.register_module()
class BEVDepth4D_mix_encoder(BEVDepth4D):
    def __init__(self,
                 longterm_model=None,
                 reduc_conv=None,
                 imc=256,
                 longterm_imc=256,
                 diff_bev=None,
                 **kwargs):
        super(BEVDepth4D_mix_encoder, self).__init__(**kwargs)
        self.longterm_model = builder.build_detector(longterm_model)
        if reduc_conv!=None:
            self.reduc_conv = ConvModule(
                    imc + longterm_imc,
                    imc + longterm_imc,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                    act_cfg=dict(type='ReLU'),
                    inplace=False)
            if 'se' in reduc_conv and reduc_conv['se']:
                print('Using SE Block!')
                self.se = SE_Block(imc + longterm_imc)
        self.diff_bev = diff_bev
        if self.diff_bev is not None:
            self.st2lt = BEVGridTransform(input_scope=self.diff_bev['st_scope'],
                                          output_scope=self.diff_bev['lt_scope'])
            self.lt2st = BEVGridTransform(input_scope=self.diff_bev['lt_scope'],
                                          output_scope=self.diff_bev['st_scope'])
            if reduc_conv != None:
                self.reduc_conv_forseg = ConvModule(
                    imc + longterm_imc,
                    imc + longterm_imc,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                    act_cfg=dict(type='ReLU'),
                    inplace=False)
                if 'se' in reduc_conv and reduc_conv['se']:
                    print('Using SE Block!')
                    self.se_forseg = SE_Block(imc + longterm_imc)


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_masks_bev=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      img_inputs_lt=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # print("@@@@@@@@", img_inputs[0].shape)
        # print("########", img_inputs_lt[0].shape)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, with_bevencoder=False, **kwargs)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)

        img_feats_lt, pts_feats_lt, depth_lt = self.longterm_model.extract_feat(
            points, img=img_inputs_lt, img_metas=img_metas, with_bevencoder=False, **kwargs)
        gt_depth_lt = kwargs['gt_depth_lt']
        # print("@@@", depth.shape)
        # print("@@@", gt_depth.shape)
        # print("###", depth_lt.shape)
        # print("###", gt_depth_lt.shape)
        loss_depth_lt = self.longterm_model.img_view_transformer.get_depth_loss(gt_depth_lt, depth_lt)
        losses_lt = dict(loss_depth_longterm=loss_depth_lt)
        losses.update(losses_lt)

        # print("#### short term BEV: ", len(img_feats), img_feats[0].shape)
        # print("#### long term BEV: ", len(img_feats_lt), img_feats_lt[0].shape)

        if self.diff_bev is None:
            new_img_feats = []
            for i in range(0, len(img_feats)):
                new_img_feat = self.reduc_conv(torch.cat((img_feats[i], img_feats_lt[i]), dim=1))
                if hasattr(self, 'se') and self.se is not None:
                    new_img_feat = self.se(new_img_feat)
                new_img_feat = self.bev_encoder(new_img_feat)
                new_img_feats.append(new_img_feat)

            losses_pts = self.forward_pts_train(new_img_feats, gt_bboxes_3d,
                                                gt_labels_3d, gt_masks_bev,
                                                img_metas, gt_bboxes_ignore)
            losses.update(losses_pts)

        else:
            new_img_feats = []
            new_img_feats_forseg = []

            img_feats_st2lt = [self.st2lt(img_feats[0])]
            img_feats_lt2st = [self.lt2st(img_feats_lt[0])]

            for i in range(0, len(img_feats)):
                new_img_feat = self.reduc_conv(torch.cat((img_feats[i], img_feats_lt2st[i]), dim=1))
                if hasattr(self, 'se') and self.se is not None:
                    new_img_feat = self.se(new_img_feat)
                new_img_feat = self.bev_encoder(new_img_feat)
                new_img_feats.append(new_img_feat)

            for i in range(0, len(img_feats)):
                new_img_feat = self.reduc_conv_forseg(torch.cat((img_feats_st2lt[i], img_feats_lt[i]), dim=1))
                if hasattr(self, 'se_forseg') and self.se_forseg is not None:
                    new_img_feat = self.se_forseg(new_img_feat)
                new_img_feat = self.bev_encoder_forseg(new_img_feat)
                new_img_feats_forseg.append(new_img_feat)

            # print("#### det BEV: ", len(new_img_feats), new_img_feats[0].shape)
            # print("#### seg BEV: ", len(new_img_feats_forseg), new_img_feats_forseg[0].shape)

            losses_pts = self.forward_pts_train(new_img_feats, gt_bboxes_3d,
                                                gt_labels_3d, gt_masks_bev,
                                                img_metas, gt_bboxes_ignore,
                                                pts_feats_forseg=new_img_feats_forseg)
            losses.update(losses_pts)

        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     img_inputs_lt=None,
                     gt_masks_bev=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0], img_inputs_lt[0], gt_masks_bev, **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)


    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    img_lt=None,
                    gt_masks_bev=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""

        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, with_bevencoder=False, **kwargs)

        img_feats_lt, _, _ = self.longterm_model.extract_feat(
            points, img=img_lt, img_metas=img_metas, with_bevencoder=False, **kwargs)

        if self.diff_bev is None:
            new_img_feats = []
            new_img_feats_forseg = None
            for i in range(0, len(img_feats)):
                new_img_feat = self.reduc_conv(torch.cat((img_feats[i], img_feats_lt[i]), dim=1))
                if hasattr(self, 'se') and self.se is not None:
                    new_img_feat = self.se(new_img_feat)
                new_img_feat = self.bev_encoder(new_img_feat)
                new_img_feats.append(new_img_feat)

        else:
            new_img_feats = []
            new_img_feats_forseg = []

            img_feats_st2lt = [self.st2lt(img_feats[0])]
            img_feats_lt2st = [self.lt2st(img_feats_lt[0])]

            for i in range(0, len(img_feats)):
                new_img_feat = self.reduc_conv(torch.cat((img_feats[i], img_feats_lt2st[i]), dim=1))
                if hasattr(self, 'se') and self.se is not None:
                    new_img_feat = self.se(new_img_feat)
                new_img_feat = self.bev_encoder(new_img_feat)
                new_img_feats.append(new_img_feat)

            for i in range(0, len(img_feats)):
                new_img_feat = self.reduc_conv_forseg(torch.cat((img_feats_st2lt[i], img_feats_lt[i]), dim=1))
                if hasattr(self, 'se_forseg') and self.se_forseg is not None:
                    new_img_feat = self.se_forseg(new_img_feat)
                new_img_feat = self.bev_encoder_forseg(new_img_feat)
                new_img_feats_forseg.append(new_img_feat)

        # import numpy as np
        # import matplotlib.pyplot as plt
        # print(gt_masks_bev[0].shape)
        # gt_to_vis = gt_masks_bev[0]
        # plt.figure(figsize=(10, 10))
        # x = np.arange(-51.2, 51.2, 0.8)
        # y = np.arange(-51.2, 51.2, 0.8)
        # plt.pcolormesh(x, y, gt_masks_bev[0].cpu().numpy(), cmap='viridis')
        # plt.scatter
        # plt.savefig("/home/xiazhongyu/vis/gtgtgt.png")
        # exit(0)

        bbox_list = [dict() for _ in range(len(img_metas))]
        if self.pts_bbox_head:
            bbox_pts = self.simple_test_pts(new_img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if self.pts_seg_head:
            if new_img_feats_forseg is None:
                bbox_segs = self.pts_seg_head(new_img_feats, gt_masks_bev)
            else:
                bbox_segs = self.pts_seg_head(new_img_feats_forseg, gt_masks_bev)
            for result_dict, pts_seg, gt in zip(bbox_list, bbox_segs, gt_masks_bev):
                result_dict['pts_seg'] = pts_seg
                result_dict['gt_masks_bev'] = gt
        return bbox_list


@DETECTORS.register_module()
class BEVStereo4D_mix_encoder(BEVStereo4D):
    def __init__(self,
                 longterm_model=None,
                 reduc_conv=None,
                 imc=256,
                 longterm_imc=256,
                 diff_bev=None,
                 numC_Trans=80,
                 d2t=False,
                 **kwargs):
        super(BEVStereo4D_mix_encoder, self).__init__(**kwargs)
        self.longterm_model = builder.build_detector(longterm_model)
        if reduc_conv!=None:
            self.reduc_conv = ConvModule(
                    imc + longterm_imc,
                    imc + longterm_imc,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                    act_cfg=dict(type='ReLU'),
                    inplace=False)
            if 'se' in reduc_conv and reduc_conv['se']:
                print('Using SE Block!')
                self.se = SE_Block(imc + longterm_imc)
        self.diff_bev = diff_bev
        self.d2t = d2t
        if self.d2t:
            self.Local = LocalContextAttentionBlock(numC_Trans*2, numC_Trans*2, 9, last_affine=True)
        self.numC_Trans = numC_Trans
        if self.diff_bev is not None:
            self.st2lt = BEVGridTransform(input_scope=self.diff_bev['st_scope'],
                                          output_scope=self.diff_bev['lt_scope'])
            self.lt2st = BEVGridTransform(input_scope=self.diff_bev['lt_scope'],
                                          output_scope=self.diff_bev['st_scope'])
            if reduc_conv != None:
                self.reduc_conv_forseg = ConvModule(
                    imc + longterm_imc,
                    imc + longterm_imc,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                    act_cfg=dict(type='ReLU'),
                    inplace=False)
                if 'se' in reduc_conv and reduc_conv['se']:
                    print('Using SE Block!')
                    self.se_forseg = SE_Block(imc + longterm_imc)

    def d2t_func(self, feats, gamma=1):

        BN, C, W, H = feats.shape
        frames = C // self.numC_Trans

        feat_list = []
        for i in range(frames):
            feat_list.append(feats[:, i*self.numC_Trans:(i+1)*self.numC_Trans, :, :])

        # for i in range(frames-1, 0, -1):
        #     feat_list[i-1] = self.Local(feat_list[i-1], feat_list[i])

        # for i in range(frames-1):
        #     feat_list[i+1] = self.Local(feat_list[i], feat_list[i+1])

        for i in range(frames-1, 0, -1):
            cat_feat = torch.cat([feat_list[i - 1], feat_list[i]], dim=1)
            attn_out = self.Local(cat_feat, cat_feat)
            feat_list[i - 1] = feat_list[i - 1] + gamma * (
                    attn_out[:, :self.numC_Trans, :, :] + attn_out[:, self.numC_Trans:, :, :]
            ) / 2

        for i in range(frames-1):
            cat_feat = torch.cat([feat_list[i], feat_list[i + 1]], dim=1)
            attn_out = self.Local(cat_feat, cat_feat)
            feat_list[i + 1] = feat_list[i + 1] + gamma * (
                    attn_out[:, :self.numC_Trans, :, :] + attn_out[:, self.numC_Trans:, :, :]
            ) / 2

        return torch.cat(feat_list, dim=1)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_masks_bev=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      img_inputs_lt=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # print("@@@@@@@@", img_inputs[0].shape)
        # print("########", img_inputs_lt[0].shape)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, with_bevencoder=False, **kwargs)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)

        img_feats_lt, pts_feats_lt, depth_lt = self.longterm_model.extract_feat(
            points, img=img_inputs_lt, img_metas=img_metas, with_bevencoder=False, **kwargs)
        gt_depth_lt = kwargs['gt_depth_lt']
        # print("@@@", depth.shape)
        # print("@@@", gt_depth.shape)
        # print("###", depth_lt.shape)
        # print("###", gt_depth_lt.shape)
        loss_depth_lt = self.longterm_model.img_view_transformer.get_depth_loss(gt_depth_lt, depth_lt)
        losses_lt = dict(loss_depth_longterm=loss_depth_lt)
        losses.update(losses_lt)

        # print("#### short term BEV: ", len(img_feats), img_feats[0].shape)
        # print("#### long term BEV: ", len(img_feats_lt), img_feats_lt[0].shape)

        if self.diff_bev is None:
            new_img_feats = []
            for i in range(0, len(img_feats)):
                if self.d2t:
                    new_img_feat = self.reduc_conv(self.d2t_func(torch.cat((img_feats[i], img_feats_lt[i]), dim=1)))
                else:
                    new_img_feat = self.reduc_conv(torch.cat((img_feats[i], img_feats_lt[i]), dim=1))
                if hasattr(self, 'se') and self.se is not None:
                    new_img_feat = self.se(new_img_feat)
                new_img_feat = self.bev_encoder(new_img_feat)
                new_img_feats.append(new_img_feat)

            losses_pts = self.forward_pts_train(new_img_feats, gt_bboxes_3d,
                                                gt_labels_3d, gt_masks_bev,
                                                img_metas, gt_bboxes_ignore)
            losses.update(losses_pts)

        else:
            new_img_feats = []
            new_img_feats_forseg = []

            img_feats_st2lt = [self.st2lt(img_feats[0])]
            img_feats_lt2st = [self.lt2st(img_feats_lt[0])]

            for i in range(0, len(img_feats)):
                if self.d2t:
                    new_img_feat = self.reduc_conv(self.d2t_func(torch.cat((img_feats[i], img_feats_lt2st[i]), dim=1)))
                else:
                    new_img_feat = self.reduc_conv(torch.cat((img_feats[i], img_feats_lt2st[i]), dim=1))
                if hasattr(self, 'se') and self.se is not None:
                    new_img_feat = self.se(new_img_feat)
                new_img_feat = self.bev_encoder(new_img_feat)
                new_img_feats.append(new_img_feat)

            for i in range(0, len(img_feats)):
                if self.d2t:
                    new_img_feat = self.reduc_conv(self.d2t_func(torch.cat((img_feats_st2lt[i], img_feats_lt[i]), dim=1)))
                else:
                    new_img_feat = self.reduc_conv_forseg(torch.cat((img_feats_st2lt[i], img_feats_lt[i]), dim=1))
                if hasattr(self, 'se_forseg') and self.se_forseg is not None:
                    new_img_feat = self.se_forseg(new_img_feat)
                new_img_feat = self.bev_encoder_forseg(new_img_feat)
                new_img_feats_forseg.append(new_img_feat)

            # print("#### det BEV: ", len(new_img_feats), new_img_feats[0].shape)
            # print("#### seg BEV: ", len(new_img_feats_forseg), new_img_feats_forseg[0].shape)

            losses_pts = self.forward_pts_train(new_img_feats, gt_bboxes_3d,
                                                gt_labels_3d, gt_masks_bev,
                                                img_metas, gt_bboxes_ignore,
                                                pts_feats_forseg=new_img_feats_forseg)
            losses.update(losses_pts)

        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     img_inputs_lt=None,
                     gt_masks_bev=None,
                     gt_bboxes_3d=None,  # for vis
                     gt_labels_3d=None,  # for vis
                     imgs_ori=None,  # for vis
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        vis = False
        if vis:
            dict_for_vis = {}
            dict_for_vis['img_inputs'] = img_inputs
            dict_for_vis['imgs_ori'] = imgs_ori
            dict_for_vis['gt_bbox_corners'] = gt_bboxes_3d[0][0].corners
            dict_for_vis['gt_bbox_labels'] = gt_labels_3d[0][0]
            dict_for_vis['gt_seg'] = gt_masks_bev[0][0]

        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            bbox_list = self.simple_test(points[0], img_metas[0], img_inputs[0], img_inputs_lt[0], gt_masks_bev, **kwargs)

            if vis:
                dict_for_vis['pred_bbox_corners'] = bbox_list[0]['pts_bbox']['boxes_3d'].corners
                dict_for_vis['pred_bbox_labels'] = bbox_list[0]['pts_bbox']['labels_3d']
                dict_for_vis['pred_seg'] = bbox_list[0]['pts_seg']
                millis = str(int(round(time.time() * 1000)))
                # with open("/home/xiazhongyu/vis/" + millis + ".pkl", "wb") as tf:
                #     pickle.dump(dict_for_vis, tf)
                #     print("dict for vis saved as " + "/home/xiazhongyu/vis/" + millis + ".pkl")
                from tools.vis_local import draw_xiazhongyu
                draw_xiazhongyu(dict_for_vis, "/home/xiazhongyu/vis/" + millis + ".png")

            return bbox_list
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)


    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    img_lt=None,
                    gt_masks_bev=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""

        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, with_bevencoder=False, **kwargs)

        img_feats_lt, _, _ = self.longterm_model.extract_feat(
            points, img=img_lt, img_metas=img_metas, with_bevencoder=False, **kwargs)

        if self.diff_bev is None:
            new_img_feats = []
            new_img_feats_forseg = None
            for i in range(0, len(img_feats)):
                new_img_feat = self.reduc_conv(torch.cat((img_feats[i], img_feats_lt[i]), dim=1))
                if hasattr(self, 'se') and self.se is not None:
                    new_img_feat = self.se(new_img_feat)
                new_img_feat = self.bev_encoder(new_img_feat)
                new_img_feats.append(new_img_feat)

        else:
            new_img_feats = []
            new_img_feats_forseg = []

            img_feats_st2lt = [self.st2lt(img_feats[0])]
            img_feats_lt2st = [self.lt2st(img_feats_lt[0])]

            for i in range(0, len(img_feats)):
                new_img_feat = self.reduc_conv(torch.cat((img_feats[i], img_feats_lt2st[i]), dim=1))
                if hasattr(self, 'se') and self.se is not None:
                    new_img_feat = self.se(new_img_feat)
                new_img_feat = self.bev_encoder(new_img_feat)
                new_img_feats.append(new_img_feat)

            for i in range(0, len(img_feats)):
                new_img_feat = self.reduc_conv_forseg(torch.cat((img_feats_st2lt[i], img_feats_lt[i]), dim=1))
                if hasattr(self, 'se_forseg') and self.se_forseg is not None:
                    new_img_feat = self.se_forseg(new_img_feat)
                new_img_feat = self.bev_encoder_forseg(new_img_feat)
                new_img_feats_forseg.append(new_img_feat)

        # import numpy as np
        # import matplotlib.pyplot as plt
        # print(gt_masks_bev[0].shape)
        # gt_to_vis = gt_masks_bev[0]
        # plt.figure(figsize=(10, 10))
        # x = np.arange(-51.2, 51.2, 0.8)
        # y = np.arange(-51.2, 51.2, 0.8)
        # plt.pcolormesh(x, y, gt_masks_bev[0].cpu().numpy(), cmap='viridis')
        # plt.scatter
        # plt.savefig("/home/xiazhongyu/vis/gtgtgt.png")
        # exit(0)

        bbox_list = [dict() for _ in range(len(img_metas))]
        if self.pts_bbox_head:
            bbox_pts = self.simple_test_pts(new_img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if self.pts_seg_head:
            if new_img_feats_forseg is None:
                bbox_segs = self.pts_seg_head(new_img_feats, gt_masks_bev)
            else:
                bbox_segs = self.pts_seg_head(new_img_feats_forseg, gt_masks_bev)
            for result_dict, pts_seg, gt in zip(bbox_list, bbox_segs, gt_masks_bev):
                result_dict['pts_seg'] = pts_seg
                result_dict['gt_masks_bev'] = gt

        return bbox_list



