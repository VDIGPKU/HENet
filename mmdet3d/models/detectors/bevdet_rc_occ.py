# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVStereo4D, BEVDepth4D

import torch
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np
from mmdet3d.models.builder import build_head, build_neck, build_backbone
import random

from .. import builder
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmcv.ops import Voxelization
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential


class Unet3D(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Unet3D, self).__init__()
        self.init_dres = nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.hg1 = Hourglass3D(mid_channels)
        self.hg2 = Hourglass3D(mid_channels)

    def forward(self, x):
        dres = self.init_dres(x)
        out1, pre1, post1 = self.hg1(dres, None, None)
        out1 = out1 + dres
        out2, pre2, post2 = self.hg2(out1, pre1, post1)
        out2 = out2 + dres
        return out2


class Hourglass3D(nn.Module):
    def __init__(self, mid_channels):
        super(Hourglass3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, mid_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, 2 * mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, 2 * mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, 2 * mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x, presqu=None, postsqu=None):
        out = self.conv1(x)  # 1 64 10 128 128
        pre = self.conv2(out)  # 1 64 10 128 128

        if postsqu is not None:
            pre = F.leaky_relu(pre + postsqu, inplace=True)
        else:
            pre = F.leaky_relu(pre, inplace=True)
        out = self.conv3(pre)  # 1 64 5 64 64
        out = self.conv4(out)  # 1 64 5 64 64
        out = F.interpolate(out, (pre.shape[-3], pre.shape[-2], pre.shape[-1]), mode='trilinear', align_corners=True)
        out = self.conv5(out)  # 1 64 10 128 128
        if presqu is not None:
            post = F.leaky_relu(out + presqu, inplace=True)
        else:
            post = F.leaky_relu(out + pre, inplace=True)
        out = F.interpolate(post, (x.shape[-3], x.shape[-2], x.shape[-1]), mode='trilinear', align_corners=True)
        out = self.conv6(out)
        return out, pre, post


@DETECTORS.register_module()
class BEVStereo4DOCCHopRC(BEVStereo4D):

    def __init__(self,

                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,

                 with_hop=False,
                 hop_cfg=None,
                 hop_load_all=False,
                 use_short=False,

                 radar_voxel_layer=None,
                 radar_voxel_encoder=None,
                 radar_middle_encoder=None,
                 radar_bev_backbone=None,
                 radar_bev_neck=None,
                 radar_reduc_conv=False,  # new
                 imc=256, rac=64,  # im ra 特征维度
                 freeze_img=False,
                 sparse_shape=None,

                 **kwargs):
        super(BEVStereo4DOCCHopRC, self).__init__(**kwargs)
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
            self.img_view_transformer.out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d'))
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes),
            )
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False

        self.with_hop = with_hop
        if self.with_hop:
            self.hop_cfg = hop_cfg
            self.long_term_backbone = build_backbone(self.hop_cfg.long_term_backbone)
            self.long_term_neck = build_neck(self.hop_cfg.long_term_neck)

            self.target_frame = self.hop_cfg.target_frame

            self.radar_long_term_backbone = build_backbone(self.hop_cfg.radar_long_term_backbone)
            self.radar_long_term_neck = build_neck(self.hop_cfg.radar_long_term_neck)

        self.hop_load_all = hop_load_all

        self.use_short = use_short
        if self.use_short:
            self.short_term_decoder = nn.Sequential(
                nn.Conv3d(in_channels=32 * 2, out_channels=128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=128, out_channels=32, kernel_size=3, padding=1)
            )

            self.radar_short_term_decoder = nn.Sequential(
                nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=128, out_channels=32, kernel_size=3, padding=1)
            )

        if radar_voxel_layer != None:
            self.radar_voxel_layer = Voxelization(**radar_voxel_layer)
        if radar_voxel_encoder != None:
            self.radar_voxel_encoder = builder.build_voxel_encoder(radar_voxel_encoder)
        if radar_middle_encoder != None:
            self.radar_middle_encoder = builder.build_middle_encoder(radar_middle_encoder)
        if radar_bev_backbone is not None:
            self.radar_bev_backbone = builder.build_backbone(radar_bev_backbone)
        if radar_bev_neck is not None:
            self.radar_bev_neck = builder.build_neck(radar_bev_neck)

        if radar_reduc_conv:
            self.reduc_conv = ConvModule(
                rac + imc,
                self.img_view_transformer.out_channels,  # rac change imc
                kernel_size=3,
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

            if self.with_hop:
                self.long_reduc_conv = ConvModule(
                    self.img_view_transformer.out_channels + self.hop_cfg.radar_long_term_neck.out_channels,
                    self.img_view_transformer.out_channels,  # rac change imc
                    kernel_size=3,
                    padding=1,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
                    act_cfg=dict(type='ReLU'),
                    inplace=False)

                if self.use_short:
                    self.short_reduc_conv = ConvModule(
                        32 + 32,
                        32,  # rac change imc
                        kernel_size=3,
                        padding=1,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
                        act_cfg=dict(type='ReLU'),
                        inplace=False)

        self.freeze_img = freeze_img
        self.sparse_shape = sparse_shape

    def init_weights(self):
        """Initialize model weights."""
        super(BEVStereo4DOCCHopRC, self).init_weights()
        if self.freeze_img:

            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

            for name, param in self.named_parameters():
                if 'img_view_transformer' in name:
                    param.requires_grad = False
                if 'img_bev_encoder_backbone' in name:
                    param.requires_grad = False
                if 'img_bev_encoder_neck' in name:
                    param.requires_grad = False
                if 'pre_process' in name:
                    param.requires_grad = False

            def fix_bn(m):
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                    m.track_running_stats = False

            self.img_view_transformer.apply(fix_bn)
            self.img_bev_encoder_backbone.apply(fix_bn)
            self.img_bev_encoder_neck.apply(fix_bn)

            self.img_backbone.apply(fix_bn)
            self.img_neck.apply(fix_bn)

            self.pre_process_net.apply(fix_bn)

    @torch.no_grad()
    @force_fp32()
    def radar_voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        # print(len(points))
        # assert len(points) == 3
        for res in points:
            res_voxels, res_coors, res_num_points = self.radar_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        # print(num_points)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            # print(coor.size())
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        # print(coors_batch[-1, 0] + 1)
        return voxels, num_points, coors_batch

    def extract_radar_feat(self, radar, img_metas):
        """Extract features of points."""
        voxels, num_points, coors = self.radar_voxelize(radar)

        voxel_features = self.radar_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        # print(batch_size)
        # batch_size = 5

        # x_before = self.radar_middle_encoder(voxel_features, coors, batch_size)
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)

        x_before = input_sp_tensor.dense()

        # if hasattr(self, 'radar_bev_backbone') and self.radar_bev_backbone is not None:
        # print(x_before.shape)
        x = self.radar_bev_backbone(x_before)  # 8, 64, h/2, w/2

        # if hasattr(self, 'radar_bev_neck') and self.radar_bev_neck is not None:
        x = self.radar_bev_neck(x)  # 8, 64, h/4, w/4
        # print(x.shape)
        # x = x[0]

        return [x], [x_before]

    def loss_single(self, voxel_semantics, mask_camera, preds):
        loss_ = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            loss_occ = self.loss_occ(preds, voxel_semantics, mask_camera, avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics, )
            loss_['loss_occ'] = loss_occ
        return loss_

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    gt_masks_bev=None,
                    rescale=False,
                    radar=None,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, radar=radar[0], **kwargs)
        # img_feats = out_feats[0]
        # radar_feats = out_feats
        # print(img_feats[0].shape, radar_feats[0].shape)
        fusion_feats = self.reduc_conv(torch.cat((img_feats[0], radar_feats[0]), dim=1))
        occ_pred = self.final_conv(fusion_feats).permute(0, 4, 3, 2, 1)
        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
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
        img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        losses = dict()
        # loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        # losses['loss_depth'] = loss_depth
        # print(img_feats[0].shape, radar_feats[0].shape)
        # print(bev_feat_list[0].shape, prev_radar_feats[0].shape)
        fusion_feats = self.reduc_conv(torch.cat((img_feats[0], radar_feats[0]), dim=1))

        occ_pred = self.final_conv(fusion_feats).permute(0, 4, 3, 2, 1)  # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)

        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)

        if self.hop_load_all:
            num_frames = len(kwargs['hop_voxel_semantics']['semantic'])
            # select_frame=random.randint(0,num_frames-2) #这里为了避免超出范围
            select_frame = random.randint(1, num_frames - 1)  # 这里为了避免超出范围
            gt_semantic = kwargs['hop_voxel_semantics']['semantic'][select_frame]
            gt_mask_camera = kwargs['hop_mask_camera']['mask_camera'][select_frame]
            # bev_feat_list[-1*select_frame-1]=torch.zeros_like(img_feats[0])
            # history_feature=torch.cat(bev_feat_list,dim=1)
            # bev_feat_list[-1*select_frame-1]=torch.zeros_like(img_feats[0])

            # history_feature = bev_feat_list[:-1*select_frame-2] + bev_feat_list[-1*select_frame:]
            history_feature = bev_feat_list[:select_frame] + bev_feat_list[select_frame + 1:]
            history_feature = torch.cat(history_feature, dim=1)
            # print(select_frame, history_feature.shape)
            if self.long_term_backbone:
                pred_target_feat = self.long_term_backbone(history_feature)
            if self.long_term_neck:
                pred_target_feat = self.long_term_neck(pred_target_feat)

            long_radar_feats = self.radar_long_term_backbone(prev_radar_feats[0])
            long_radar_feats = self.radar_long_term_neck(long_radar_feats)

            long_fusion_feats = self.long_reduc_conv(torch.cat((pred_target_feat, long_radar_feats), dim=1))
            occ_pred = self.final_conv(long_fusion_feats)

            if self.use_predicter:
                occ_pred = self.predicter(occ_pred.permute(0, 4, 3, 2, 1))
            loss_occ = self.loss_single(gt_semantic, gt_mask_camera, occ_pred)
            losses['random_hop_loss_occ'] = loss_occ['loss_occ']

            if self.use_short:
                # short_feature=torch.cat([bev_feat_list[-1*select_frame-2],bev_feat_list[-1*select_frame]],dim=1)
                short_feature = torch.cat([bev_feat_list[select_frame - 1], bev_feat_list[select_frame + 1]], dim=1)
                pred_target_short = self.short_term_decoder(short_feature)

                short_radar_feats = self.radar_short_term_decoder(prev_radar_feats[0])

                short_fusion_feats = self.short_reduc_conv(torch.cat((pred_target_short, short_radar_feats), dim=1))
                # short_fusion_feats = self.short_reduc_conv(torch.cat((pred_target_short, long_radar_feats), dim=1))

                occ_pred = self.final_conv(short_fusion_feats)
                if self.use_predicter:
                    occ_pred = self.predicter(occ_pred.permute(0, 4, 3, 2, 1))
                loss_occ = self.loss_single(gt_semantic, gt_mask_camera, occ_pred)

                losses['random_short_loss_occ'] = loss_occ['loss_occ']


        elif self.with_hop:
            # not used
            assert False
            bev_feat_list[self.target_frame] = torch.zeros_like(img_feats[0])

            history_feature = torch.cat(bev_feat_list, dim=1)

            if self.long_term_backbone:
                pred_target_feat = self.long_term_backbone(history_feature)
            if self.long_term_neck:
                pred_target_feat = self.long_term_neck(pred_target_feat)

            occ_pred = self.final_conv(pred_target_feat)
            if self.use_predicter:
                occ_pred = self.predicter(occ_pred.permute(0, 4, 3, 2, 1))

            voxel_semantics = kwargs['hop_voxel_semantics']
            mask_camera = kwargs['hop_mask_camera']
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17

            loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
            losses['hop_loss_occ'] = loss_occ['loss_occ']

        return losses

    def extract_feat(self, points, img, img_metas, radar, **kwargs):
        """Extract features from images and points.
        Return:
        (BEV Feature, None, depth)
        """
        img_feats, depth, prev_feats = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None

        radar_feats, prev_radar_feats = self.extract_radar_feat(radar, img_metas)

        return (img_feats, pts_feats, depth, prev_feats, radar_feats, prev_radar_feats)

    def extract_img_feat(self,
                         img,
                         img_metas,
                         with_bevencoder=True,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            # Todo
            assert False
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(img, stereo=True)
        """Extract features of images."""
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame - 1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame - self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)
                if key_frame:
                    bev_feat, depth, feat_curr_iv = \
                        self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = \
                            self.prepare_bev_feat(*inputs_curr)
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv
        if pred_prev:
            # Todo
            assert False
        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) == 4:
                b, c, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1),
                                  h, w]).to(bev_feat_key), bev_feat_key]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1), z,
                                  h, w]).to(bev_feat_key), bev_feat_key]
        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame - 2):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame - 2 - adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        # print(bev_feat.shape)
        if with_bevencoder:
            x = self.bev_encoder(bev_feat)
            return [x], depth_key_frame, bev_feat_list
        else:
            return [bev_feat], depth_key_frame, bev_feat_list


@DETECTORS.register_module()
class BEVStereo4DOCCRC(BEVStereo4D):

    def __init__(self,

                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,

                 radar_voxel_layer=None,
                 radar_voxel_encoder=None,
                 radar_middle_encoder=None,
                 radar_bev_backbone=None,
                 radar_bev_neck=None,
                 radar_reduc_conv=False,  # new
                 imc=256, rac=64,  # im ra 特征维度
                 freeze_img=False,
                 sparse_shape=None,
                 ret_2d_feat=False,
                 **kwargs):
        super(BEVStereo4DOCCRC, self).__init__(**kwargs)
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
            self.img_view_transformer.out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d'))
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes),
            )
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False

        if radar_voxel_layer != None:
            self.radar_voxel_layer = Voxelization(**radar_voxel_layer)
        if radar_voxel_encoder != None:
            self.radar_voxel_encoder = builder.build_voxel_encoder(radar_voxel_encoder)
        if radar_middle_encoder != None:
            self.radar_middle_encoder = builder.build_middle_encoder(radar_middle_encoder)
        if radar_bev_backbone is not None:
            self.radar_bev_backbone = builder.build_backbone(radar_bev_backbone)
        if radar_bev_neck is not None:
            self.radar_bev_neck = builder.build_neck(radar_bev_neck)

        # voxel_channel = rac//2*5
        voxel_channel = imc * 2
        self.radar_bev_to_voxel_conv = nn.Conv2d(rac, voxel_channel * 16, kernel_size=1)

        if radar_reduc_conv:
            self.reduc_conv = ConvModule(
                voxel_channel + imc,
                # self.img_view_transformer.out_channels,  #rac change imc
                imc,
                kernel_size=3,
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

        self.freeze_img = freeze_img
        self.sparse_shape = sparse_shape
        self.ret_2d_feat = ret_2d_feat

    def init_weights(self):
        """Initialize model weights."""
        super(BEVStereo4DOCCRC, self).init_weights()
        if self.freeze_img:

            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

            for name, param in self.named_parameters():
                if 'img_view_transformer' in name:
                    param.requires_grad = False
                if 'img_bev_encoder_backbone' in name:
                    param.requires_grad = False
                if 'img_bev_encoder_neck' in name:
                    param.requires_grad = False
                if 'pre_process' in name:
                    param.requires_grad = False

            def fix_bn(m):
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                    m.track_running_stats = False

            self.img_view_transformer.apply(fix_bn)
            self.img_bev_encoder_backbone.apply(fix_bn)
            self.img_bev_encoder_neck.apply(fix_bn)

            self.img_backbone.apply(fix_bn)
            self.img_neck.apply(fix_bn)

            self.pre_process_net.apply(fix_bn)

    @torch.no_grad()
    @force_fp32()
    def radar_voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        # print(len(points))
        # assert len(points) == 3
        for res in points:
            res_voxels, res_coors, res_num_points = self.radar_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        # print(num_points)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            # print(coor.size())
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        # print(coors_batch[-1, 0] + 1)
        return voxels, num_points, coors_batch

    def extract_radar_feat(self, radar, img_metas):
        """Extract features of points."""
        voxels, num_points, coors = self.radar_voxelize(radar)

        voxel_features = self.radar_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        # print(batch_size)
        # batch_size = 5

        x_before = self.radar_middle_encoder(voxel_features, coors, batch_size)
        # coors = coors.int()
        # input_sp_tensor = SparseConvTensor(voxel_features, coors,
        #                                    self.sparse_shape, batch_size)

        # x_before = input_sp_tensor.dense()

        # if hasattr(self, 'radar_bev_backbone') and self.radar_bev_backbone is not None:
        # print(x_before.shape)
        x = self.radar_bev_backbone(x_before)  # 8, 64, h/2, w/2

        # if hasattr(self, 'radar_bev_neck') and self.radar_bev_neck is not None:
        x = self.radar_bev_neck(x)  # 8, 64, h/4, w/4

        x = torch.nn.functional.interpolate(x[0], scale_factor=2, mode='bilinear')
        # print(x.shape)
        # x = x[0]

        # if self.radar_bev_to_voxel is not None:
        x = self.radar_bev_to_voxel(x)
        return [x], [x_before]

    def radar_bev_to_voxel(self, x):
        x = self.radar_bev_to_voxel_conv(x)
        # x = x.reshape()
        bs, c, h, w = x.shape
        x = x.reshape(bs, c // 16, 16, h, w)

        return x

    def loss_single(self, voxel_semantics, mask_camera, preds):
        loss_ = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            loss_occ = self.loss_occ(preds, voxel_semantics, mask_camera, avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics, )
            loss_['loss_occ'] = loss_occ
        return loss_

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    gt_masks_bev=None,
                    rescale=False,
                    radar=None,
                    **kwargs):
        """Test function without augmentaiton."""
        # out_feats = self.extract_feat(
        #     points, img=img, img_metas=img_metas, radar=radar[0],**kwargs)
        # img_feats = out_feats[0]
        # occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
        if self.ret_2d_feat:
            img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats, feat_2d = self.extract_feat(
                points, img=img, img_metas=img_metas, radar=radar[0], **kwargs)
        else:
            img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats = self.extract_feat(
                points, img=img, img_metas=img_metas, radar=radar[0], **kwargs)
        # img_feats = out_feats[0]
        # radar_feats = out_feats
        fusion_feats = self.reduc_conv(torch.cat((img_feats[0], radar_feats[0]), dim=1))
        occ_pred = self.final_conv(fusion_feats).permute(0, 4, 3, 2, 1)
        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)

        vis = False
        if vis:
            # import ipdb; ipdb.set_trace()
            from tools.vis_occ_mask import draw_occ
            voxel_semantics = kwargs['voxel_semantics'][0][0].cpu().numpy().astype(np.uint8)
            occ_res[voxel_semantics == 17] = 17
            data_idx = kwargs['data_idx']
            print('get result, data_idx=', data_idx[0].item(), ', device=', fusion_feats.device)
            file_path_pred = '/home/xiazhongyu/bevperception/work_dirs/changanvis/occpred/%d.jpg' % data_idx[0].item()
            file_path_gt = '/home/xiazhongyu/bevperception/work_dirs/changanvis/occgt/%d.jpg' % data_idx[0].item()
            draw_occ(occ_res, file_path_pred)
            draw_occ(voxel_semantics, file_path_gt)

        res_dict = {
            'pts_occ': occ_res
        }

        return [res_dict]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
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
        if self.ret_2d_feat:
            img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats, feat_2d = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas, **kwargs)
        else:
            img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth
        # print(img_feats[0].shape, radar_feats[0].shape)
        # print(bev_feat_list[0].shape, prev_radar_feats[0].shape)
        # radar_feats_up = torch.nn.functional.interpolate(radar_feats[0], scale_factor=2, mode='bilinear')
        fusion_feats = self.reduc_conv(torch.cat((img_feats[0], radar_feats[0]), dim=1))

        occ_pred = self.final_conv(fusion_feats).permute(0, 4, 3, 2, 1)  # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)

        # import ipdb; ipdb.set_trace()
        # from tools.vis_occ_mask import draw_occ
        # draw_occ(torch.argmax(occ_pred, dim=-1)[0].detach().cpu(), 'pred.jpg')
        # voxel_semantics = kwargs['voxel_semantics']
        # draw_occ(voxel_semantics[0].cpu(), 'gt.jpg')

        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)

        # feat_2d torch.Size([1, 6, 256, 24, 44])

        # ipdb > pts_feats[0].shape
        # torch.Size([1, 48, 256, 64, 176])
        # ipdb > pts_feats[1].shape
        # torch.Size([1, 48, 256, 32, 88])
        # ipdb > pts_feats[2].shape
        # torch.Size([1, 48, 256, 16, 44])
        # ipdb > pts_feats[3].shape
        # torch.Size([1, 48, 256, 8, 22])

        return losses

    def extract_feat(self, points, img, img_metas, radar, **kwargs):
        """Extract features from images and points.
        Return:
        (BEV Feature, None, depth)
        """
        if self.ret_2d_feat:
            img_feats, depth, prev_feats, feat_2d = self.extract_img_feat(img, img_metas, **kwargs)
        else:
            img_feats, depth, prev_feats = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None

        radar_feats, prev_radar_feats = self.extract_radar_feat(radar, img_metas)

        if self.ret_2d_feat:
            return (img_feats, pts_feats, depth, prev_feats, radar_feats, prev_radar_feats, feat_2d)
        else:
            return (img_feats, pts_feats, depth, prev_feats, radar_feats, prev_radar_feats)

    def extract_img_feat(self,
                         img,
                         img_metas,
                         with_bevencoder=True,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            # Todo
            assert False
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(img, stereo=True)
        """Extract features of images."""
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame - 1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame - self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)
                if key_frame:
                    if self.ret_2d_feat:
                        bev_feat, depth, feat_curr_iv, feat_2d = \
                            self.prepare_bev_feat(*inputs_curr, ret_2d_feat=self.ret_2d_feat)
                    else:
                        bev_feat, depth, feat_curr_iv = \
                            self.prepare_bev_feat(*inputs_curr, ret_2d_feat=self.ret_2d_feat)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        if self.ret_2d_feat:
                            bev_feat, depth, feat_curr_iv, feat_2d = \
                                self.prepare_bev_feat(*inputs_curr, ret_2d_feat=self.ret_2d_feat)
                        else:
                            bev_feat, depth, feat_curr_iv = \
                                self.prepare_bev_feat(*inputs_curr, ret_2d_feat=self.ret_2d_feat)
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv
        if pred_prev:
            # Todo
            assert False
        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) == 4:
                b, c, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1),
                                  h, w]).to(bev_feat_key), bev_feat_key]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1), z,
                                  h, w]).to(bev_feat_key), bev_feat_key]
        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame - 2):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame - 2 - adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        # print(bev_feat.shape)
        if with_bevencoder:
            x = self.bev_encoder(bev_feat)
            if self.ret_2d_feat:
                return [x], depth_key_frame, bev_feat_list, feat_2d
            else:
                return [x], depth_key_frame, bev_feat_list,
        else:
            if self.ret_2d_feat:
                return [bev_feat], depth_key_frame, bev_feat_list, feat_2d
            else:
                return [bev_feat], depth_key_frame, bev_feat_list


@DETECTORS.register_module()
class BEVDepth4DOCCRC(BEVDepth4D):

    def __init__(self,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 radar_voxel_layer=None,
                 radar_voxel_encoder=None,
                 radar_middle_encoder=None,
                 radar_bev_backbone=None,
                 radar_bev_neck=None,
                 radar_reduc_conv=False,  # new
                 imc=256, rac=64,  # im ra 特征维度
                 freeze_img=False,
                 sparse_shape=None,
                 ret_2d_feat=False,
                 **kwargs):
        super(BEVDepth4DOCCRC, self).__init__(**kwargs)
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = nn.Conv2d(imc, out_channels * 16, kernel_size=1)
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes),
            )
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False

        if radar_voxel_layer != None:
            self.radar_voxel_layer = Voxelization(**radar_voxel_layer)
        if radar_voxel_encoder != None:
            self.radar_voxel_encoder = builder.build_voxel_encoder(radar_voxel_encoder)
        if radar_middle_encoder != None:
            self.radar_middle_encoder = builder.build_middle_encoder(radar_middle_encoder)
        if radar_bev_backbone is not None:
            self.radar_bev_backbone = builder.build_backbone(radar_bev_backbone)
        if radar_bev_neck is not None:
            self.radar_bev_neck = builder.build_neck(radar_bev_neck)

        # voxel_channel = rac//2*5
        self.radar_bev_to_voxel_conv = None
        if radar_reduc_conv:
            self.reduc_conv = ConvModule(
                rac + imc,
                # self.img_view_transformer.out_channels,  #rac change imc
                imc,
                kernel_size=3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

        self.freeze_img = freeze_img
        self.sparse_shape = sparse_shape
        self.ret_2d_feat = ret_2d_feat

    def init_weights(self):
        """Initialize model weights."""
        super(BEVDepth4DOCCRC, self).init_weights()
        if self.freeze_img:

            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

            for name, param in self.named_parameters():
                if 'img_view_transformer' in name:
                    param.requires_grad = False
                if 'img_bev_encoder_backbone' in name:
                    param.requires_grad = False
                if 'img_bev_encoder_neck' in name:
                    param.requires_grad = False
                if 'pre_process' in name:
                    param.requires_grad = False

            def fix_bn(m):
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                    m.track_running_stats = False

            self.img_view_transformer.apply(fix_bn)
            self.img_bev_encoder_backbone.apply(fix_bn)
            self.img_bev_encoder_neck.apply(fix_bn)

            self.img_backbone.apply(fix_bn)
            self.img_neck.apply(fix_bn)

            self.pre_process_net.apply(fix_bn)

    @torch.no_grad()
    @force_fp32()
    def radar_voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        # print(len(points))
        # assert len(points) == 3
        for res in points:
            res_voxels, res_coors, res_num_points = self.radar_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        # print(num_points)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            # print(coor.size())
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        # print(coors_batch[-1, 0] + 1)
        return voxels, num_points, coors_batch

    def extract_radar_feat(self, radar, img_metas):
        """Extract features of points."""
        voxels, num_points, coors = self.radar_voxelize(radar)

        voxel_features = self.radar_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        # print(batch_size)
        # batch_size = 5
        x_before = self.radar_middle_encoder(voxel_features, coors, batch_size)

        return [x_before]

    def radar_bev_to_voxel(self, x):
        x = self.radar_bev_to_voxel_conv(x)
        # x = x.reshape()
        bs, c, h, w = x.shape
        x = x.reshape(bs, c // 16, 16, h, w)

        return x

    def loss_single(self, voxel_semantics, mask_camera, preds):
        loss_ = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            loss_occ = self.loss_occ(preds, voxel_semantics, mask_camera, avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics, )
            loss_['loss_occ'] = loss_occ
        return loss_

    def bev_to_voxel(self, x):
        x = self.final_conv(x)
        # x = x.reshape()
        bs, c, h, w = x.shape
        x = x.reshape(bs, c // 16, 16, h, w)

        return x

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    gt_masks_bev=None,
                    rescale=False,
                    radar=None,
                    **kwargs):
        """Test function without augmentaiton."""
        # out_feats = self.extract_feat(
        #     points, img=img, img_metas=img_metas, radar=radar[0],**kwargs)
        # img_feats = out_feats[0]
        # occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
        if self.ret_2d_feat:
            img_feats, pts_feats, depth, bev_feat_list, radar_feats, feat_2d = self.extract_feat(
                points, img=img, img_metas=img_metas, radar=radar[0], **kwargs)
        else:
            img_feats, pts_feats, depth, bev_feat_list, radar_feats = self.extract_feat(
                points, img=img, img_metas=img_metas, radar=radar[0], **kwargs)
        # img_feats = out_feats[0]
        # radar_feats = out_feats
        fusion_feats = self.reduc_conv(torch.cat((img_feats[0], radar_feats[0]), dim=1))
        occ_pred = self.bev_to_voxel(fusion_feats).permute(0, 4, 3, 2, 1)
        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        res_dict = {
            'pts_occ': occ_res
        }
        return [res_dict]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
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
        if self.ret_2d_feat:
            img_feats, pts_feats, depth, bev_feat_list, radar_feats, feat_2d = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas, **kwargs)
        else:
            img_feats, pts_feats, depth, bev_feat_list, radar_feats = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth
        # print(img_feats[0].shape, radar_feats[0].shape)
        # print(bev_feat_list[0].shape, prev_radar_feats[0].shape)
        # radar_feats_up = torch.nn.functional.interpolate(radar_feats[0], scale_factor=2, mode='bilinear')
        fusion_feats = self.reduc_conv(torch.cat((img_feats[0], radar_feats[0]), dim=1))
        occ_pred = self.bev_to_voxel(fusion_feats).permute(0, 4, 3, 2, 1)  # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)

        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)

        # feat_2d torch.Size([1, 6, 256, 24, 44])

        # ipdb > pts_feats[0].shape
        # torch.Size([1, 48, 256, 64, 176])
        # ipdb > pts_feats[1].shape
        # torch.Size([1, 48, 256, 32, 88])
        # ipdb > pts_feats[2].shape
        # torch.Size([1, 48, 256, 16, 44])
        # ipdb > pts_feats[3].shape
        # torch.Size([1, 48, 256, 8, 22])

        return losses

    def extract_feat(self, points, img, img_metas, radar, **kwargs):
        """Extract features from images and points.
        Return:
        (BEV Feature, None, depth)
        """
        if self.ret_2d_feat:
            img_feats, depth, prev_feats, feat_2d = self.extract_img_feat(img, img_metas, **kwargs)
        else:
            img_feats, depth, prev_feats = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None

        radar_feats = self.extract_radar_feat(radar, img_metas)

        if self.ret_2d_feat:
            return (img_feats, pts_feats, depth, prev_feats, radar_feats, feat_2d)
        else:
            return (img_feats, pts_feats, depth, prev_feats, radar_feats)

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.contiguous().view(B * N, C, imH, imW)
        feat_2d = self.img_backbone(imgs)
        x = (feat_2d[0], feat_2d[2], feat_2d[3])
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
        x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        if self.ret_2d_feat:
            return x, stereo_feat, feat_2d
        else:
            return x, stereo_feat
            
    def extract_img_feat(self,
                         img,
                         img_metas,
                         with_bevencoder=True,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            # Todo
            assert False
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(img, stereo=False)
        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        key_frame = True  # back propagation for key frame only
        for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
                imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input)
                if key_frame:
                    if self.ret_2d_feat:
                        bev_feat, depth, feat_2d = \
                            self.prepare_bev_feat(*inputs_curr, ret_2d_feat=self.ret_2d_feat)
                    else:
                        bev_feat, depth = \
                            self.prepare_bev_feat(*inputs_curr, ret_2d_feat=self.ret_2d_feat)                
                else:
                    with torch.no_grad():
                        if self.ret_2d_feat:
                            bev_feat, depth, feat_2d = \
                                self.prepare_bev_feat(*inputs_curr, ret_2d_feat=self.ret_2d_feat)
                        else:
                            bev_feat, depth = \
                                self.prepare_bev_feat(*inputs_curr, ret_2d_feat=self.ret_2d_feat)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False
        if pred_prev:
            # Todo
            assert False
        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        # print(bev_feat.shape)
        if with_bevencoder:
            x = self.bev_encoder(bev_feat)
            if self.ret_2d_feat:
                return [x], depth_list[0], bev_feat_list, feat_2d
            else:
                return [x], depth_list[0], bev_feat_list,
        else:
            if self.ret_2d_feat:
                return [bev_feat], depth_list[0], bev_feat_list, feat_2d
            else:
                return [bev_feat], depth_list[0], bev_feat_list


@DETECTORS.register_module()
class BEVStereo4DOCCHopRCV2(BEVStereo4D):

    def __init__(self,

                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,

                 with_hop=False,
                 hop_cfg=None,
                 hop_load_all=False,
                 use_short=False,

                 radar_voxel_layer=None,
                 radar_voxel_encoder=None,
                 radar_middle_encoder=None,
                 radar_bev_backbone=None,
                 radar_bev_neck=None,
                 radar_reduc_conv=False,  # new
                 imc=256, rac=64,  # im ra 特征维度
                 freeze_img=False,
                 sparse_shape=None,

                 **kwargs):
        super(BEVStereo4DOCCHopRCV2, self).__init__(**kwargs)
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
            self.img_view_transformer.out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d'))
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes),
            )
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False

        self.with_hop = with_hop
        if self.with_hop:
            self.hop_cfg = hop_cfg
            self.long_term_backbone = build_backbone(self.hop_cfg.long_term_backbone)
            self.long_term_neck = build_neck(self.hop_cfg.long_term_neck)

            self.target_frame = self.hop_cfg.target_frame

            # self.radar_long_term_backbone=build_backbone(self.hop_cfg.radar_long_term_backbone)
            # self.radar_long_term_neck=build_neck(self.hop_cfg.radar_long_term_neck)

        self.hop_load_all = hop_load_all

        self.use_short = use_short
        if self.use_short:
            self.short_term_decoder = nn.Sequential(
                nn.Conv3d(in_channels=32 * 2, out_channels=128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=128, out_channels=32, kernel_size=3, padding=1)
            )

            # self.radar_short_term_decoder=nn.Sequential(
            #     nn.Conv3d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            #     nn.ReLU(),
            #     nn.Conv3d(in_channels=128,out_channels=32,kernel_size=3,padding=1)
            # )

        if radar_voxel_layer != None:
            self.radar_voxel_layer = Voxelization(**radar_voxel_layer)
        if radar_voxel_encoder != None:
            self.radar_voxel_encoder = builder.build_voxel_encoder(radar_voxel_encoder)
        if radar_middle_encoder != None:
            self.radar_middle_encoder = builder.build_middle_encoder(radar_middle_encoder)
        if radar_bev_backbone is not None:
            self.radar_bev_backbone = builder.build_backbone(radar_bev_backbone)
        if radar_bev_neck is not None:
            self.radar_bev_neck = builder.build_neck(radar_bev_neck)

        voxel_channel = imc * 2
        self.radar_bev_to_voxel_conv = nn.Conv2d(rac, voxel_channel * 16, kernel_size=1)

        if radar_reduc_conv:
            self.reduc_conv = ConvModule(
                voxel_channel + imc,
                self.img_view_transformer.out_channels,  # rac change imc
                kernel_size=3,
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

            if self.with_hop:
                self.long_reduc_conv = ConvModule(
                    self.img_view_transformer.out_channels + voxel_channel,
                    # self.hop_cfg.radar_long_term_neck.out_channels,
                    self.img_view_transformer.out_channels,  # rac change imc
                    kernel_size=3,
                    padding=1,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
                    act_cfg=dict(type='ReLU'),
                    inplace=False)

                if self.use_short:
                    self.short_reduc_conv = ConvModule(
                        32 + voxel_channel,
                        # 32,
                        32,  # rac change imc
                        kernel_size=3,
                        padding=1,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
                        act_cfg=dict(type='ReLU'),
                        inplace=False)

        self.freeze_img = freeze_img
        self.sparse_shape = sparse_shape

    def init_weights(self):
        """Initialize model weights."""
        super(BEVStereo4DOCCHopRCV2, self).init_weights()
        if self.freeze_img:

            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

            for name, param in self.named_parameters():
                if 'img_view_transformer' in name:
                    param.requires_grad = False
                if 'img_bev_encoder_backbone' in name:
                    param.requires_grad = False
                if 'img_bev_encoder_neck' in name:
                    param.requires_grad = False
                if 'pre_process' in name:
                    param.requires_grad = False

            def fix_bn(m):
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                    m.track_running_stats = False

            self.img_view_transformer.apply(fix_bn)
            self.img_bev_encoder_backbone.apply(fix_bn)
            self.img_bev_encoder_neck.apply(fix_bn)

            self.img_backbone.apply(fix_bn)
            self.img_neck.apply(fix_bn)

            self.pre_process_net.apply(fix_bn)

    @torch.no_grad()
    @force_fp32()
    def radar_voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        # print(len(points))
        # assert len(points) == 3
        for res in points:
            res_voxels, res_coors, res_num_points = self.radar_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        # print(num_points)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            # print(coor.size())
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        # print(coors_batch[-1, 0] + 1)
        return voxels, num_points, coors_batch

    def extract_radar_feat(self, radar, img_metas):
        """Extract features of points."""
        voxels, num_points, coors = self.radar_voxelize(radar)

        voxel_features = self.radar_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        # print(batch_size)
        # batch_size = 5

        x_before = self.radar_middle_encoder(voxel_features, coors, batch_size)
        # coors = coors.int()
        # input_sp_tensor = SparseConvTensor(voxel_features, coors,
        #    self.sparse_shape, batch_size)

        # x_before = input_sp_tensor.dense()

        # if hasattr(self, 'radar_bev_backbone') and self.radar_bev_backbone is not None:
        # print(x_before.shape)
        x = self.radar_bev_backbone(x_before)  # 8, 64, h/2, w/2

        # if hasattr(self, 'radar_bev_neck') and self.radar_bev_neck is not None:
        x = self.radar_bev_neck(x)  # 8, 64, h/4, w/4
        x = torch.nn.functional.interpolate(x[0], scale_factor=2, mode='bilinear')
        # print(x.shape)

        x = self.radar_bev_to_voxel(x)
        return [x], [x_before]

    def radar_bev_to_voxel(self, x):
        x = self.radar_bev_to_voxel_conv(x)
        # x = x.reshape()
        bs, c, h, w = x.shape
        x = x.reshape(bs, c // 16, 16, h, w)

        return x

    def loss_single(self, voxel_semantics, mask_camera, preds):
        loss_ = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            loss_occ = self.loss_occ(preds, voxel_semantics, mask_camera, avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics, )
            loss_['loss_occ'] = loss_occ
        return loss_

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    gt_masks_bev=None,
                    rescale=False,
                    radar=None,
                    **kwargs):
        """Test function without augmentaiton."""
        # out_feats = self.extract_feat(
        #     points, img=img, img_metas=img_metas, radar=radar[0],**kwargs)
        # img_feats = out_feats[0]
        # occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
        img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, radar=radar[0], **kwargs)
        # img_feats = out_feats[0]
        # radar_feats = out_feats
        fusion_feats = self.reduc_conv(torch.cat((img_feats[0], radar_feats[0]), dim=1))
        occ_pred = self.final_conv(fusion_feats).permute(0, 4, 3, 2, 1)
        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
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
        img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth
        # print(img_feats[0].shape, radar_feats[0].shape)
        # print(bev_feat_list[0].shape, prev_radar_feats[0].shape)
        fusion_feats = self.reduc_conv(torch.cat((img_feats[0], radar_feats[0]), dim=1))

        occ_pred = self.final_conv(fusion_feats).permute(0, 4, 3, 2, 1)  # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)

        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)

        if self.hop_load_all:
            num_frames = len(kwargs['hop_voxel_semantics']['semantic'])
            # select_frame=random.randint(0,num_frames-2) #这里为了避免超出范围
            select_frame = random.randint(1, num_frames - 1)  # 这里为了避免超出范围
            gt_semantic = kwargs['hop_voxel_semantics']['semantic'][select_frame]
            gt_mask_camera = kwargs['hop_mask_camera']['mask_camera'][select_frame]
            # bev_feat_list[-1*select_frame-1]=torch.zeros_like(img_feats[0])
            # history_feature=torch.cat(bev_feat_list,dim=1)
            # bev_feat_list[-1*select_frame-1]=torch.zeros_like(img_feats[0])

            # history_feature = bev_feat_list[:-1*select_frame-2] + bev_feat_list[-1*select_frame:]
            history_feature = bev_feat_list[:select_frame] + bev_feat_list[select_frame + 1:]
            history_feature = torch.cat(history_feature, dim=1)
            # print(select_frame, history_feature.shape)
            if self.long_term_backbone:
                pred_target_feat = self.long_term_backbone(history_feature)
            if self.long_term_neck:
                pred_target_feat = self.long_term_neck(pred_target_feat)

            # long_radar_feats = self.radar_long_term_backbone(prev_radar_feats[0])
            # long_radar_feats = self.radar_long_term_neck(long_radar_feats)
            long_radar_feats = radar_feats[0]

            long_fusion_feats = self.long_reduc_conv(torch.cat((pred_target_feat, long_radar_feats), dim=1))
            occ_pred = self.final_conv(long_fusion_feats)

            if self.use_predicter:
                occ_pred = self.predicter(occ_pred.permute(0, 4, 3, 2, 1))
            loss_occ = self.loss_single(gt_semantic, gt_mask_camera, occ_pred)
            losses['random_hop_loss_occ'] = loss_occ['loss_occ']

            if self.use_short:
                # short_feature=torch.cat([bev_feat_list[-1*select_frame-2],bev_feat_list[-1*select_frame]],dim=1)
                short_feature = torch.cat([bev_feat_list[select_frame - 1], bev_feat_list[select_frame + 1]], dim=1)
                pred_target_short = self.short_term_decoder(short_feature)

                # short_radar_feats = self.radar_short_term_decoder(prev_radar_feats[0])
                short_radar_feats = radar_feats[0]

                short_fusion_feats = self.short_reduc_conv(torch.cat((pred_target_short, short_radar_feats), dim=1))
                # short_fusion_feats = self.short_reduc_conv(torch.cat((pred_target_short, long_radar_feats), dim=1))

                occ_pred = self.final_conv(short_fusion_feats)
                if self.use_predicter:
                    occ_pred = self.predicter(occ_pred.permute(0, 4, 3, 2, 1))
                loss_occ = self.loss_single(gt_semantic, gt_mask_camera, occ_pred)

                losses['random_short_loss_occ'] = loss_occ['loss_occ']


        elif self.with_hop:
            # not used
            assert False
            bev_feat_list[self.target_frame] = torch.zeros_like(img_feats[0])

            history_feature = torch.cat(bev_feat_list, dim=1)

            if self.long_term_backbone:
                pred_target_feat = self.long_term_backbone(history_feature)
            if self.long_term_neck:
                pred_target_feat = self.long_term_neck(pred_target_feat)

            occ_pred = self.final_conv(pred_target_feat)
            if self.use_predicter:
                occ_pred = self.predicter(occ_pred.permute(0, 4, 3, 2, 1))

            voxel_semantics = kwargs['hop_voxel_semantics']
            mask_camera = kwargs['hop_mask_camera']
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17

            loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
            losses['hop_loss_occ'] = loss_occ['loss_occ']

        return losses

    def extract_feat(self, points, img, img_metas, radar, **kwargs):
        """Extract features from images and points.
        Return:
        (BEV Feature, None, depth)
        """
        img_feats, depth, prev_feats = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None

        radar_feats, prev_radar_feats = self.extract_radar_feat(radar, img_metas)

        return (img_feats, pts_feats, depth, prev_feats, radar_feats, prev_radar_feats)

    def extract_img_feat(self,
                         img,
                         img_metas,
                         with_bevencoder=True,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            # Todo
            assert False
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(img, stereo=True)
        """Extract features of images."""
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame - 1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame - self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)
                if key_frame:
                    bev_feat, depth, feat_curr_iv = \
                        self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = \
                            self.prepare_bev_feat(*inputs_curr)
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv
        if pred_prev:
            # Todo
            assert False
        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) == 4:
                b, c, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1),
                                  h, w]).to(bev_feat_key), bev_feat_key]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1), z,
                                  h, w]).to(bev_feat_key), bev_feat_key]
        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame - 2):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame - 2 - adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        # print(bev_feat.shape)
        if with_bevencoder:
            x = self.bev_encoder(bev_feat)
            return [x], depth_key_frame, bev_feat_list
        else:
            return [bev_feat], depth_key_frame, bev_feat_list
