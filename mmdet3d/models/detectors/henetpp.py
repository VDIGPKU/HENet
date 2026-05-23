import torch.nn.functional as F
from .bevdet_rc_occ import BEVStereo4DOCCRC
from ..sparsebev.utils import GridMask, pad_multiple, GpuPhotoMetricDistortion
import queue
from mmdet3d.core import bbox3d2result
from mmdet.models.backbones.resnet import ResNet
from mmdet3d.models.backbones import VovNetFPN, VoVNet_s, ResNet_withcp
from .bevdet import BEVStereo4D
import torch
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from torch import nn
import numpy as np
from .. import builder
from mmcv.runner import force_fp32
from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2WithZ, TRTBEVPoolv2
from collections import defaultdict


@DETECTORS.register_module()
class HenetppRC(BEVStereo4DOCCRC):

    def __init__(self, neck_det=None, data_aug=None, pts_bbox_head=None, train_cfg=None, test_cfg=None,
                 stop_prev_grad=0, **kwargs):
        super(HenetppRC, self).__init__(**kwargs)
        self.stop_prev_grad = stop_prev_grad
        self.color_aug = GpuPhotoMetricDistortion()
        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = True
        self.data_aug = data_aug
        if neck_det is not None:
            self.neck_det = builder.build_neck(neck_det)
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
        self.memory = {}
        self.queue = queue.Queue()

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.contiguous().view(B * N, C, imH, imW)
        # with torch.no_grad():
        feat_2d = self.img_backbone(imgs)
        if isinstance(self.img_backbone, VoVNet_s):
            x = (feat_2d['stage2'], feat_2d['stage4'], feat_2d['stage5'])
        elif isinstance(self.img_backbone, ResNet) or isinstance(self.img_backbone, ResNet_withcp):
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

    def extract_stereo_ref_feat(self, x):
        B, N, C, imH, imW = x.shape
        if isinstance(self.img_backbone, VoVNet_s):
            # with torch.no_grad():
            x = x.view(B * N, C, imH, imW)
            x = self.img_backbone(x)
            return x['stage2']
        else:
            return super(HenetppRC, self).extract_stereo_ref_feat(x)

    def extract_img_feat_mf(self, img):
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.neck_det(img_feats)

        return img_feats

    def extract_feat_mf(self, img, img_metas):
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        assert img.dim() == 5

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img = img.float()

        # move some augmentations to GPU
        if self.data_aug is not None:
            if 'img_color_aug' in self.data_aug and self.data_aug['img_color_aug'] and self.training:
                img = self.color_aug(img)

            if 'img_norm_cfg' in self.data_aug:
                img_norm_cfg = self.data_aug['img_norm_cfg']

                norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
                norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

                if img_norm_cfg['to_rgb']:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            for b in range(B):
                img_shape = (img.shape[2], img.shape[3], img.shape[1])
                img_metas[b]['img_shape'] = [img_shape for _ in range(N)]
                img_metas[b]['ori_shape'] = [img_shape for _ in range(N)]

            if 'img_pad_cfg' in self.data_aug:
                img_pad_cfg = self.data_aug['img_pad_cfg']
                img = pad_multiple(img, img_metas, size_divisor=img_pad_cfg['size_divisor'])

        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if self.training and self.stop_prev_grad > 0:
            H, W = input_shape
            img = img.reshape(B, -1, 6, C, H, W)

            img_grad = img[:, :self.stop_prev_grad]
            img_nograd = img[:, self.stop_prev_grad:]

            all_img_feats = [self.extract_img_feat_mf(img_grad.reshape(-1, C, H, W))]

            with torch.no_grad():
                self.eval()
                for k in range(img_nograd.shape[1]):
                    all_img_feats.append(self.extract_img_feat_mf(img_nograd[:, k].reshape(-1, C, H, W)))
                self.train()

            img_feats = []
            for lvl in range(len(all_img_feats[0])):
                C, H, W = all_img_feats[0][lvl].shape[1:]
                img_feat = torch.cat([feat[lvl].reshape(B, -1, 6, C, H, W) for feat in all_img_feats], dim=1)
                img_feat = img_feat.reshape(-1, C, H, W)
                img_feats.append(img_feat)
        else:
            img_feats = self.extract_img_feat_mf(img)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function."""
        # import ipdb;ipdb.set_trace()
        # print(gt_labels_3d[0].dtype, gt_labels_3d[0])
        if self.ret_2d_feat:
            img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats, feat_2d = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas, **kwargs)
            feat_2d_mf = self.neck_det(feat_2d) + 'cached feat_2d'
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

        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)

        if not self.ret_2d_feat:
            feat_2d_mf = self.extract_feat_mf(img, img_metas)
        for i in range(len(img_metas)):
            img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i]
            img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]

        det_outs = self.pts_bbox_head(feat_2d_mf, img_metas, **kwargs)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, det_outs]
        loss_det = self.pts_bbox_head.loss(*loss_inputs)
        losses.update(loss_det)

        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img_input=None,
                    gt_masks_bev=None,
                    rescale=False,
                    radar=None,
                    img=None,
                    **kwargs):
        """Test function without augmentaiton."""

        img = [img] if img is None else img
        img = img[0]

        if self.ret_2d_feat:
            img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats, feat_2d = self.extract_feat(
                points, img=img_input, img_metas=img_metas, radar=radar[0], **kwargs)
            feat_2d_mf = self.neck_det(feat_2d) + 'cached self.neck_det(feat_2d)'
        else:
            img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats = self.extract_feat(
                points, img=img_input, img_metas=img_metas, radar=radar[0], **kwargs)

        fusion_feats = self.reduc_conv(torch.cat((img_feats[0], radar_feats[0]), dim=1))
        occ_pred = self.final_conv(fusion_feats).permute(0, 4, 3, 2, 1)
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)

        feat_2d_mf = self.extract_feat_mf(img, img_metas)
        outs = self.pts_bbox_head(feat_2d_mf, img_metas, **kwargs)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas[0], rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        res_dict = {
            'pts_bbox': bbox_results[0],
            'pts_occ': occ_res
        }

        return [res_dict]


    def extract_img_feat_sequential(self,
                         img,
                         img_metas,
                         with_bevencoder=True,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
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
                            self.prepare_bev_feat(*inputs_curr, ret_2d_feat=self.ret_2d_feat, for_henetppbev=True)
                    else:
                        bev_feat, depth, feat_curr_iv = \
                            self.prepare_bev_feat(*inputs_curr, ret_2d_feat=self.ret_2d_feat, for_henetppbev=True)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        if self.ret_2d_feat:
                            bev_feat, depth, feat_curr_iv, feat_2d = \
                                self.prepare_bev_feat(*inputs_curr, ret_2d_feat=self.ret_2d_feat, for_henetppbev=True)
                        else:
                            bev_feat, depth, feat_curr_iv = \
                                self.prepare_bev_feat(*inputs_curr, ret_2d_feat=self.ret_2d_feat, for_henetppbev=True)
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                if not key_frame:
                    feat_prev_iv = feat_curr_iv

        bev_feat = torch.cat(bev_feat_list[:self.num_frame - 2], dim=1)
        return bev_feat, feat_prev_iv, feat_2d

    def get_bev_pool_input(self, input):
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(input, stereo=True, flag=True)
        mlp_input_list = []
        metas_list = []
        grid_list = []
        feat_prev_iv = None
        for fid in range(self.num_frame - 1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame - self.extra_ref_frames
            # sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0] # assert align_after_view_transformation==True
            mlp_input = self.img_view_transformer.get_mlp_input(
                sensor2keyegos[0], ego2globals[0], intrin,
                post_rot, post_tran, bda)
            if key_frame:
                mlp_input_list.append(mlp_input)

            inputs_curr = (img, sensor2keyego, ego2global, intrin,
                            post_rot, post_tran, bda, mlp_input,
                            feat_prev_iv, curr2adjsensor[fid],
                            extra_ref_frame)
            x, stereo_feat, _ = self.image_encoder(img, stereo=True)
            feat_prev_iv = stereo_feat # stereo_feat == feat_curr_iv
            if extra_ref_frame:
                continue
            # generate gen_grid results, stereo only
            metas = dict(k2s_sensor=curr2adjsensor[fid],
                        intrins=intrin,
                        post_rots=post_rot,
                        post_trans=post_tran,
                        frustum=self.img_view_transformer.cv_frustum.to(x),
                        cv_downsample=4,
                        downsample=self.img_view_transformer.downsample,
                        grid_config=self.img_view_transformer.grid_config,
                        cv_feat_list=[feat_prev_iv, stereo_feat])
            prev, curr = metas['cv_feat_list']
            group_size = 4
            _, c, hf, wf = curr.shape
            hi, wi = hf * 4, wf * 4
            B, N, _ = metas['post_trans'].shape
            D, H, W, _ = metas['frustum'].shape
            grid = self.img_view_transformer.depth_net.gen_grid(metas, B, N, D, H, W, hi, wi).to(curr.dtype)
            if key_frame:
                grid_list.append(grid)

            # generate voxel_pooling_prepare_v2 results (which use get_lidar_coor)
            coor = self.img_view_transformer.get_lidar_coor(sensor2keyego, ego2global, intrin, post_rot, post_tran, bda)
            metas = self.img_view_transformer.voxel_pooling_prepare_v2(coor)
            if key_frame:
                metas_list.append(metas)
        return imgs, mlp_input_list, metas_list, grid_list

