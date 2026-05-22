
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from torch import nn as nn

from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from mmdet.models.backbones.resnet import ResNet
from mmcv.ops import Voxelization
from mmcv.cnn import ConvModule, xavier_init
from tools.misc.vis_tools import print_gt_and_bev,print_pcgt_on_bev
from mmdet3d.models.backbones import VovNetFPN, SwinTransformer
from collections import OrderedDict

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

@DETECTORS.register_module()
class BEVMAEPP_LRC_BEVDet(CenterPoint):
    r"""BEVDet paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
        imc (int): channel dimension of camera BEV feature.
        lic (int): channel dimension of radar BEV feature.
    """

    def __init__(
                self,
                img_view_transformer, 
                img_bev_encoder_backbone=None,
                img_bev_encoder_neck=None,
                radar_voxel_layer=None,
                radar_voxel_encoder=None,
                radar_middle_encoder=None,
                radar_bev_backbone=None,
                radar_bev_neck=None,
                reduc_conv=None, #new
                se=True,
                imc=256, rac=64, #im ra 特征维度,
                lic=256,
                freeze_img=False,
                freeze_radar=False,
                freeze_lidar=False,
                module_fusion=['L', 'R', 'C'],
                lidar_ckpt=None,
                radar_ckpt=None,
                cam_ckpt=None,
                interpolate_feat=False,
                **kwargs):
        super(BEVMAEPP_LRC_BEVDet, self).__init__(**kwargs)
        self.module_fusion = module_fusion
        self.lidar_ckpt=lidar_ckpt
        self.cam_ckpt=cam_ckpt
        self.radar_ckpt=radar_ckpt
        self.interpolate_feat = interpolate_feat
        self.p_step = False


        self.img_view_transformer = builder.build_neck(img_view_transformer)
        if img_bev_encoder_backbone is None:
            self.with_bevencoder = False
            print('warning!!!!! not use BEVencoder for img')
        else:
            self.with_bevencoder = True
            self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
            self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
        #new
        if radar_voxel_layer!=None:
            self.radar_voxel_layer = Voxelization(**radar_voxel_layer)
        if radar_voxel_encoder!=None:
            self.radar_voxel_encoder = builder.build_voxel_encoder(radar_voxel_encoder)
        if radar_middle_encoder!=None:
            self.radar_middle_encoder = builder.build_middle_encoder(radar_middle_encoder)
        if radar_bev_backbone is not None:
            self.radar_bev_backbone = builder.build_backbone(radar_bev_backbone)
        if radar_bev_neck is not None:
            self.radar_bev_neck = builder.build_neck(radar_bev_neck)
        # # if reduc_conv!=None:
        # input_feat_c = 0
        # output_feat_c = 0
        # if 'R' in module_fusion:
        #     input_feat_c += rac
        #     output_feat_c = rac
        # if 'C' in module_fusion:
        #     input_feat_c += imc
        #     output_feat_c = imc
        # if 'L' in module_fusion:
        #     input_feat_c += lic
        #     output_feat_c = lic

        # self.reduc_conv = ConvModule(
        #         input_feat_c,
        #         output_feat_c,  #rac change imc
        #         kernel_size=3,
        #         padding=1,
        #         conv_cfg=None,
        #         norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        #         act_cfg=dict(type='ReLU'),
        #         inplace=False)
        # # self.se=None
        # if se:
        #     print('Using SE Block for Fusion!!!!!!!!!!!!!!!!')
        #     self.se = SE_Block(output_feat_c)
        
        self.freeze_img=freeze_img
        self.freeze_radar=freeze_radar

        self.freeze_lidar = freeze_lidar

    def extract_pts_feat(self, pts, img_feats=None, img_metas=None):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        bev_x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        # print(bev_x.shape, bev_x.sum(dim=1).min())
        if self.with_pts_backbone:
            x = self.pts_backbone(bev_x)
            if self.with_pts_neck:
                x = self.pts_neck(x)
        else:
            x = bev_x
        return x, bev_x.detach()
    

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        # print("imgs!!", imgs.shape)
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        # if stereo:
        #     print("stereo_feat!!", stereo_feat.shape)
        # print("x!!!!!!", x.shape)
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat
    

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
        for res in points:
            res_voxels, res_coors, res_num_points = self.radar_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch



    @force_fp32()
    def bev_encoder(self, x):
        if hasattr(self, 'img_down2top_encoder_backbone'):
            x = self.img_down2top_encoder_backbone(x)
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def extract_img_feat(self, img, img_metas, with_bevencoder=True, **kwargs):
        """Extract features of images."""
        img = self.prepare_inputs(img)
        x, _ = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:7])
        if with_bevencoder:
            x = self.bev_encoder(x)
        return [x], depth

    def extract_radar_feat(self, radar, img_metas):
        """Extract features of points."""
        #data1 = open("radar_type.txt",'w',encoding="utf-8")
        #print(type(radar),file=data1)    
        #训练单括号list 测试双括号list
        # if self.test_cfg != None :
        #radar=radar[0] #test 时临时
        voxels, num_points, coors = self.radar_voxelize(radar)

        voxel_features = self.radar_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        
        x = self.radar_middle_encoder(voxel_features, coors, batch_size)

        if hasattr(self, 'radar_bev_backbone') and self.radar_bev_backbone is not None:
            # print(x.size()) 
            x = self.radar_bev_backbone(x) # 8, 64, h/2, w/2
        
        if hasattr(self, 'radar_bev_neck') and self.radar_bev_neck is not None:
            # print(len(x), x[0].size())
            x = self.radar_bev_neck(x) # 8, 64, h/4, w/4
            # print(len(x), x[0].size())
            x = x[0]

        # if hasattr(self, 'se') and self.se is not None:
        #     x = self.se(x)

        return [x]

    def extract_feat(self, points, img, img_metas, radar, with_bevencoder=True, **kwargs):  #add gt
        """Extract features from images and points."""
        pts_feats = None
        img_feats = None
        radar_feats = None
        if 'L' in self.module_fusion:
            pts_feats, pts_bev_feat = self.extract_pts_feat(points, None, img_metas)

        if 'C' in self.module_fusion:
            img_feats, depth = self.extract_img_feat(img, img_metas, with_bevencoder=with_bevencoder, **kwargs)
        
        if 'R' in self.module_fusion:
            radar_feats = self.extract_radar_feat(radar, img_metas) #new
        
        if self.interpolate_feat:
            img_feats[0] = torch.nn.functional.interpolate(img_feats[0], size=pts_feats[0].shape[-2:])


        return img_feats, pts_feats, depth, pts_bev_feat

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_masks_bev=None,
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
        img_feats, pts_feats, _, before_bev_feat = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, with_bevencoder=self.with_bevencoder, **kwargs)
        losses = dict()
        # losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
        #                                     gt_labels_3d, gt_masks_bev,
        #                                     img_metas, gt_bboxes_ignore)
        # losses.update(losses_pts)
        before_bev_feat = before_bev_feat.sum(dim=1).unsqueeze(dim=1)
        mask_tensor = torch.zeros_like(before_bev_feat).detach()
        print(mask_tensor)
        print(before_bev_feat)
        mask_tensor = (mask_tensor != before_bev_feat)
        mask_tensor = mask_tensor.detach()

        outs = self.pts_bbox_head(img_feats, pts_feats, self.p_step, mask_tensor)
        # loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses.update(outs)

        self.p_step = 1 - self.p_step

        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     gt_masks_bev=None,
                     radar=None,
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
            return self.simple_test(points[0], img_metas[0], radar[0], img_inputs[0], gt_masks_bev, **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_masks_bev,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_masks_bev (list[torch.Tensor]): Ground truth labels for bev segmentation
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        loss_dict = dict()
        if self.pts_bbox_head:
            outs = self.pts_bbox_head(pts_feats)
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)
            loss_dict.update(losses)
        if self.pts_seg_head:
            losses = self.pts_seg_head(pts_feats, gt_masks_bev)
            loss_dict.update(losses)
        # print(loss_dict)
        return loss_dict

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                    points,
                    img_metas,
                    radar,
                    img=None,
                    gt_masks_bev=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""

        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, radar=radar, with_bevencoder=True, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        if self.pts_bbox_head:
            bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale) 
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
            # debug
            if False:
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                import random
                from datetime import datetime
                print('\n******** BEGIN PRINT PRED**********\n')
                fig = plt.figure(figsize=(16, 16))
                plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
                plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
                box = bbox_list[0]['pts_bbox']['boxes_3d']
                corner = box.corners
                for i in range(corner.shape[0]):
                    x1 = corner[i][0][0]
                    y1 = corner[i][0][1]
                    x2 = corner[i][2][0]
                    y2 = corner[i][2][1]
                    x3 = corner[i][6][0]
                    y3 = corner[i][6][1]
                    x4 = corner[i][4][0]
                    y4 = corner[i][4][1]
                    plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], lw=0.5)
                plt.savefig("/home/wangxinhao/vis/"+datetime.now().strftime('%H:%M:%S')+'--scene='+img_metas[0]['scene_token']+'--idx='+img_metas[0]['sample_idx']+".png")
                print('\n******** END PRINT GT**********\n')
        if self.pts_seg_head:
            bbox_segs = self.pts_seg_head(img_feats, gt_masks_bev)
            for result_dict, pts_seg, gt in zip(bbox_list, bbox_segs, gt_masks_bev):
                result_dict['pts_seg'] = pts_seg
                result_dict['gt_masks_bev'] = gt
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, with_bevencoder=True, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs


@DETECTORS.register_module()
class BEVMAEPP_LRC_BEVDet4D(BEVMAEPP_LRC_BEVDet):
    r"""BEVDet4D paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2203.17054>`_

    Args:
        pre_process (dict | None): Configuration dict of BEV pre-process net.
        align_after_view_transfromation (bool): Whether to align the BEV
            Feature after view transformation. By default, the BEV feature of
            the previous frame is aligned during the view transformation.
        num_adj (int): Number of adjacent frames.
        with_prev (bool): Whether to set the BEV feature of previous frame as
            all zero. By default, False.
    """
    def __init__(self,
                 pre_process=None,
                 align_after_view_transfromation=False,
                 num_adj=1,
                 with_prev=True,
                 **kwargs):
        super(BEVMAEPP_LRC_BEVDet4D, self).__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1

        self.with_prev = with_prev
        self.grid = None
    
    def load_ckpt(self, ckpt):
        checkpoint = torch.load(ckpt, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        tmp = self.load_state_dict(state_dict, strict=False)
        print(f"LOAD {ckpt}:")
        # print(tmp.__class__)
        # print(tmp)
        # print(tmp[0])
        return tmp

    # def load_ckpt(self, ckpt_list):
    #     all_state_dict=OrderedDict()
    #     for ckpt in ckpt_list:
    #         if ckpt is not None:
    #             print(f"LOAD {ckpt}:")
    #             checkpoint = torch.load(ckpt, map_location='cpu')
    #             if 'state_dict' in checkpoint:
    #                 state_dict = checkpoint['state_dict']
    #             elif 'model' in checkpoint:
    #                 state_dict = checkpoint['model']
    #             else:
    #                 state_dict = checkpoint
    #             # all_state_dict.update(state_dict)
    #             # for k in list(state_dict.keys())
    #             # print(state_dict)
    #             print(state_dict.__class__)
    #             # print(state_dict.keys())
    #             print(state_dict['pts_middle_encoder.conv_input.0.weight'].shape, state_dict['pts_middle_encoder.conv_input.0.weight'].__class__)
    #             for k, v in state_dict.items():
    #                 all_state_dict[k] = v
    #             print(all_state_dict['pts_middle_encoder.conv_input.0.weight'].shape, all_state_dict['pts_middle_encoder.conv_input.0.weight'].__class__)
    #             print(all_state_dict==state_dict)
    #     tmp = self.load_state_dict(state_dict, strict=False)
    #     # print(f"LOAD {ckpt}:")
    #     print(tmp)

    def init_weights(self):
        """Initialize model weights."""
        super(BEVMAEPP_LRC_BEVDet4D, self).init_weights()
        
        # self.load_ckpt([self.cam_ckpt, self.radar_ckpt, self.lidar_ckpt])
        # tmp = self.named_parameters()
        # self.load_ckpt([self.lidar_ckpt])
        unload_params = None
        extra_params = None
        if self.cam_ckpt is not None:
            tmp=self.load_ckpt(self.cam_ckpt)
            if unload_params is None:
                unload_params=set(tmp[0])
            else:
                unload_params = unload_params.intersection(set(tmp[0]))

            if extra_params is None:
                extra_params=set(tmp[1])
            else:
                extra_params = extra_params.union(set(tmp[1]))
        if self.radar_ckpt is not None:
            tmp=self.load_ckpt(self.radar_ckpt)
            if unload_params is None:
                unload_params=set(tmp[0])
            else:
                unload_params = unload_params.intersection(set(tmp[0]))

            if extra_params is None:
                extra_params=set(tmp[1])
            else:
                extra_params = extra_params.union(set(tmp[1]))
        if self.lidar_ckpt is not None:
            tmp=self.load_ckpt(self.lidar_ckpt)
            if unload_params is None:
                unload_params=set(tmp[0])
            else:
                unload_params = unload_params.intersection(set(tmp[0]))

            if extra_params is None:
                extra_params=set(tmp[1])
            else:
                extra_params = extra_params.union(set(tmp[1]))
        print('missing keys:', unload_params)
        print('unexpected keys', extra_params)

        def fix_bn(m):
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

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
            
            self.img_view_transformer.apply(fix_bn)
            self.img_bev_encoder_backbone.apply(fix_bn)
            self.img_bev_encoder_neck.apply(fix_bn)
            
            self.img_backbone.apply(fix_bn)
            self.img_neck.apply(fix_bn)

            self.pre_process_net.apply(fix_bn)

        if self.freeze_radar:
            True
        
        if self.freeze_lidar:
            print('freeze lidar backbone and head')
            for name, param in self.named_parameters():
                if 'pts' in name and 'pts_bbox_head' not in name:
                    param.requires_grad = False
                if 'pts_bbox_head.decoder.0' in name:
                    param.requires_grad = False
                if 'pts_bbox_head.shared_conv' in name and 'pts_bbox_head.shared_conv_img' not in name:
                    param.requires_grad = False
                if 'pts_bbox_head.heatmap_head' in name and 'pts_bbox_head.heatmap_head_img' not in name:
                    param.requires_grad = False
                if 'pts_bbox_head.prediction_heads.0' in name:
                    param.requires_grad = False
                if 'pts_bbox_head.class_encoding' in name:
                    param.requires_grad = False

            self.pts_voxel_layer.apply(fix_bn)
            self.pts_voxel_encoder.apply(fix_bn)
            self.pts_middle_encoder.apply(fix_bn)
            self.pts_backbone.apply(fix_bn)
            self.pts_neck.apply(fix_bn)
            
            # self.pts_bbox_head.heatmap_head.apply(fix_bn)
            # self.pts_bbox_head.shared_conv.apply(fix_bn)
            # self.pts_bbox_head.class_encoding.apply(fix_bn)
            # self.pts_bbox_head.decoder[0].apply(fix_bn)
            # self.pts_bbox_head.prediction_heads[0].apply(fix_bn)

    def gen_grid(self, input, sensor2keyegos, bda, bda_adj=None):
        n, c, h, w = input.shape
        _, v, _, _ = sensor2keyegos[0].shape
        if self.grid is None:
            # generate grid
            xs = torch.linspace(
                0, w - 1, w, dtype=input.dtype,
                device=input.device).view(1, w).expand(h, w)
            ys = torch.linspace(
                0, h - 1, h, dtype=input.dtype,
                device=input.device).view(h, 1).expand(h, w)
            grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
            self.grid = grid
        else:
            grid = self.grid
        grid = grid.view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)

        # get transformation from current ego frame to adjacent ego frame
        # transformation from current camera frame to current ego frame
        c02l0 = sensor2keyegos[0][:, 0:1, :, :]

        # transformation from adjacent camera frame to current ego frame
        c12l0 = sensor2keyegos[1][:, 0:1, :, :]

        # add bev data augmentation
        bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        bda_[:, :, :3, :3] = bda.unsqueeze(1)
        bda_[:, :, 3, 3] = 1
        c02l0 = bda_.matmul(c02l0)
        if bda_adj is not None:
            bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
            bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
        c12l0 = bda_.matmul(c12l0)

        # transformation from current ego frame to adjacent ego frame
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:, 0, :, :].view(
            n, 1, 1, 4, 4)
        '''
          c02l0 * inv(c12l0)
        = c02l0 * inv(l12l0 * c12l1)
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

        l02l1 = l02l1[:, :, :,
                      [True, True, False, True], :][:, :, :, :,
                                                    [True, True, False, True]]

        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1,
                                                            2) * 2.0 - 1.0
        return grid

    @force_fp32()
    def shift_feature(self, input, sensor2keyegos, bda, bda_adj=None):
        grid = self.gen_grid(input, sensor2keyegos, bda, bda_adj=bda_adj)
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True)
        return output

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input):
        x, _ = self.image_encoder(img)
        bev_feat, depth = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth

    def extract_img_feat_sequential(self, inputs, feat_prev, with_bevencoder=True):
        imgs, sensor2keyegos_curr, ego2globals_curr, intrins = inputs[:4]
        sensor2keyegos_prev, _, post_rots, post_trans, bda = inputs[4:]
        bev_feat_list = []
        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos_curr[0:1, ...], ego2globals_curr[0:1, ...],
            intrins, post_rots, post_trans, bda[0:1, ...])
        inputs_curr = (imgs, sensor2keyegos_curr[0:1, ...],
                       ego2globals_curr[0:1, ...], intrins, post_rots,
                       post_trans, bda[0:1, ...], mlp_input)
        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
        bev_feat_list.append(bev_feat)

        # align the feat_prev
        _, C, H, W = feat_prev.shape
        feat_prev = self.shift_feature(feat_prev,
                               [sensor2keyegos_curr, sensor2keyegos_prev], bda)
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))

        bev_feat = torch.cat(bev_feat_list, dim=1)
        if with_bevencoder:
            x = self.bev_encoder(bev_feat)
        else:
            x = bev_feat
        return [x], depth

    def prepare_inputs(self, inputs, stereo=False):
        # split the inputs into each frame
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        curr2adjsensor = None
        if stereo:
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = \
                sensor2egos_cv[:, :self.temporal_frame, ...].double()
            ego2globals_curr = \
                ego2globals_cv[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = \
                sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = \
                ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
            curr2adjsensor = \
                torch.inverse(ego2globals_adj @ sensor2egos_adj) \
                @ ego2globals_curr @ sensor2egos_curr
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda, curr2adjsensor

    def extract_img_feat(self,
                         img,
                         img_metas,
                         with_bevencoder=True,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            return self.extract_img_feat_sequential(img, kwargs['feat_prev'], with_bevencoder=with_bevencoder)
        imgs, sensor2keyegos, ego2globals, intrins, \
        post_rots, post_trans, bda, _ = self.prepare_inputs(img)
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
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot,
                               post_tran, bda, mlp_input)
                if key_frame:
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False
        if pred_prev:
            assert self.align_after_view_transfromation
            assert sensor2keyegos[0].shape[0] == 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            ego2globals_curr = \
                ego2globals[0].repeat(self.num_frame - 1, 1, 1, 1)
            sensor2keyegos_curr = \
                sensor2keyegos[0].repeat(self.num_frame - 1, 1, 1, 1)
            ego2globals_prev = torch.cat(ego2globals[1:], dim=0)
            sensor2keyegos_prev = torch.cat(sensor2keyegos[1:], dim=0)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
            return feat_prev, [imgs[0],
                               sensor2keyegos_curr, ego2globals_curr,
                               intrins[0],
                               sensor2keyegos_prev, ego2globals_prev,
                               post_rots[0], post_trans[0],
                               bda_curr]
        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0], sensor2keyegos[adj_id]], bda)
        if not hasattr(self,'img_down2top_encoder_backbone'):
            bev_feat = torch.cat(bev_feat_list, dim=1)
            if with_bevencoder:
                x = self.bev_encoder(bev_feat)
            else:
                x = bev_feat
        else:
            if with_bevencoder:
                x = self.bev_encoder(bev_feat_list)
            else:
                raise ValueError('Define img_down2top_encoder_backbone but set with_bevencoder'
                                 '=False. You should skip bev_encoder when defining a mix'
                                 'backbone model.')
        return [x], depth_list[0]


@DETECTORS.register_module()
class BEVMAEPP_LRC_BEVDepth4D(BEVMAEPP_LRC_BEVDet4D):

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      radar=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_masks_bev=None,
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
        img_feats, pts_feats, depth, before_bev_feat = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, with_bevencoder=self.with_bevencoder, **kwargs ,radar=radar)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        # losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
        #                                     gt_labels_3d, gt_masks_bev,
        #                                     img_metas, gt_bboxes_ignore)

        # loss_dict = dict()

        # outs = self.pts_bbox_head(pts_feats)
        # loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        # losses.update(self.pts_bbox_head.loss(*loss_inputs))

        # mask_tensor = torch.zeros_like(before_bev_feat).detach()
        # mask_tensor = (mask_tensor == before_bev_feat).max(dim=1)[0]
        # mask_tensor = mask_tensor.detach()
        before_bev_feat = before_bev_feat.sum(dim=1).unsqueeze(dim=1)
        mask_tensor = torch.zeros_like(before_bev_feat).detach()
        # print(mask_tensor)
        # print(before_bev_feat)
        mask_tensor = (mask_tensor != before_bev_feat)
        mask_tensor = mask_tensor.detach()

        outs = self.pts_bbox_head(img_feats, pts_feats, self.p_step, mask_tensor)
        # loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses.update(outs)

        self.p_step = 1 - self.p_step

        return losses

@DETECTORS.register_module()
class BEVMAEPP_LRC_BEVDepth4D_d2t(BEVMAEPP_LRC_BEVDepth4D):
    def __init__(self,img_down2top_encoder_backbone, **kwargs):
        super(BEVMAEPP_LRC_BEVDepth4D, self).__init__(**kwargs)
        self.img_down2top_encoder_backbone = builder.build_backbone(img_down2top_encoder_backbone)


@DETECTORS.register_module()
class BEVMAEPP_LRC_BEVStereo4D(BEVMAEPP_LRC_BEVDepth4D):
    def __init__(self, **kwargs):
        super(BEVMAEPP_LRC_BEVStereo4D, self).__init__(**kwargs)
        self.extra_ref_frames = 1
        self.temporal_frame = self.num_frame
        self.num_frame += self.extra_ref_frames

    def extract_stereo_ref_feat(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        if isinstance(self.img_backbone, ResNet):
            if self.img_backbone.deep_stem:
                x = self.img_backbone.stem(x)
            else:
                x = self.img_backbone.conv1(x)
                x = self.img_backbone.norm1(x)
                x = self.img_backbone.relu(x)
            x = self.img_backbone.maxpool(x)
            for i, layer_name in enumerate(self.img_backbone.res_layers):
                res_layer = getattr(self.img_backbone, layer_name)
                x = res_layer(x)
                return x

        elif isinstance(self.img_backbone, SwinTransformer):
            x = self.img_backbone.patch_embed(x)
            hw_shape = (self.img_backbone.patch_embed.DH,
                        self.img_backbone.patch_embed.DW)
            if self.img_backbone.use_abs_pos_embed:
                x = x + self.img_backbone.absolute_pos_embed
            x = self.img_backbone.drop_after_pos(x)

            for i, stage in enumerate(self.img_backbone.stages):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
                out = out.view(-1, *out_hw_shape,
                               self.img_backbone.num_features[i])
                out = out.permute(0, 3, 1, 2).contiguous()
                return out

        elif isinstance(self.img_backbone, VovNetFPN):
            x = self.img_backbone(x)
            return x[0]

        else:
            raise TypeError("stereo do not support backbone type", type(self.img_backbone))

    def prepare_bev_feat(self, img, sensor2keyego, ego2global, intrin,
                         post_rot, post_tran, bda, mlp_input, feat_prev_iv,
                         k2s_sensor, extra_ref_frame):
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat(img)
            return None, None, stereo_feat
        x, stereo_feat = self.image_encoder(img, stereo=True)
        metas = dict(k2s_sensor=k2s_sensor,
                     intrins=intrin,
                     post_rots=post_rot,
                     post_trans=post_tran,
                     frustum=self.img_view_transformer.cv_frustum.to(x),
                     cv_downsample=4,
                     downsample=self.img_view_transformer.downsample,
                     grid_config=self.img_view_transformer.grid_config,
                     cv_feat_list=[feat_prev_iv, stereo_feat])
        bev_feat, depth = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             mlp_input], metas)
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth, stereo_feat

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
        for fid in range(self.num_frame-1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame-self.extra_ref_frames
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
            if len(bev_feat_key.shape) ==4:
                b,c,h,w = bev_feat_key.shape
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
            for adj_id in range(self.num_frame-2):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame-2-adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        if with_bevencoder:
            x = self.bev_encoder(bev_feat)
            return [x], depth_key_frame
        else:
            return [bev_feat], depth_key_frame
