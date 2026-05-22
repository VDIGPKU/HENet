# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet_occ import BEVStereo4DOCC
import torch.nn.functional as F
import torch
import cv2
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np
from .. import builder
import time
from .Unet3D import UNet3D
from mmdet3d.models.builder import build_head, build_neck,build_backbone


# occ3d-nuscenes
nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808, 2457947, 
            497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031, 
            141625221, 2307405309])

@DETECTORS.register_module()
class RenderOcc(BEVStereo4DOCC):
    def __init__(self,
                 out_dim=32,
                 num_classes=18,
                 nerf_head=None,
                 test_threshold=8.5,
                 use_lss_depth_loss=True,
                 use_3d_loss=False,
                 balance_cls_weight=True,
                 final_softplus=False,
                 **kwargs):
        super(RenderOcc, self).__init__(use_predicter=False, **kwargs)
        self.out_dim = out_dim
        self.use_3d_loss = use_3d_loss
        self.test_threshold = test_threshold
        self.use_lss_depth_loss = use_lss_depth_loss
        self.balance_cls_weight = balance_cls_weight
        self.final_softplus = final_softplus

        if self.balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:17] + 0.001)).float()
            self.semantic_loss = nn.CrossEntropyLoss(
                    weight=self.class_weights, reduction="mean"
                )
        else:
            self.semantic_loss = nn.CrossEntropyLoss(reduction="mean")

        self.final_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        self.out_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))

        if self.final_softplus:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, 2),
                nn.Softplus(),
            )
        else:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, 2),
            )

        self.semantic_mlp = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim*2),
            nn.Softplus(),
            nn.Linear(self.out_dim*2, num_classes-1),
        )

        self.nerf_head = builder.build_head(nerf_head)
      

    def loss_3d(self,voxel_semantics,mask_camera,density_prob, semantic):
        voxel_semantics=voxel_semantics.long()
     
        voxel_semantics=voxel_semantics.reshape(-1)
        density_prob=density_prob.reshape(-1, 2)
        semantic = semantic.reshape(-1, self.num_classes-1)
        density_target = (voxel_semantics==17).long()
        semantic_mask = voxel_semantics!=17

        # compute loss
        loss_geo=self.loss_occ(density_prob, density_target)
        loss_sem = self.semantic_loss(semantic[semantic_mask], voxel_semantics[semantic_mask].long())

        loss_ = dict()
        loss_['loss_3d_geo'] = loss_geo
        loss_['loss_3d_sem'] = loss_sem
        return loss_


    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        # extract volumn feature
        img_feats, depth = self.extract_img_feat(img, img_metas, **kwargs)
        voxel_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        
        # predict SDF
        density_prob = self.density_mlp(voxel_feats)
        density = density_prob[...,0]
        semantic = self.semantic_mlp(voxel_feats)

        # SDF --> Occupancy
        no_empty_mask = density > self.test_threshold
        semantic_res = semantic.argmax(-1)

        B, H, W, Z, C = voxel_feats.shape
        occ = torch.ones((B,H,W,Z), dtype=semantic_res.dtype).to(semantic_res.device)
        occ = occ * (self.num_classes-1)
        occ[no_empty_mask] = semantic_res[no_empty_mask]

        occ = occ.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # extract volumn feature
        
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        voxel_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
                bda=img_inputs
        
        # predict SDF
        density_prob = self.density_mlp(voxel_feats)
        density = density_prob[..., 0]
        semantic = self.semantic_mlp(voxel_feats)

        # compute loss
        losses = dict()
        if self.use_3d_loss:      # 3D loss
            voxel_semantics = kwargs['voxel_semantics']
            mask_camera = kwargs['mask_camera']
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
            loss_occ = self.loss_3d(voxel_semantics, mask_camera, density_prob, semantic)
            losses.update(loss_occ)
        if self.nerf_head:          # 2D rendering loss
            loss_rendering = self.nerf_head(density, semantic, rays=kwargs['rays'], bda=bda)
            losses.update(loss_rendering)
        if self.use_lss_depth_loss: # lss-depth loss (BEVStereo's feature)
            loss_depth = self.img_view_transformer.get_depth_loss(kwargs['gt_depth'], depth)
            losses['loss_lss_depth'] = loss_depth
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
       
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
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)
        
        
        
@DETECTORS.register_module()
class HopRenderOcc(RenderOcc):
    def __init__(self,
                 long_term=9,
                 short_term=1,
                 with_hop=True,
                 historay_target_frame=1,
                 historay_decoder_3d=dict(),
                 **kwargs):
        super(HopRenderOcc, self).__init__(**kwargs)
        self.long_term=long_term
        self.short_term=short_term
        self.with_hop=with_hop

        self.history_decoder_3d_cfg=None
        self.history_decoder_2d_cfg=None
        
        if historay_decoder_3d:
            self.history_decoder_3d_cfg=historay_decoder_3d
            self.hop_3d_backbone=build_backbone(self.history_decoder_3d_cfg.hop_backbone)
            self.hop_3d_neck=build_neck(self.history_decoder_3d_cfg.hop_neck)
                
        self.historay_target_frame=historay_target_frame
        
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        
        # extract volumn feature
        img_feats, depth,bev_feat_list = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        voxel_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bn cdhw->bn whdc   
        
        # predict SDF
        density_prob = self.density_mlp(voxel_feats)
        density = density_prob[..., 0]
        semantic = self.semantic_mlp(voxel_feats)
        
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
                bda=img_inputs
        
        # compute loss
        losses = dict()
        if self.use_3d_loss:      # 3D loss
            voxel_semantics = kwargs['voxel_semantics']
            mask_camera = kwargs['mask_camera']
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
            loss_occ = self.loss_3d(voxel_semantics, mask_camera, density_prob, semantic)
            losses.update(loss_occ)
            
        if self.nerf_head:          # 2D rendering loss
            cur_frame_rays=kwargs['rays']['cur_rays']
            cur_frame_rays=torch.cat(cur_frame_rays,dim=1)
            loss_rendering = self.nerf_head(density, semantic, rays=cur_frame_rays, bda=bda)
            losses.update(loss_rendering)
            
        if self.use_lss_depth_loss: # lss-depth loss (BEVStereo's feature)
            loss_depth = self.img_view_transformer.get_depth_loss(kwargs['gt_depth'], depth)
            losses['loss_lss_depth'] = loss_depth
        
        if self.with_hop:
            adj_frame_rays=kwargs['rays']['adj_rays']
            adj_frames=len(adj_frame_rays)
            
            if self.history_decoder_3d_cfg:
                #采用3d时序建模
                voxel_feat_list=[]
                target_voxel_feat=self.final_conv(bev_feat_list[self.historay_target_frame+2]).permute(0, 4, 3, 2, 1)
                target_rays=adj_frame_rays[self.historay_target_frame]  #bev_feat_list[0]是None，第一帧是stereo预测depth多出的一帧，所以bev_feat_list[i+2]对应gt_rays[i]
                for i in range(adj_frames):
                    if i == self.historay_target_frame:
                        voxel_feat=torch.zeros_like(voxel_feats)
                    else:
                        voxel_feat=self.final_conv(bev_feat_list[i+2]).permute(0, 4, 3, 2, 1) # bn cdhw->bn whdc 
                    voxel_feat_list.append(voxel_feat)
                history_voxel_feats=torch.cat(voxel_feat_list,dim=4)
                
                if self.hop_3d_backbone:
                    pred_target_feat=self.hop_3d_backbone(history_voxel_feats.permute(0,4,1,2,3))
                if self.hop_3d_neck:
                    pred_target_feat=self.hop_3d_neck(pred_target_feat)
                
                # predict SDF
                density_prob = self.density_mlp(pred_target_feat.permute(0, 4, 3, 2, 1))
                density = density_prob[..., 0]
                semantic = self.semantic_mlp(pred_target_feat.permute(0, 4, 3, 2, 1))   
                
                target_rays=torch.cat(target_rays,dim=1)
                loss_rendering = self.nerf_head(density, semantic, rays=target_rays, bda=bda)
                
                for key in loss_rendering.keys():
                    new_key='hop_'+key
                    loss_rendering[new_key]=loss_rendering.pop(key)
                
                losses.update(loss_rendering)
                
        return losses
        
        
        
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
                # if not extra_ref_frame:
                #     bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            
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
        bev_feat = torch.cat(bev_feat_list[1:], dim=1)
        if with_bevencoder:
            x = self.bev_encoder(bev_feat)
            return [x], depth_key_frame,bev_feat_list
        else:
            return [bev_feat], depth_key_frame,bev_feat_list
        
        
    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        # extract volumn feature
        img_feats, depth,bev_feat_list = self.extract_img_feat(img, img_metas, **kwargs)
        voxel_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        
        # predict SDF
        density_prob = self.density_mlp(voxel_feats)
        density = density_prob[...,0]
        semantic = self.semantic_mlp(voxel_feats)

        # SDF --> Occupancy
        no_empty_mask = density > self.test_threshold
        semantic_res = semantic.argmax(-1)

        B, H, W, Z, C = voxel_feats.shape
        occ = torch.ones((B,H,W,Z), dtype=semantic_res.dtype).to(semantic_res.device)
        occ = occ * (self.num_classes-1)
        occ[no_empty_mask] = semantic_res[no_empty_mask]

        occ = occ.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ]