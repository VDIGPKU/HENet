import queue
import torch
import numpy as np
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner import get_dist_info
from mmcv.runner.fp16_utils import cast_tensor_type
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from .utils import GridMask, pad_multiple, GpuPhotoMetricDistortion

from torch import nn as nn
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import ConvModule, xavier_init, normal_init
from ..model_utils.ops.modules.ms_deform_attn import MSDeformAttn, LearnedPositionalEncoding3D, SinePositionalEncoding3D
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
import math
from .. import builder
from mmcv.ops import Voxelization
from einops import rearrange, repeat

from tools.misc.vis_tools import print_gt_and_bev, print_pcgt_on_bev_radar
from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes

from mmdet3d.ops.voxelization.voxelize import Voxelization as MyVoxelization


class RadarConvFuser(BaseModule):
    def __init__(self, in_channels: int, out_channels: int, deconv_blocks: int) -> None:
        super(RadarConvFuser, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        deconv = []
        deconv_in = [sum(in_channels) + out_channels]
        deconv_out = [out_channels]
        for i in range(deconv_blocks - 1):
            deconv_in.append(out_channels)
            deconv_out.append(out_channels)
        for i in range(deconv_blocks):
            deconv.append(nn.Sequential(
                nn.Conv2d(deconv_in[i], deconv_out[i], 3, padding=1, bias=False),
                nn.BatchNorm2d(deconv_out[i]),
                nn.ReLU(True))
            )
        self.deconv = nn.ModuleList(deconv)

    def init_weights(self):
        super().init_weights()
        normal_init(self.fuse_conv, mean=0, std=0.001)
        for i in enumerate(self.deconv):
            normal_init(i, mean=0, std=0.001)

    def forward(self, input1, input2) -> torch.Tensor:
        res = torch.cat((input1, input2), dim=1)
        res2 = res.clone()
        out = self.fuse_conv(res)
        out = torch.cat([out, res2], dim=1)
        for layer in self.deconv:
            out = layer(out)
        return out
    

@DETECTORS.register_module()
class SparseBEV_rc(MVXTwoStageDetector):
    def __init__(self,
                 data_aug=None,
                 stop_prev_grad=0,
                 radar_voxel_layer=None,
                 pts_pillar_layer=None,
                 radar_voxel_encoder=None,
                 pts_voxel_encoder=None,
                 radar_middle_encoder=None,
                 radar_bev_backbone=None,
                 radar_bev_neck=None,
                 radar_reduc_conv=None, #new
                 imgpts_neck=None,
                 DeformAttn=None,
                 imc=256, rac=384, #im ra 特征维度
                 freeze_img=False,
                 freeze_radar=False,
                 dynamic_reference_point=False,
                 num_reference_points=16,
                 bev_size=128,
                 **kwargs):

        super(SparseBEV_rc, self).__init__(**kwargs)
        self.data_aug = data_aug
        self.stop_prev_grad = stop_prev_grad
        self.color_aug = GpuPhotoMetricDistortion()
        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = True

        self.memory = {}
        self.queue = queue.Queue()


        #new
        if radar_voxel_layer!=None:
            self.radar_voxel_layer = Voxelization(**radar_voxel_layer)
        if pts_pillar_layer!=None:
            self.pts_pillar_layer = Voxelization(**pts_pillar_layer)
        if pts_voxel_encoder is not None:
            self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        if radar_voxel_encoder!=None:
            self.radar_voxel_encoder = builder.build_voxel_encoder(radar_voxel_encoder)
        if radar_middle_encoder!=None:
            self.radar_middle_encoder = builder.build_middle_encoder(radar_middle_encoder)
        if radar_bev_backbone is not None:
            self.radar_bev_backbone = builder.build_backbone(radar_bev_backbone)
        if radar_bev_neck is not None:
            self.radar_bev_neck = builder.build_neck(radar_bev_neck)

        self.bev_size = bev_size
        if DeformAttn is not None:
            self.DeformAttn1 = MSDeformAttn( d_model=256, n_levels=1, n_heads=8, n_points=8) #d_model=256, n_levels=1, n_heads=8, n_points=4
            self.DeformAttn2 = MSDeformAttn( d_model=256, n_levels=1, n_heads=8, n_points=8) #d_model=256, n_levels=1, n_heads=8, n_points=4
            patch_row = bev_size #// 4 pretrain_img_size[0] patch_size
            patch_col = bev_size #// 4
            num_patches = patch_row * patch_col
            self.LearnedPositionalEncoding1=LearnedPositionalEncoding3D(num_feats=imc//2, row_num_embed=bev_size, col_num_embed=bev_size)
            self.LearnedPositionalEncoding2=LearnedPositionalEncoding3D(num_feats=imc//2, row_num_embed=bev_size, col_num_embed=bev_size) 

        if radar_reduc_conv is not None:
            self.radar_reduc_conv = ConvModule(
                    rac,
                    imc,  #rac change imc
                    kernel_size=3,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                    act_cfg=dict(type='ReLU'),
                    inplace=False) 
            self.RadarConvFuser_fuse = RadarConvFuser(in_channels = (imc,imc),out_channels = imc,deconv_blocks = 3)
        self.dynamic_reference_point = dynamic_reference_point
        if self.dynamic_reference_point:
            self.reference_points = nn.Linear(imc, 2*num_reference_points)
        else:
            self.reference_points = None

        
        self.freeze_img=freeze_img
        self.freeze_radar=freeze_radar


    def init_weights(self):
        """Initialize model weights."""
        super(SparseBEV_rc, self).init_weights()
        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
        if self.dynamic_reference_point:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)

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
            # print(res.shape)
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
    
    @auto_fp16(apply_to=('radar'), out_fp32=True) 
    def extract_radar_feat(self, radar, img_metas, gt_bboxes_3d):
        """Extract features of points."""
        #data1 = open("radar_type.txt",'w',encoding="utf-8")
        #print(type(radar),file=data1)    
        # start_time = time.perf_counter()
        # elapsed = 0
        voxels, num_points, coors = self.radar_voxelize(radar)
        # print('voxel', time.perf_counter()-start_time)
        # start_time = time.perf_counter()
        voxel_features = self.radar_voxel_encoder(voxels, num_points, coors)
        # print('encoder', time.perf_counter()-start_time)
        batch_size = coors[-1, 0] + 1
        # start_time = time.perf_counter()
        x = self.radar_middle_encoder(voxel_features, coors, batch_size)
        # print('middle', time.perf_counter()-start_time)
        # start_time = time.perf_counter()

        if hasattr(self, 'radar_bev_backbone') and self.radar_bev_backbone is not None:
            # print(x.size()) 
            x = self.radar_bev_backbone(x) # 8, 64, h/2, w/2
        
        if hasattr(self, 'radar_bev_neck') and self.radar_bev_neck is not None:
            # print(len(x), x[0].size())
            x = self.radar_bev_neck(x) # 8, 64, h/4, w/4
            # print(len(x), x[0].size())
            x = x[0]
        # print('bevencoder', time.perf_counter()-start_time)
        # if hasattr(self, 'se') and self.se is not None:
        #     x = self.se(x)

        return [x]

    def extract_radar_feat_v2(self, voxels, num_points, coors, img_metas, gt_bboxes_3d):
        """Extract features of points."""
        #data1 = open("radar_type.txt",'w',encoding="utf-8")
        #print(type(radar),file=data1)    
        # start_time = time.perf_counter()
        # elapsed = 0
        # print('voxel', time.perf_counter()-start_time)
        # start_time = time.perf_counter()
        coors = F.pad(coors, (1, 0), mode='constant', value=0)

        voxel_features = self.radar_voxel_encoder.forward_trt(voxels, num_points, coors)
        # print('encoder', time.perf_counter()-start_time)
        batch_size = coors[-1, 0] + 1
        # start_time = time.perf_counter()
        x = self.radar_middle_encoder(voxel_features, coors, batch_size)
        # print('middle', time.perf_counter()-start_time)
        # start_time = time.perf_counter()

        if hasattr(self, 'radar_bev_backbone') and self.radar_bev_backbone is not None:
            # print(x.size()) 
            x = self.radar_bev_backbone(x) # 8, 64, h/2, w/2
        
        if hasattr(self, 'radar_bev_neck') and self.radar_bev_neck is not None:
            # print(len(x), x[0].size())
            x = self.radar_bev_neck(x) # 8, 64, h/4, w/4
            # print(len(x), x[0].size())
            x = x[0]
        # print('bevencoder', time.perf_counter()-start_time)
        # if hasattr(self, 'se') and self.se is not None:
        #     x = self.se(x)

        return [x]

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos
    
    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='2d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @auto_fp16(apply_to=('img'), out_fp32=True) 
    def extract_img_feat(self, img):
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_feat(self, img, img_metas, radar, gt_bboxes_3d=None):
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        assert img.dim() == 5

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img = img.float()


        #print_pcgt_on_bev_radar(radar, gt_bboxes_3d)
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

            all_img_feats = [self.extract_img_feat(img_grad.reshape(-1, C, H, W))]

            with torch.no_grad():
                self.eval()
                for k in range(img_nograd.shape[1]):
                    all_img_feats.append(self.extract_img_feat(img_nograd[:, k].reshape(-1, C, H, W)))
                self.train()

            img_feats = []
            for lvl in range(len(all_img_feats[0])):
                C, H, W = all_img_feats[0][lvl].shape[1:]
                img_feat = torch.cat([feat[lvl].reshape(B, -1, 6, C, H, W) for feat in all_img_feats], dim=1)
                img_feat = img_feat.reshape(-1, C, H, W)
                img_feats.append(img_feat)
        else:
            img_feats = self.extract_img_feat(img)
        

        
        #print(img_feats_reshaped[0].size())
        
        # start_time = time.perf_counter()
        #img_feats=img_feats_reshaped        
        # elapsed = time.perf_counter() - start_time
        # print('image', elapsed)
        # start_time = time.perf_counter()
        pts_feats = None
        radar_feats = self.extract_radar_feat(radar, img_metas, gt_bboxes_3d) #new
        # elapsed = time.perf_counter() - start_time
        # print('radar', elapsed)

        #print_gt_and_bev(radar_feats[0], gt=gt_bboxes_3d, img_metas=img_metas, name='vis')
        '''
        fusion_feats = []
        bev_height = img_feats[0].shape[2]
        bev_width = img_feats[0].shape[3]

        #radar_feats=radar_feats[0]
        #radar_feats=torch.sum(radar_feats, dim=1,keepdim=True)
        #print_gt_and_bev(radar_feats, gt=gt_bboxes_3d, img_metas=img_metas, name='vis')
        # start_time = time.perf_counter()
        for i in range(0,len(img_feats)):
        #    if hasattr(self, 'radar_reduc_conv') and self.radar_reduc_conv is not None and self.imgpts_neck == None:
        #        fusion_f = self.radar_reduc_conv(torch.cat((img_feats[i], radar_feats[i]), dim=1))

            if hasattr(self, 'imgpts_neck') and self.imgpts_neck is not None:
                #print(img_metas)
                new_img_feat, new_pts_feat = self.imgpts_neck(img_feats=img_feats[i], pts_feats=radar_feats[i],img_metas=img_metas, pts_metas=None)
                fusion_f = self.radar_reduc_conv(torch.cat((new_img_feat, new_pts_feat), dim=1))

            if hasattr(self, 'DeformAttn1') and self.DeformAttn1 is not None:
                print(radar_feats[0].size())
                radar_feats[i] = self.radar_reduc_conv(radar_feats[i])
                radar_feats = rearrange(radar_feats[i], 'b c h w -> b (h w) c')
                img_feats = rearrange(img_feats[i], 'b c h w -> b (h w) c')
                
                device = torch.device("cuda")  # Get the CUDA device
                mask = torch.zeros(1, 1, self.bev_size, self.bev_size).to(device)
                pos1 = self.LearnedPositionalEncoding1(mask)
                pos2 = self.LearnedPositionalEncoding2(mask)
                # print(pos1.shape)
                # print(radar_feats.shape)

                
                if self.dynamic_reference_point:
                    b, hw, c = img_feats.shape
                    reference_point1 = self.reference_points(radar_feats).sigmoid().reshape(b, hw, -1, 2)
                    reference_point2 = self.reference_points(img_feats).sigmoid().reshape(b, hw, -1, 2)
                else:
                    reference_point1=self.get_reference_points(self.bev_size,self.bev_size)
                    reference_point2=self.get_reference_points(self.bev_size,self.bev_size)
                    # print(reference_point2.shape)
                

                # if gt_bboxes_3d is not None:
                #     mask = generate_gaussian_mask(gt_bboxes_3d, [bev_height, bev_width])
                #     mask = torch.from_numpy(mask).to(device)
                #     mask = rearrange(mask, 'b h w -> b (h w)')
                    
                # else:
                #     mask = None
                # assert mask is None
                mask = None

                fusion_f1 = self.DeformAttn1(query=self.with_pos_embed(radar_feats, pos1), 
                                             reference_points = reference_point1, 
                                             input_flatten = self.with_pos_embed(img_feats, pos2), 
                                             input_spatial_shapes=torch.tensor([(self.bev_size, self.bev_size)]).to(device), 
                                             input_level_start_index=torch.tensor([0, self.bev_size*self.bev_size]).to(device), 
                                             input_padding_mask=mask)
                fusion_f2 = self.DeformAttn2(query=self.with_pos_embed(img_feats, pos2), 
                                             reference_points = reference_point2, 
                                             input_flatten = self.with_pos_embed(radar_feats, pos1), 
                                             input_spatial_shapes=torch.tensor([(self.bev_size, self.bev_size)]).to(device), 
                                             input_level_start_index=torch.tensor([0, self.bev_size*self.bev_size]).to(device) , 
                                             input_padding_mask=mask)
                fusion_f1 = rearrange(fusion_f1, 'b (h w) c -> b c h w', h=self.bev_size, w=self.bev_size) 
                fusion_f2 = rearrange(fusion_f2, 'b (h w) c -> b c h w', h=self.bev_size, w=self.bev_size)
                fusion_f = self.RadarConvFuser_fuse(fusion_f1,fusion_f2)
                #fusion_f = self.fuse_reduc_conv(torch.cat((fusion_f1, fusion_f2), dim=1))
                """
                :param query                       (N, Length_{query}, C)
                :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                                or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
                :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
                :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
                :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
                :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

                :return output                     (N, Length_{query}, C)
                """
                
            if hasattr(self, 'CBAM') and self.CBAM is not None:
                fusion_f = self.radar_reduc_conv(torch.cat((img_feats[i], radar_feats[i]), dim=1))
                print(1)
                #fusion_f = self.CBAM(fusion_f)


            fusion_feats.append(fusion_f) #cat  
        
        # elapsed = time.perf_counter() - start_time
        # print('fusion', elapsed)
        #img_feats = [self.reduc_conv(torch.cat((img_feats, radar_feats), dim=1))] #cat  self.reduc_conv
        img_feats=fusion_feats
        '''
        #print(len(img_feats)) #4
        
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            #print(img_feat.size())
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        #return img_feats, pts_feats
    
        return img_feats_reshaped, radar_feats

    def extract_feat_v2(self, img, img_metas, voxels, num_points, coors, gt_bboxes_3d=None):
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

            all_img_feats = [self.extract_img_feat(img_grad.reshape(-1, C, H, W))]

            with torch.no_grad():
                self.eval()
                for k in range(img_nograd.shape[1]):
                    all_img_feats.append(self.extract_img_feat(img_nograd[:, k].reshape(-1, C, H, W)))
                self.train()

            img_feats = []
            for lvl in range(len(all_img_feats[0])):
                C, H, W = all_img_feats[0][lvl].shape[1:]
                img_feat = torch.cat([feat[lvl].reshape(B, -1, 6, C, H, W) for feat in all_img_feats], dim=1)
                img_feat = img_feat.reshape(-1, C, H, W)
                img_feats.append(img_feat)
        else:
            img_feats = self.extract_img_feat(img)
        
        pts_feats = None
        radar_feats = self.extract_radar_feat_v2(voxels, num_points, coors, img_metas, gt_bboxes_3d) #new
        
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            #print(img_feat.size())
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        #return img_feats, pts_feats
    
        return img_feats_reshaped, radar_feats

    def forward_pts_train(self,
                          pts_feats,
                          radar_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats,radar_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        return losses

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      radar=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None):
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
        img_feats, radar_feats = self.extract_feat(img, img_metas, radar, gt_bboxes_3d)

        for i in range(len(img_metas)):
            img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i]
            img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]

        losses = self.forward_pts_train(img_feats,radar_feats , gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)

        return losses

    def forward_test(self, img_metas, img=None, radar=None, **kwargs):
        if False:
            import pickle

            data_dict = dict()
            radar = radar[0][0]
            data_dict['radar'] = radar
            radar_shape = radar.shape
            input_shapes = dict(
                radar=dict(
                    min_shape=radar_shape,
                    opt_shape=radar_shape,
                    max_shape=radar_shape
                ),
            )
            data_dict['input_shapes'] = input_shapes
            with open('debug/radar.pkl', 'wb') as f:
                pickle.dump(data_dict, f)
                f.close()
            
            exit(0)

        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], radar[0], **kwargs)

    def simple_test_pts(self, x, radar_feats, img_metas, rescale=False):
        outs = self.pts_bbox_head(x, radar_feats, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas[0], rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results
    
    def simple_test(self, img_metas, img=None, radar=None, rescale=False):
        world_size = get_dist_info()[1]
        if world_size == 1:  # online
            return self.simple_test_online(img_metas, img, radar, rescale)
        else:  # offline
            return self.simple_test_offline(img_metas, img, radar, rescale)

    def simple_test_offline(self, img_metas, img=None, radar=None, rescale=False):
        img_feats, radar_feats= self.extract_feat(img=img, img_metas=img_metas, radar=radar)

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, radar_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list

    def simple_test_online(self, img_metas, img=None, radar=None, rescale=False):
        self.fp16_enabled = False
        assert len(img_metas) == 1  # batch_size = 1

        B, N, C, H, W = img.shape
        img = img.reshape(B, N//6, 6, C, H, W)

        img_filenames = img_metas[0]['filename']
        num_frames = len(img_filenames) // 6
        # assert num_frames == img.shape[1]

        img_shape = (H, W, C)
        img_metas[0]['img_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['ori_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['pad_shape'] = [img_shape for _ in range(len(img_filenames))]

        img_feats_list, img_metas_list = [], []
        # extract feature frame by frame
        for i in range(num_frames):
            img_indices = list(np.arange(i * 6, (i + 1) * 6))

            img_metas_curr = [{}]
            for k in img_metas[0].keys():
                if isinstance(img_metas[0][k], list):
                    img_metas_curr[0][k] = [img_metas[0][k][i] for i in img_indices]

            if img_filenames[img_indices[0]] in self.memory:
                # found in memory
                img_feats_curr = self.memory[img_filenames[img_indices[0]]]
            else:
                # extract feature and put into memory
                img_feats_curr , radar_feats = self.extract_feat(img[:, i], img_metas_curr, radar)
                self.memory[img_filenames[img_indices[0]]] = img_feats_curr
                self.queue.put(img_filenames[img_indices[0]])
                while self.queue.qsize() >= 16:  # avoid OOM
                    pop_key = self.queue.get()
                    self.memory.pop(pop_key)

            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)

        # reorganize
        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat([img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0)
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)

        img_metas_reorganized = img_metas_list[0]
        for i in range(1, len(img_metas_list)):
            for k, v in img_metas_list[i][0].items():
                if isinstance(v, list):
                    img_metas_reorganized[0][k].extend(v)

        img_feats = img_feats_reorganized
        img_metas = img_metas_reorganized
        img_feats = cast_tensor_type(img_feats, torch.half, torch.float32)

        # run detector
        bbox_list = [dict() for _ in range(1)]
        bbox_pts = self.simple_test_pts(img_feats, radar_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list


@DETECTORS.register_module()
class SparseBEV_rcTRT(SparseBEV_rc):
    def __init__(self,
                 **kwargs):

        super(SparseBEV_rcTRT, self).__init__(**kwargs)
        if kwargs['radar_voxel_layer'] != None:
            self.radar_voxel_layer = MyVoxelization(**kwargs['radar_voxel_layer'])
        if kwargs['pts_pillar_layer'] != None:
            self.pts_pillar_layer = MyVoxelization(**kwargs['pts_pillar_layer'])

    # def forward(self, img=None, radar=None, lidar2img=None, img_timestamp=None, len_img_filenames=None, feat_prev_1=None, feat_prev_2=None, feat_prev_3=None, feat_prev_4=None):
    #     len_img_filenames = int(len_img_filenames)

    #     # img = torch.stack(img)
    #     return self.simple_test(img, radar, lidar2img, img_timestamp, len_img_filenames, feat_prev_1, feat_prev_2, feat_prev_3, feat_prev_4)
        
    def forward(self, img=None, voxels=None, num_points=None, coors=None, lidar2img=None, img_timestamp=None, len_img_filenames=None, feat_prev_1=None, feat_prev_2=None, feat_prev_3=None, feat_prev_4=None):
        len_img_filenames = int(len_img_filenames)

        # img = torch.stack(img)
        return self.simple_test_v2(img, voxels, num_points, coors, lidar2img, img_timestamp, len_img_filenames, feat_prev_1, feat_prev_2, feat_prev_3, feat_prev_4)
    
    def simple_test(self, img=None, radar=None, lidar2img=None, img_timestamp=None, len_img_filenames=None, feat_prev_1=None, feat_prev_2=None, feat_prev_3=None, feat_prev_4=None, rescale=False):
        self.fp16_enabled = False
        # img_metas[0] is a key, img.shape = [1, 6, 3, 256, 704], radar is a list with one tensor([1165, 7])

        B, N, C, H, W = img.shape
        B = int(B)
        N = int(N)
        C = int(C)
        H = int(H)
        W = int(W)
        assert B == 1
        img_metas = [{}]
        img = img.reshape(B, N//6, 6, C, H, W)

        num_frames = len_img_filenames // 6
        # assert num_frames == img.shape[1]

        img_shape = (H, W, C)
        img_metas[0]['img_shape'] = [img_shape for _ in range(len_img_filenames)]
        img_metas[0]['ori_shape'] = [img_shape for _ in range(len_img_filenames)]
        img_metas[0]['pad_shape'] = [img_shape for _ in range(len_img_filenames)]

        img_feats_list, img_metas_list = [], []

        # extract feature frame by frame
        for i in range(num_frames):
            img_indices = list(np.arange(i * 6, (i + 1) * 6))

            img_metas_curr = [{}]
            for k in img_metas[0].keys():
                if k == 'img_shape' or k == 'ori_shape' or k == 'pad_shape':
                    img_metas_curr[0][k] = [img_metas[0][k][i] for i in img_indices]

            if i == 0:
                # extract feature and put into memory
                # img_feats_curr is a list with 4 tensors
                img_feats_curr , radar_feats = self.extract_feat(img[:, i], img_metas_curr, radar)
                img_feats_curr_ret = img_feats_curr
            else:                
                # found in memory
                img_feats_curr = [feat_prev_1[0, i-1], feat_prev_2[0, i-1], feat_prev_3[0, i-1], feat_prev_4[0, i-1]]

            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)
    
        # reorganize
        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            # import ipdb;ipdb.set_trace()
            feat_l = torch.cat([img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0)
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)

        img_metas_reorganized = img_metas_list[0]
        for i in range(1, len(img_metas_list)):
            for k, v in img_metas_list[i][0].items():
                if isinstance(v, list):
                    img_metas_reorganized[0][k].extend(v)

        img_metas_reorganized[0]['lidar2img'] = lidar2img # shape is BxNxCxC
        img_metas_reorganized[0]['img_timestamp'] = img_timestamp[0] # shape is BxNxCxC

        img_feats = img_feats_reorganized
        img_metas = img_metas_reorganized
        # img_feats = cast_tensor_type(img_feats, torch.half, torch.float32)

        # run detector
        cls_scores, bbox_preds = self.pts_bbox_head.forward_trt(img_feats, radar_feats, img_metas)        
        return cls_scores, bbox_preds, img_feats_curr_ret[0], img_feats_curr_ret[1], img_feats_curr_ret[2], img_feats_curr_ret[3]

    def simple_test_v2(self, img=None, voxels=None, num_points=None, coors=None, lidar2img=None, img_timestamp=None, len_img_filenames=None, feat_prev_1=None, feat_prev_2=None, feat_prev_3=None, feat_prev_4=None, rescale=False):
        self.fp16_enabled = False
        # img_metas[0] is a key, img.shape = [1, 6, 3, 256, 704], radar is a list with one tensor([1165, 7])

        B, N, C, H, W = img.shape
        B = int(B)
        N = int(N)
        C = int(C)
        H = int(H)
        W = int(W)
        assert B == 1
        img_metas = [{}]
        img = img.reshape(B, N//6, 6, C, H, W)

        num_frames = len_img_filenames // 6
        # assert num_frames == img.shape[1]

        img_shape = (H, W, C)
        img_metas[0]['img_shape'] = [img_shape for _ in range(len_img_filenames)]
        img_metas[0]['ori_shape'] = [img_shape for _ in range(len_img_filenames)]
        img_metas[0]['pad_shape'] = [img_shape for _ in range(len_img_filenames)]

        img_feats_list, img_metas_list = [], []

        # extract feature frame by frame
        for i in range(num_frames):
            img_indices = list(np.arange(i * 6, (i + 1) * 6))

            img_metas_curr = [{}]
            for k in img_metas[0].keys():
                if k == 'img_shape' or k == 'ori_shape' or k == 'pad_shape':
                    img_metas_curr[0][k] = [img_metas[0][k][i] for i in img_indices]

            if i == 0:
                # extract feature and put into memory
                # img_feats_curr is a list with 4 tensors
                img_feats_curr , radar_feats = self.extract_feat_v2(img[:, i], img_metas_curr, voxels[0], num_points[0], coors[0])
                img_feats_curr_ret = img_feats_curr
            else:                
                # found in memory
                img_feats_curr = [feat_prev_1[0, i-1], feat_prev_2[0, i-1], feat_prev_3[0, i-1], feat_prev_4[0, i-1]]

            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)
    
        # reorganize
        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            # import ipdb;ipdb.set_trace()
            feat_l = torch.cat([img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0)
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)

        img_metas_reorganized = img_metas_list[0]
        for i in range(1, len(img_metas_list)):
            for k, v in img_metas_list[i][0].items():
                if isinstance(v, list):
                    img_metas_reorganized[0][k].extend(v)

        img_metas_reorganized[0]['lidar2img'] = lidar2img # shape is BxNxCxC
        img_metas_reorganized[0]['img_timestamp'] = img_timestamp[0] # shape is BxNxCxC

        img_feats = img_feats_reorganized
        img_metas = img_metas_reorganized

        # run detector
        cls_scores, bbox_preds = self.pts_bbox_head.forward_trt(img_feats, radar_feats, img_metas)        
        return cls_scores, bbox_preds, img_feats_curr_ret[0], img_feats_curr_ret[1], img_feats_curr_ret[2], img_feats_curr_ret[3]

@DETECTORS.register_module()
class SparseBEV_rcTRTDEBUG(SparseBEV_rcTRT):
    def __init__(self,
                 **kwargs):

        super(SparseBEV_rcTRTDEBUG, self).__init__(**kwargs)
    
    def simple_test_v2(self, img=None, voxels=None, num_points=None, coors=None, lidar2img=None, img_timestamp=None, len_img_filenames=None, feat_prev_1=None, feat_prev_2=None, feat_prev_3=None, feat_prev_4=None, rescale=False):
        self.fp16_enabled = False
        # img_metas[0] is a key, img.shape = [1, 6, 3, 256, 704], radar is a list with one tensor([1165, 7])

        B, N, C, H, W = img.shape
        B = int(B)
        N = int(N)
        C = int(C)
        H = int(H)
        W = int(W)
        assert B == 1
        img_metas = [{}]
        img = img.reshape(B, N//6, 6, C, H, W)

        num_frames = len_img_filenames // 6
        # assert num_frames == img.shape[1]

        img_shape = (H, W, C)
        img_metas[0]['img_shape'] = [img_shape for _ in range(len_img_filenames)]
        img_metas[0]['ori_shape'] = [img_shape for _ in range(len_img_filenames)]
        img_metas[0]['pad_shape'] = [img_shape for _ in range(len_img_filenames)]

        img_feats_list, img_metas_list = [], []

        # extract feature frame by frame
        for i in range(num_frames):
            img_indices = list(np.arange(i * 6, (i + 1) * 6))

            img_metas_curr = [{}]
            for k in img_metas[0].keys():
                if k == 'img_shape' or k == 'ori_shape' or k == 'pad_shape':
                    img_metas_curr[0][k] = [img_metas[0][k][i] for i in img_indices]

            if i == 0:
                # extract feature and put into memory
                # img_feats_curr is a list with 4 tensors
                img_feats_curr , radar_feats = self.extract_feat_v2(img[:, i], img_metas_curr, voxels[0], num_points[0], coors[0])
                img_feats_curr_ret = img_feats_curr
            else:                
                # found in memory
                img_feats_curr = [feat_prev_1[0, i-1], feat_prev_2[0, i-1], feat_prev_3[0, i-1], feat_prev_4[0, i-1]]

            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)

        # run detector
        import ipdb;ipdb.set_trace()
        
        return img_feats_curr_ret[0], img_feats_curr_ret[1], img_feats_curr_ret[2], img_feats_curr_ret[3]
