# The backbone of our MonoNeRD model.
# including 2D feature extraction, stereo volume construction, stereo network, stereo space -> 3D space conversion
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

import math
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import matplotlib.pyplot as plt
import time
from mmdet.models.builder import build_backbone, build_neck
from . import mono_submodule
from .mono_submodule import convbn, convbn_3d, feature_extraction_neck
from mmdet3d.utils.render_utils import sample_along_rays, construct_ray_warps, compute_alpha_weights, grid_generation, \
    LaplaceDensity, voxel_eikonal_loss
from utils.loss_utils import ssim,l1_loss
import cv2


def project_pseudo_lidar_to_rectcam(pts_3d):
    xs, ys, zs = pts_3d[..., 0], pts_3d[..., 1], pts_3d[..., 2]
    return torch.stack([-ys, -zs, xs], dim=-1)


def project_rectcam_to_pseudo_lidar(pts_3d):
    xs, ys, zs = pts_3d[..., 0], pts_3d[..., 1], pts_3d[..., 2]
    return torch.stack([zs, -xs, -ys], dim=-1)


def project_rect_to_image(pts_3d_rect, P):
    n = pts_3d_rect.shape[0]
    ones = torch.ones_like(pts_3d_rect[..., 2:3], device=pts_3d_rect.device)
    pts_3d_rect = torch.cat([pts_3d_rect, ones], dim=-1)
    pts_2d = torch.matmul(pts_3d_rect, torch.transpose(P, 0, 1))  # nx3
    pts_2d[..., 0] /= pts_2d[..., 2]
    pts_2d[..., 1] /= pts_2d[..., 2]
    return pts_2d


def unproject_image_to_rect(pts_image, P):
    pts_3d = torch.cat([pts_image[..., :2], torch.ones_like(pts_image[..., 2:3])], -1)
    pts_3d = pts_3d * pts_image[..., 2:3]
    pts_3d = torch.cat([pts_3d, torch.ones_like(pts_3d[..., 2:3])], -1)
    P4x4 = torch.eye(4, dtype=P.dtype, device=P.device)
    P4x4[:3, :] = P
    invP = torch.inverse(P4x4)
    pts_3d = torch.matmul(pts_3d, torch.transpose(invP, 0, 1))
    return pts_3d[..., :3]


# @BACKBONES.register_module()
class MonoNeRDBackbone(nn.Module):
    def __init__(self, model_cfg, class_names, grid_size, voxel_size, point_cloud_range, boxes_gt_in_cam2_view=False,
                 **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        # general config
        self.class_names = class_names
        self.GN = model_cfg.GN
        self.boxes_gt_in_cam2_view = boxes_gt_in_cam2_view

        # volume construction config
        self.cat_img_feature = model_cfg.cat_img_feature
        self.voxel_attentionbydensity = model_cfg.voxel_attentionbydensity
        self.rpn3d_dim = model_cfg.rpn3d_dim

        # frustum config
        self.frustum_cfg = model_cfg.frustum_cfg
        self.use_qkv = self.frustum_cfg.use_qkv
        self.pe = self.frustum_cfg.pe
        self.num_3dconvs = model_cfg.num_3dconvs
        self.volume_dim = model_cfg.volume_dim
        # self.image_dim = model_cfg.image_dim

        # nerf config
        self.nerf_cfg = model_cfg.nerf_cfg
        self.use_nerf = self.nerf_cfg.use_nerf
        self.nerf_num_samples = self.nerf_cfg.nerf_num_samples
        self.nerf_near = self.nerf_cfg.nerf_near
        self.nerf_far = self.nerf_cfg.nerf_far
        self.uniform_sampling = self.nerf_cfg.uniform_sampling
        self.cat_rgb = self.nerf_cfg.cat_rgb
        self.t_to_s, self.s_to_t = construct_ray_warps(self.nerf_near, self.nerf_far, uniform=self.uniform_sampling)

        # feature extraction
        # self.feature_backbone = build_backbone(model_cfg.feature_backbone)
        # self.feature_neck = feature_extraction_neck(model_cfg.feature_neck)

        if getattr(model_cfg, 'sem_neck', None):
            self.sem_neck = build_neck(model_cfg.sem_neck)
        else:
            self.sem_neck = None

        if self.use_qkv:
            self.to_q = nn.Sequential(
                    nn.Linear(3, self.volume_dim),
                    nn.ReLU(),
                    nn.Linear(self.volume_dim, self.volume_dim),
                )
            self.to_k = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, self.volume_dim),
                )
            self.to_v = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, self.volume_dim),
                )
        # direct repeat the image feature
        else:
            if self.pe == 'learned':
                self.positional_embedding = nn.Sequential(
                    nn.Linear(3, self.volume_dim),
                    nn.ReLU(),
                    nn.Linear(self.volume_dim, self.volume_dim),
                )
                FUSION_INPUT_DIM = self.volume_dim +256
            elif self.pe == 'no':
                self.positional_embedding = None
                FUSION_INPUT_DIM = self.volume_dim + 3
            elif self.pe == 'sin':
                raise NotImplementedError
            else:
                raise NotImplementedError
            self.simple_fusion = nn.Sequential(
                nn.Conv3d(FUSION_INPUT_DIM, self.volume_dim+128, 3, 1, 1, bias=True),
                nn.ReLU(),
                nn.Conv3d(self.volume_dim+128, self.volume_dim, 3, 1, 1, bias=True)
            )

        # VOXEL_INPUT_DIM += 32
        if self.use_nerf:
            self.density = LaplaceDensity(beta=0.01)
            self.sdf_conv = nn.Sequential(
                nn.Conv3d(self.volume_dim, self.volume_dim, 3, 1, 1, bias=True),
                nn.Softplus(beta=100),
                nn.Conv3d(self.volume_dim, self.volume_dim, 3, 1, 1, bias=True),
                nn.Softplus(beta=100),
                nn.Conv3d(self.volume_dim, self.volume_dim + 1, 3, 1, 1, bias=True),
            )
            self.rgb_conv = nn.Sequential(
                nn.Conv3d(self.volume_dim, 3, 3, 1, 1, bias=False),
                nn.Sigmoid(),
            )

            self.depth_upsample = nn.UpsamplingBilinear2d(scale_factor=16.)
            self.rgb_upsample = nn.UpsamplingBilinear2d(scale_factor=16.)
            # self.rgb_upsample = nn.Sequential(
            #     nn.UpsamplingBilinear2d(scale_factor=2.),
            #     nn.Conv2d(self.volume_dim, self.image_dim, 3, 1, 1),
            #     nn.LeakyReLU(0.2, inplace=True),
            #     nn.UpsamplingBilinear2d(scale_factor=2.),
            #     nn.Conv2d(self.image_dim, 3, 3, 1, 1),
            #     nn.Sigmoid(),
            # )
        # rpn3d convs
        RPN3D_INPUT_DIM = self.volume_dim
        if self.cat_rgb:
            RPN3D_INPUT_DIM += 3
        rpn3d_convs = []
        for i in range(self.num_3dconvs):
            rpn3d_convs.append(
                nn.Sequential(
                    # convbn_3d(RPN3D_INPUT_DIM if i == 0 else self.rpn3d_dim,
                    #           self.rpn3d_dim, 3, 1, 1, gn=self.GN),
                    nn.Conv3d(RPN3D_INPUT_DIM, self.rpn3d_dim, 3, 1, 1, bias=True),
                    nn.ReLU(inplace=True)))
        # self.rpn3d_convs = nn.Sequential(*rpn3d_convs)
        # self.rpn3d_pool = torch.nn.AvgPool3d((4, 1, 1), stride=(4, 1, 1))
        # self.num_3d_features = self.rpn3d_dim

        # prepare tensors
        self.prepare_coordinates_3d(point_cloud_range, voxel_size, grid_size)
        self.init_params()

        feature_backbone_pretrained = getattr(model_cfg, 'feature_backbone_pretrained', None)
        if feature_backbone_pretrained:
            self.feature_backbone.init_weights(pretrained=feature_backbone_pretrained)

    def prepare_coordinates_3d(self, point_cloud_range, voxel_size, grid_size, sample_rate=(1, 1, 1)):
        self.X_MIN, self.Y_MIN, self.Z_MIN = point_cloud_range[:3]
        self.X_MAX, self.Y_MAX, self.Z_MAX = point_cloud_range[3:]
        self.VOXEL_X_SIZE, self.VOXEL_Y_SIZE, self.VOXEL_Z_SIZE = voxel_size
        self.GRID_X_SIZE, self.GRID_Y_SIZE, self.GRID_Z_SIZE = grid_size

        self.VOXEL_X_SIZE /= sample_rate[0]
        self.VOXEL_Y_SIZE /= sample_rate[1]
        self.VOXEL_Z_SIZE /= sample_rate[2]

        self.GRID_X_SIZE *= sample_rate[0]
        self.GRID_Y_SIZE *= sample_rate[1]
        self.GRID_Z_SIZE *= sample_rate[2]

        zs = torch.linspace(self.Z_MIN + self.VOXEL_Z_SIZE / 2., self.Z_MAX - self.VOXEL_Z_SIZE / 2.,
                            self.GRID_Z_SIZE, dtype=torch.float32)
        ys = torch.linspace(self.Y_MIN + self.VOXEL_Y_SIZE / 2., self.Y_MAX - self.VOXEL_Y_SIZE / 2.,
                            self.GRID_Y_SIZE, dtype=torch.float32)
        xs = torch.linspace(self.X_MIN + self.VOXEL_X_SIZE / 2., self.X_MAX - self.VOXEL_X_SIZE / 2.,
                            self.GRID_X_SIZE, dtype=torch.float32)
        zs, ys, xs = torch.meshgrid(zs, ys, xs)
        coordinates_3d = torch.stack([xs, ys, zs], dim=-1)
        self.coordinates_3d = coordinates_3d.float()
        self.norm_coordinates_3d = (self.coordinates_3d - torch.as_tensor([self.X_MIN, self.Y_MIN, self.Z_MIN],
                                                                          device=self.coordinates_3d.device)) / \
                                   torch.as_tensor(
                                       [self.X_MAX - self.X_MIN, self.Y_MAX - self.Y_MIN, self.Z_MAX - self.Z_MIN],
                                       device=self.coordinates_3d.device)
        self.norm_coordinates_3d = self.norm_coordinates_3d * 2. - 1.

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[
                    2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                torch.nn.init.normal_(m.weight, 0.0, np.sqrt(2) / np.sqrt(self.volume_dim))

    def forward(self, batch_dict,view_id=1,debug=False):
        stereo_feat=batch_dict['img_feat'][:,view_id,:,:,:]#view_id=1表示front视角
        images=batch_dict['img_inputs'][0][:,view_id,:,:,:]
        # 我们的feature neck是没有 sem feature的输出的，这个先保留
        N,C,H,W=stereo_feat.shape
       
        # construct  frustum
        frustum_feature = F.interpolate(stereo_feat, scale_factor=1.0)[:, :, None, :, :]  # (b, C, 1, h, w)

        # sample on disparity
        s_vals = sample_along_rays(N, self.nerf_num_samples, randomized=self.training)

        # get normlized 2d coordinates
        norm_coord_2d, dir_coord_2d = grid_generation(frustum_feature.shape[-2], frustum_feature.shape[-1])
        norm_coord_2d = norm_coord_2d[None, :, :, None, :].repeat(N, 1, 1, self.nerf_num_samples, 1)  # (b, h, w, d, 2)
        sampled_disparity = s_vals[:, :-1][:, None, None, :, None].repeat(1, frustum_feature.shape[-2],
                                                                          frustum_feature.shape[-1], 1, 1)
        
        import torch
        norm_coord_frustum = torch.cat([norm_coord_2d, sampled_disparity], dim=-1).cuda()  # (b, h, w, d, 3)

        # get directions
        directions = []
        intrs=batch_dict['img_inputs'][3][:,view_id,:,:]
                
        for i in range(N):
            dir_coord_2d = dir_coord_2d * images.shape[-1] // frustum_feature.shape[-1]
            
            calibs_Proj_P2=torch.zeros((3, 4), dtype=torch.float32).to(intrs.device)
            calibs_Proj_P2[:3,:3]=intrs[i]
            
            dir_coord_3d = unproject_image_to_rect(dir_coord_2d.cuda(), calibs_Proj_P2)
            direction = dir_coord_3d[:, :, 1, :] - dir_coord_3d[:, :, 0, :]
            direction /= images.shape[-1] // frustum_feature.shape[-1]
            directions.append(direction)
        directions = torch.stack(directions, dim=0)

        # construct frustum features representations
        if self.use_qkv:
            # self.to_q.to(norm_coord_frustum.device)
            # self.to_k.to(frustum_feature.device)
            # self.to_v.to(frustum_feature.device)
            
            query = self.to_q(norm_coord_frustum)  # (b, h, w, d, C)
            key = self.to_k(rearrange(frustum_feature, 'b c d h w -> b h w d c'))  # (b h w 1 32)
            value = self.to_v(rearrange(frustum_feature, 'b c d h w -> b h w d c'))  # (b h w 1 32)

            dot = query * key  # (b h w d 32)
            att = dot.softmax(dim=-2)  # (b h w d 32)
            frustum_feature = att * value  # (b, h, w, d, 32)
            frustum_feature = rearrange(frustum_feature, 'b h w d c -> b c d h w')
        else:
            if self.pe == 'learned':
                positional_embedding = self.positional_embedding(norm_coord_frustum)  # (b, h, w, d, 32)
            elif self.pe == 'no':
                positional_embedding = norm_coord_frustum
            else:
                raise NotImplementedError
            positional_embedding = rearrange(positional_embedding, 'b h w d c -> b c d h w')
            frustum_feature = repeat(frustum_feature, 'b c d h w -> b c (repeat d) h w', repeat=positional_embedding.shape[2])
            frustum_feature = torch.cat([positional_embedding, frustum_feature], dim=1)
            frustum_feature = self.simple_fusion(frustum_feature)

        # build nerd representations
        if self.use_nerf:
            # sdf
            sdf_output = self.sdf_conv(frustum_feature)
            sdf = sdf_output[:, :1]
            feature_vectors = sdf_output[:, 1:]

            # rgb
            rgb = self.rgb_conv(feature_vectors)

            # density
            density = self.density(sdf)  # b c d h w
            
            # weights
            weights, tdist = compute_alpha_weights(density, s_vals.cuda(), directions, self.s_to_t)
            acc = weights.sum(dim=1)

            # reconstruct depth and rgb image
            t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
            background_rgb = rearrange((1 - acc)[..., None] * torch.tensor([1.0, 1.0, 1.0]).float().cuda(),
                                       'b h w c -> b c h w')
            rgb_values = torch.sum(weights.unsqueeze(1) * rgb, dim=2) + background_rgb
            # rgb_values = rearrange(rgb_values, 'b h w c -> b c h w')
            nerf_rgb_preds = self.rgb_upsample(rgb_values)
            nerf_rgb_preds=F.interpolate(nerf_rgb_preds,size=(256,704))
           
            background_depth = (1 - acc) * torch.tensor([1.0]).float().cuda() * self.nerf_far
            depth = (weights * t_mids[..., None, None]).sum(dim=1) + background_depth
            depth = depth.unsqueeze(1)
            nerf_depth_preds = self.depth_upsample(depth)
            nerf_depth_preds=F.interpolate(nerf_depth_preds,size=(256,704))

            # near surface sdf loss 
            
            #先去掉这一段sdf的loss试一下
            # if self.training:
            #     pts_3d = torch.stack(batch_dict['points'])
            #     norm_coord_pts2imgs = []
            #     valid_pts = []
            #     for i in range(N):
            #         # in pseudo lidar coord
            #         c3d = project_pseudo_lidar_to_rectcam(pts_3d[i])
                    
            #         calibs_Proj_P2=torch.zeros((3, 4), dtype=torch.float32).to(intrs.device)
            #         calibs_Proj_P2[:3,:3]=intrs[i]
                    
            #         coord_pts2img = project_rect_to_image(
            #             c3d,
            #             calibs_Proj_P2.float().cuda())
            #         coord_pts2img = coord_pts2img.view(1, 1, -1, 3)
            #         img_shape = batch_dict['image_shape'][i]
            #         valid_mask_pts = (coord_pts2img[..., 0] >= 0) & (coord_pts2img[..., 0] <= img_shape[1]) & \
            #                         (coord_pts2img[..., 1] >= 0) & (coord_pts2img[..., 1] <= img_shape[0])
            #         valid_pts.append(valid_mask_pts)

            #         # TODO: crop augmentation
            #         crop_x1, crop_x2 = 0, left.shape[3]
            #         crop_y1, crop_y2 = 0, left.shape[2]
            #         norm_coord_pts2img = (coord_pts2img[..., :2] - torch.as_tensor([crop_x1, crop_y1],
            #                                                                      device=coord_pts2img.device)) / torch.as_tensor(
            #             [crop_x2 - 1 - crop_x1, crop_y2 - 1 - crop_y1],
            #             device=coord_pts2img.device)
            #         norm_coord_pts2img_depth = self.t_to_s(coord_pts2img[..., 2:3])
            #         norm_coord_pts2img = torch.cat([norm_coord_pts2img, norm_coord_pts2img_depth], dim=-1)
            #         norm_coord_pts2img = norm_coord_pts2img * 2. - 1.
            #         norm_coord_pts2imgs.append(norm_coord_pts2img)

            #     norm_coord_pts2imgs = torch.stack(norm_coord_pts2imgs, dim=0)
            #     valid_pts = torch.stack(valid_pts, dim=0)
            #     valid_pts = valid_pts & (norm_coord_pts2imgs[..., 2] >= -1.) & (norm_coord_pts2imgs[..., 2] <= 1.)
            #     valid_pts = valid_pts.float()

            #     pts_sdf = F.grid_sample(sdf, norm_coord_pts2imgs, align_corners=True)  # (1, 1, 1, 1, pts)
            #     pts_sdf = pts_sdf * valid_pts[: None] # (1, 32, 20, 304, 288)
            #     batch_dict['sdf_near_surface'] = pts_sdf

        else:
            #这个else是不使用nerf的else，别看错了
            feature_vectors = frustum_feature
     
        # get 3d voxel normalized coordinates
        # coordinates_3d = self.coordinates_3d.cuda()
        # batch_dict['coord_3d'] = coordinates_3d
        # norm_coord_3d2imgs = []
        # coord_3d2imgs = []
        # valids2d = []

        # for i in range(N):
        #     # voxel 3d to 2d image
        #     c3d = coordinates_3d.view(-1, 3)
         
        #     # in pseudo lidar coord
        #     c3d = project_pseudo_lidar_to_rectcam(c3d)
            
        #     calibs_Proj_P2=torch.zeros((3, 4), dtype=torch.float32).to(intrs.device)
        #     calibs_Proj_P2[:3,:3]=intrs[i]
            
        #     coord_3d2img = project_rect_to_image(
        #         c3d,
        #         calibs_Proj_P2.float().cuda())

        #     coord_3d2img = coord_3d2img.view(*self.coordinates_3d.shape[:3], 3)

        #     coord_3d2imgs.append(coord_3d2img)

        #     img_shape = batch_dict['img_inputs'][0][i][0][0].shape
        #     valid_mask_2d = (coord_3d2img[..., 0] >= 0) & (coord_3d2img[..., 0] <= img_shape[1]) & \
        #                     (coord_3d2img[..., 1] >= 0) & (coord_3d2img[..., 1] <= img_shape[0])
        #     valids2d.append(valid_mask_2d)

        #     # TODO: crop augmentation
        #     crop_x1, crop_x2 = 0, img_shape[-1]
        #     crop_y1, crop_y2 = 0, img_shape[-2]
        #     norm_coord_3d2img = (coord_3d2img[..., :2] - torch.as_tensor([crop_x1, crop_y1],
        #                                                                  device=coord_3d2img.device)) / torch.as_tensor(
        #         [crop_x2 - 1 - crop_x1, crop_y2 - 1 - crop_y1],
        #         device=coord_3d2img.device)
        #     norm_coord_3d2img_depth = self.t_to_s(coord_3d2img[..., 2:3])
        #     norm_coord_3d2img = torch.cat([norm_coord_3d2img, norm_coord_3d2img_depth], dim=-1)
        #     norm_coord_3d2img = norm_coord_3d2img * 2. - 1.
        #     norm_coord_3d2imgs.append(norm_coord_3d2img)

        # norm_coord_3d2imgs = torch.stack(norm_coord_3d2imgs, dim=0)
        # coord_3d2imgs = torch.stack(coord_3d2imgs, dim=0)
        # valids2d = torch.stack(valids2d, dim=0)

        # batch_dict['norm_coord_3d2imgs'] = norm_coord_3d2imgs
        # batch_dict['coord_3d2imgs'] = coord_3d2imgs

        # valids = valids2d & (norm_coord_3d2imgs[..., 2] >= -1.) & (norm_coord_3d2imgs[..., 2] <= 1.)
        # batch_dict['valids'] = valids
        # valids = valids.float()

        # # Retrieve Voxel Feature from Frustum
        # Voxel = F.grid_sample(feature_vectors, norm_coord_3d2imgs, align_corners=True)
        # if self.use_nerf:
        #     voxel_density = F.grid_sample(density, norm_coord_3d2imgs, align_corners=True)  # (1, 1, 20, 304, 288)
        #     bev_density_vis = torch.sum(voxel_density.cpu().detach(), dim=2)
        #     batch_dict['bev_density_vis'] = []
        #     batch_dict['bev_density_vis'].append(
        #         torch.flip(rearrange(bev_density_vis, 'b c w d -> b c d w'), dims=[2, 3]))  # (1, 1, 288, 304)
        #     if self.cat_rgb:
        #         voxel_rgb = F.grid_sample(rgb, norm_coord_3d2imgs, align_corners=True)
        #         Voxel = torch.cat([Voxel, voxel_rgb], dim=1)
        #     if self.voxel_attentionbydensity:
        #         Voxel = Voxel * voxel_density.tanh()
         
        # Voxel = Voxel * valids[:, None, :, :, :]  # (1, 32, 20, 304, 288)
   

        # # begin 3d detection
        # Voxel = self.rpn3d_convs(Voxel)  # (64, 190, 20, 300)
        # batch_dict['volume_features_nopool'] = Voxel

        # Voxel = self.rpn3d_pool(Voxel)  # [B, C, Nz, Ny, Nx] in cam view

        # batch_dict['volume_features'] = Voxel
        
        mean = torch.from_numpy(np.array([123.675, 116.28, 103.53], dtype=np.float32)).to(images.device).reshape(3,1,1)
        std = torch.from_numpy(np.array([58.395, 57.12, 57.375], dtype=np.float32)).to(images.device).reshape(3,1,1)
        images=images*std+mean
    
        if debug:
            import torchvision.transforms as transforms
            import torch
            to_pil = transforms.ToPILImage()
            image_pil = to_pil(images[0])

            image_pil.save(f'/data0/jinhongbo/0829/bevperception/vis_results/all_gt_{view_id}.png') 
            
            image_pil = to_pil(nerf_rgb_preds[0])

            image_pil.save(f'/data0/jinhongbo/0829/bevperception/vis_results/all_render_{view_id}.png') 
            
        rgb_l1_loss=l1_loss(nerf_rgb_preds,images)
        rgb_ssim_loss=1-ssim(nerf_rgb_preds,images)
        RGB_loss=rgb_l1_loss+rgb_ssim_loss
        return feature_vectors,density, RGB_loss, nerf_depth_preds
            
           
