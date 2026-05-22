import torch
from mmdet3d.core.bbox.utils import decode_bbox
from .utils import rotation_3d_in_axis, DUMP
from mmdet3d.ops.csrc.wrapper import msmv_sampling, TRTMSMVSampling, msmv_sampling_pytorch

import torch.nn.functional as F

from tools.misc.vis_tools import print_gt_and_bev, print_pcgt_on_bev_radar, print_sample_and_bev
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention


def make_sample_points(query_bbox, offset, pc_range):
    '''
    query_bbox: [B, Q, 10]
    offset: [B, Q, num_points, 4], normalized by stride
    '''
    query_bbox = decode_bbox(query_bbox, pc_range)  # [B, Q, 9]

    xyz = query_bbox[..., 0:3]  # [B, Q, 3]
    wlh = query_bbox[..., 3:6]  # [B, Q, 3]
    ang = query_bbox[..., 6:7]  # [B, Q, 1]

    delta_xyz = offset[..., 0:3]  # [B, Q, P, 3] P:self.num_groups * self.num_points
    delta_xyz = wlh[:, :, None, :] * delta_xyz  # [B, Q, P, 3]
    delta_xyz = rotation_3d_in_axis(delta_xyz, ang)  # [B, Q, P, 3]
    sample_xyz = xyz[:, :, None, :] + delta_xyz  # [B, Q, P, 3]

    return sample_xyz  # [B, Q, P, 3]


def sampling_4d(sample_points, mlvl_feats, scale_weights, lidar2img, image_h, image_w, eps=1e-5):
    """
    Args:
        sample_points: 3D sampling points in shape [B, Q, T, G, P, 3]
        mlvl_feats: list of multi-scale features from neck, each in shape [B*T*G, C, N, H, W]
        scale_weights: weights for multi-scale aggregation, [B, Q, G, T, P, L]
        lidar2img: 4x4 projection matrix in shape [B, TN, 4, 4]
    Symbol meaning:
        B: batch size
        Q: num of queries
        T: num of frames
        G: num of groups (we follow the group sampling mechanism of AdaMixer)
        P: num of sampling points per frame per group
        N: num of views (six for nuScenes)
        L: num of layers of feature pyramid (typically it is 4: C2, C3, C4, C5)
    """

    B, Q, T, G, P, _ = sample_points.shape  # [B, Q, T, G, P, 3]
    N = 6
    # openad
    # N = 1
    
    sample_points = sample_points.reshape(B, Q, T, G * P, 3)
    # get the projection matrix
    lidar2img = lidar2img[:, :, None, None, :, :]  # [B, TN, 1, 1, 4, 4]
    lidar2img = lidar2img.expand(B, T*N, Q, G * P, 4, 4)
    lidar2img = lidar2img.reshape(B, T, N, Q, G*P, 4, 4)

    # expand the points
    ones = torch.ones_like(sample_points[..., :1])
    sample_points = torch.cat([sample_points, ones], dim=-1)  # [B, Q, GP, 4]
    sample_points = sample_points[:, :, None, ..., None]     # [B, Q, T, GP, 4]
    sample_points = sample_points.expand(B, Q, N, T, G * P, 4, 1)
    sample_points = sample_points.transpose(1, 3)   # [B, T, N, Q, GP, 4, 1]

    # project 3d sampling points to N views
    sample_points_cam = torch.matmul(lidar2img, sample_points).squeeze(-1)  # [B, T, N, Q, GP, 4]

    # homo coord -> pixel coord
    homo = sample_points_cam[..., 2:3]
    homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + eps)
    sample_points_cam = sample_points_cam[..., 0:2] / homo_nonzero  # [B, T, N, Q, GP, 2]

    # normalize
    sample_points_cam[..., 0] /= image_w
    sample_points_cam[..., 1] /= image_h

    # check if out of image
    valid_mask = ((homo > eps) \
        & (sample_points_cam[..., 1:2] > 0.0)
        & (sample_points_cam[..., 1:2] < 1.0)
        & (sample_points_cam[..., 0:1] > 0.0)
        & (sample_points_cam[..., 0:1] < 1.0)
    ).squeeze(-1).float()  # [B, T, N, Q, GP]

    # for visualization only
    if DUMP.enabled:
        torch.save(torch.cat([sample_points_cam, homo_nonzero], dim=-1).cpu(),
                   '{}/sample_points_cam_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))
        torch.save(valid_mask.cpu(),
                   '{}/sample_points_cam_valid_mask_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

    valid_mask = valid_mask.permute(0, 1, 3, 4, 2)  # [B, T, Q, GP, N]
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 4, 2, 5)  # [B, T, Q, GP, N, 2]

    # prepare batched indexing
    i_batch = torch.arange(B, dtype=torch.long, device=sample_points.device)
    i_query = torch.arange(Q, dtype=torch.long, device=sample_points.device)
    i_time = torch.arange(T, dtype=torch.long, device=sample_points.device)
    i_point = torch.arange(G * P, dtype=torch.long, device=sample_points.device)
    i_batch = i_batch.view(B, 1, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_time = i_time.view(1, T, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_query = i_query.view(1, 1, Q, 1, 1).expand(B, T, Q, G * P, 1)
    i_point = i_point.view(1, 1, 1, G * P, 1).expand(B, T, Q, G * P, 1)
    
    # we only keep at most one valid sampling point, see https://zhuanlan.zhihu.com/p/654821380
    i_view = torch.argmax(valid_mask, dim=-1)[..., None]  # [B, T, Q, GP, 1]

    # index the only one sampling point and its valid flag
    sample_points_cam = sample_points_cam[i_batch, i_time, i_query, i_point, i_view, :]  # [B, Q, GP, 1, 2]
    valid_mask = valid_mask[i_batch, i_time, i_query, i_point, i_view]  # [B, Q, GP, 1]

    # treat the view index as a new axis for grid_sample and normalize the view index to [0, 1]
    sample_points_cam = torch.cat([sample_points_cam, i_view[..., None].float() / (N - 1)], dim=-1)

    # reorganize the tensor to stack T and G to the batch dim for better parallelism
    sample_points_cam = sample_points_cam.reshape(B, T, Q, G, P, 1, 3)
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 2, 4, 5, 6)  # [B, T, G, Q, P, 1, 3]
    sample_points_cam = sample_points_cam.reshape(B*T*G, Q, P, 3)

    # reorganize the tensor to stack T and G to the batch dim for better parallelism
    scale_weights = scale_weights.reshape(B, Q, G, T, P, -1)
    scale_weights = scale_weights.permute(0, 2, 3, 1, 4, 5)
    scale_weights = scale_weights.reshape(B*G*T, Q, P, -1)

    # multi-scale multi-view grid sample
    final = msmv_sampling(mlvl_feats, sample_points_cam, scale_weights)
    # reorganize the sampled features
    C = final.shape[2]  # [BTG, Q, C, P]
    final = final.reshape(B, T, G, Q, C, P)
    final = final.permute(0, 3, 2, 1, 5, 4)
    final = final.flatten(3, 4)  # [B, Q, G, FP, C]

    return final


def sampling_4d_trt(sample_points, mlvl_feats, scale_weights, lidar2img, image_h, image_w, eps=1e-5):
    """
    Args:
        sample_points: 3D sampling points in shape [B, Q, T, G, P, 3]
        mlvl_feats: list of multi-scale features from neck, each in shape [B*T*G, C, N, H, W]
        scale_weights: weights for multi-scale aggregation, [B, Q, G, T, P, L]
        lidar2img: 4x4 projection matrix in shape [B, TN, 4, 4]
    Symbol meaning:
        B: batch size
        Q: num of queries
        T: num of frames
        G: num of groups (we follow the group sampling mechanism of AdaMixer)
        P: num of sampling points per frame per group
        N: num of views (six for nuScenes)
        L: num of layers of feature pyramid (typically it is 4: C2, C3, C4, C5)
    """

    B, Q, T, G, P, _ = sample_points.shape  # [B, Q, T, G, P, 3]
    N = 6
    # import ipdb;ipdb.set_trace()
    sample_points = sample_points.reshape(B, Q, T, G * P, 3)
    # get the projection matrix
    lidar2img = lidar2img[:, :, None, None, :, :]  # [B, TN, 1, 1, 4, 4]
    lidar2img = lidar2img.expand(B, T*N, Q, G * P, 4, 4)
    lidar2img = lidar2img.reshape(B, T, N, Q, G*P, 4, 4)

    # expand the points
    ones = torch.ones_like(sample_points[..., :1])
    sample_points = torch.cat([sample_points, ones], dim=-1)  # [B, Q, GP, 4]
    sample_points = sample_points[:, :, None, ..., None]     # [B, Q, T, GP, 4]
    sample_points = sample_points.expand(B, Q, N, T, G * P, 4, 1)
    sample_points = sample_points.transpose(1, 3)   # [B, T, N, Q, GP, 4, 1]

    # project 3d sampling points to N views
    sample_points_cam = torch.matmul(lidar2img, sample_points) # torch.Size([1, 8, 6, 900, 16, 4, 1])
    # sample_points_cam = sample_points_cam.squeeze(-1)  # [B, T, N, Q, GP, 4]
    sample_points_cam = sample_points_cam[...,0]  # [B, T, N, Q, GP, 4]

    # homo coord -> pixel coord
    homo = sample_points_cam[..., 2:3]
    homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + eps)
    sample_points_cam = sample_points_cam[..., 0:2] / homo_nonzero  # [B, T, N, Q, GP, 2]
    # normalize
    sample_points_cam[..., 0] /= image_w
    sample_points_cam[..., 1] /= image_h

    # check if out of image
    # valid_mask = ((homo > eps) \
    #     & (sample_points_cam[..., 1:2] > 0.0)
    #     & (sample_points_cam[..., 1:2] < 1.0)
    #     & (sample_points_cam[..., 0:1] > 0.0)
    #     & (sample_points_cam[..., 0:1] < 1.0) # torch.Size([1, 8, 6, 900, 16, 1])
    # ).squeeze(-1).float()  # [B, T, N, Q, GP]
    valid_mask = ((homo > eps) \
        & (sample_points_cam[..., 1:2] > 0.0)
        & (sample_points_cam[..., 1:2] < 1.0)
        & (sample_points_cam[..., 0:1] > 0.0)
        & (sample_points_cam[..., 0:1] < 1.0) # torch.Size([1, 8, 6, 900, 16, 1])
    )[...,0].float()  # [B, T, N, Q, GP]

    valid_mask = valid_mask.permute(0, 1, 3, 4, 2)  # [B, T, Q, GP, N]
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 4, 2, 5)  # [B, T, Q, GP, N, 2]

    # prepare batched indexing
    i_batch = torch.arange(B, dtype=torch.long, device=sample_points.device)
    i_query = torch.arange(Q, dtype=torch.long, device=sample_points.device)
    i_time = torch.arange(T, dtype=torch.long, device=sample_points.device)
    i_point = torch.arange(G * P, dtype=torch.long, device=sample_points.device)
    i_batch = i_batch.view(B, 1, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_time = i_time.view(1, T, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_query = i_query.view(1, 1, Q, 1, 1).expand(B, T, Q, G * P, 1)
    i_point = i_point.view(1, 1, 1, G * P, 1).expand(B, T, Q, G * P, 1)
    
    # we only keep at most one valid sampling point, see https://zhuanlan.zhihu.com/p/654821380
    i_view = torch.argmax(valid_mask, dim=-1)[..., None]  # [B, T, Q, GP, 1]

    # index the only one sampling point and its valid flag
    sample_points_cam = sample_points_cam[i_batch, i_time, i_query, i_point, i_view, :]  # [B, Q, GP, 1, 2]
    valid_mask = valid_mask[i_batch, i_time, i_query, i_point, i_view]  # [B, Q, GP, 1]

    # treat the view index as a new axis for grid_sample and normalize the view index to [0, 1]
    sample_points_cam = torch.cat([sample_points_cam, i_view[..., None].float() / (N - 1)], dim=-1)

    # reorganize the tensor to stack T and G to the batch dim for better parallelism
    sample_points_cam = sample_points_cam.reshape(B, T, Q, G, P, 1, 3)
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 2, 4, 5, 6)  # [B, T, G, Q, P, 1, 3]
    sample_points_cam = sample_points_cam.reshape(B*T*G, Q, P, 3)

    # reorganize the tensor to stack T and G to the batch dim for better parallelism
    scale_weights = scale_weights.reshape(B, Q, G, T, P, -1)
    scale_weights = scale_weights.permute(0, 2, 3, 1, 4, 5)
    scale_weights = scale_weights.reshape(B*G*T, Q, P, -1)

    # multi-scale multi-view grid sample
    final = TRTMSMVSampling.apply(mlvl_feats[0], mlvl_feats[1], mlvl_feats[2], mlvl_feats[3], sample_points_cam, scale_weights)
    # reorganize the sampled features
    C = final.shape[2]  # [BTG, Q, C, P]
    final = final.reshape(B, T, G, Q, C, P)
    final = final.permute(0, 3, 2, 1, 5, 4)
    final = final.flatten(3, 4)  # [B, Q, G, FP, C]

    return final


def sampling_4d_trt_debug(sample_points, mlvl_feats, scale_weights, lidar2img, image_h, image_w, eps=1e-5):
    """
    Args:
        sample_points: 3D sampling points in shape [B, Q, T, G, P, 3]
        mlvl_feats: list of multi-scale features from neck, each in shape [B*T*G, C, N, H, W]
        scale_weights: weights for multi-scale aggregation, [B, Q, G, T, P, L]
        lidar2img: 4x4 projection matrix in shape [B, TN, 4, 4]
    Symbol meaning:
        B: batch size
        Q: num of queries
        T: num of frames
        G: num of groups (we follow the group sampling mechanism of AdaMixer)
        P: num of sampling points per frame per group
        N: num of views (six for nuScenes)
        L: num of layers of feature pyramid (typically it is 4: C2, C3, C4, C5)
    """

    B, Q, T, G, P, _ = sample_points.shape  # [B, Q, T, G, P, 3]
    N = 6
    # import ipdb;ipdb.set_trace()
    sample_points = sample_points.reshape(B, Q, T, G * P, 3)
    # get the projection matrix
    lidar2img = lidar2img[:, :, None, None, :, :]  # [B, TN, 1, 1, 4, 4]
    lidar2img = lidar2img.expand(B, T*N, Q, G * P, 4, 4)
    lidar2img = lidar2img.reshape(B, T, N, Q, G*P, 4, 4)

    # expand the points
    ones = torch.ones_like(sample_points[..., :1])
    sample_points = torch.cat([sample_points, ones], dim=-1)  # [B, Q, GP, 4]
    sample_points = sample_points[:, :, None, ..., None]     # [B, Q, T, GP, 4]
    sample_points = sample_points.expand(B, Q, N, T, G * P, 4, 1)
    sample_points = sample_points.transpose(1, 3)   # [B, T, N, Q, GP, 4, 1]

    # project 3d sampling points to N views
    sample_points_cam = torch.matmul(lidar2img, sample_points) # torch.Size([1, 8, 6, 900, 16, 4, 1])
    
    # sample_points_cam = sample_points_cam.squeeze(-1)  # [B, T, N, Q, GP, 4]
    sample_points_cam = sample_points_cam[...,0]  # [B, T, N, Q, GP, 4]

    # homo coord -> pixel coord
    homo = sample_points_cam[..., 2:3]
    homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + eps)
    sample_points_cam = sample_points_cam[..., 0:2] / homo_nonzero  # [B, T, N, Q, GP, 2]
    # normalize
    sample_points_cam[..., 0] /= image_w
    sample_points_cam[..., 1] /= image_h

    # check if out of image
    # valid_mask = ((homo > eps) \
    #     & (sample_points_cam[..., 1:2] > 0.0)
    #     & (sample_points_cam[..., 1:2] < 1.0)
    #     & (sample_points_cam[..., 0:1] > 0.0)
    #     & (sample_points_cam[..., 0:1] < 1.0) # torch.Size([1, 8, 6, 900, 16, 1])
    # ).squeeze(-1).float()  # [B, T, N, Q, GP]
    valid_mask = ((homo > eps) \
        & (sample_points_cam[..., 1:2] > 0.0)
        & (sample_points_cam[..., 1:2] < 1.0)
        & (sample_points_cam[..., 0:1] > 0.0)
        & (sample_points_cam[..., 0:1] < 1.0) # torch.Size([1, 8, 6, 900, 16, 1])
    )[...,0].float()  # [B, T, N, Q, GP]

    valid_mask = valid_mask.permute(0, 1, 3, 4, 2)  # [B, T, Q, GP, N]
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 4, 2, 5)  # [B, T, Q, GP, N, 2]

    # prepare batched indexing
    i_batch = torch.arange(B, dtype=torch.long, device=sample_points.device)
    i_query = torch.arange(Q, dtype=torch.long, device=sample_points.device)
    i_time = torch.arange(T, dtype=torch.long, device=sample_points.device)
    i_point = torch.arange(G * P, dtype=torch.long, device=sample_points.device)
    i_batch = i_batch.view(B, 1, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_time = i_time.view(1, T, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_query = i_query.view(1, 1, Q, 1, 1).expand(B, T, Q, G * P, 1)
    i_point = i_point.view(1, 1, 1, G * P, 1).expand(B, T, Q, G * P, 1)
    
    # we only keep at most one valid sampling point, see https://zhuanlan.zhihu.com/p/654821380
    i_view = torch.argmax(valid_mask, dim=-1)[..., None]  # [B, T, Q, GP, 1]

    # index the only one sampling point and its valid flag
    sample_points_cam = sample_points_cam[i_batch, i_time, i_query, i_point, i_view, :]  # [B, Q, GP, 1, 2]
    valid_mask = valid_mask[i_batch, i_time, i_query, i_point, i_view]  # [B, Q, GP, 1]

    # treat the view index as a new axis for grid_sample and normalize the view index to [0, 1]
    sample_points_cam = torch.cat([sample_points_cam, i_view[..., None].float() / (N - 1)], dim=-1)

    # reorganize the tensor to stack T and G to the batch dim for better parallelism
    sample_points_cam = sample_points_cam.reshape(B, T, Q, G, P, 1, 3)
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 2, 4, 5, 6)  # [B, T, G, Q, P, 1, 3]
    sample_points_cam = sample_points_cam.reshape(B*T*G, Q, P, 3)

    # reorganize the tensor to stack T and G to the batch dim for better parallelism
    scale_weights = scale_weights.reshape(B, Q, G, T, P, -1)
    scale_weights = scale_weights.permute(0, 2, 3, 1, 4, 5)
    scale_weights = scale_weights.reshape(B*G*T, Q, P, -1)

    # multi-scale multi-view grid sample
    final = TRTMSMVSampling.apply(mlvl_feats[0], mlvl_feats[1], mlvl_feats[2], mlvl_feats[3], sample_points_cam, scale_weights)
    # reorganize the sampled features
    C = final.shape[2]  # [BTG, Q, C, P]
    final = final.reshape(B, T, G, Q, C, P)
    final = final.permute(0, 3, 2, 1, 5, 4)
    final = final.flatten(3, 4)  # [B, Q, G, FP, C]

    return final


def bev_sampling_pytorch(bev_feats, sampling_locations, scale_weights):
    """
    value: [B, N, H1W1 + H2W2..., C]
    sampling_locations: [B, Q, P, 2]
    scale_weights: [B, Q, P, 4]
    """
    #assert scale_weights.shape[-1] == len(bev_feats)

    B, C, _, _ = bev_feats.shape  #(B*T*G, C, H, W)
    _, Q, P, _ = sampling_locations.shape

    #sampling_locations = sampling_locations * 2 - 1
    #sampling_locations = sampling_locations[:, :, :, :]  # [B, Q, P, 2]


    



    # for lvl, feat in enumerate(mlvl_feats):
    #     out = F.grid_sample(
    #         feat, sampling_locations, mode='bilinear',
    #         padding_mode='zeros', align_corners=True,
    #     )[..., 0]  # [B, C, Q, P]
    #     out = out * scale_weights[..., lvl].reshape(B, 1, Q, P)
    #     final += out

    out = F.grid_sample(
            bev_feats, sampling_locations, mode='bilinear',
            padding_mode='zeros', align_corners=True,
        )#[..., 0]  # [B, C, Q, P] (B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

    # output = multi_scale_deformable_attn_pytorch(
    #                 value, rad_spatial_shapes, sampling_locations, attention_weights)
    # out.shape: torch.Size([256, 96, 1290]) out.shape: torch.Size([256, 96, 1290, 4])
    # scale_weights.shape: torch.Size([256, 1290, 4, 1]) [B, C, Q, P]: 256 96 1290 4
    
    # data1 = open("sample_points_bev.txt",'w',encoding="utf-8")
    # print(sample_points_bev[..., 0:2],file=data1)
    # exit(0)
    out = out * scale_weights.reshape(B, 1, Q, P)

    return out.permute(0, 2, 1, 3) #B Q C P

def sampling_bev(sample_points, mlvl_feats, scale_weights, pc_range, eps=1e-5):
    """
    Args:
        sample_points: 3D sampling points in shape [B, Q, T, G, P, 3]
        mlvl_feats: list of multi-scale features from neck, each in shape [B*T*G, C, N, H, W]
        scale_weights: weights for multi-scale aggregation, [B, Q, G, T, P, L]
        lidar2img: 4x4 projection matrix in shape [B, TN, 4, 4]
    Symbol meaning:
        B: batch size
        Q: num of queries
        T: num of frames
        G: num of groups (we follow the group sampling mechanism of AdaMixer)
        P: num of sampling points per frame per group
        N: num of views (six for nuScenes)
        L: num of layers of feature pyramid (typically it is 4: C2, C3, C4, C5)
    """

    #  # group image features in advance for sampling, see `sampling_4d` for more details
    #     for lvl, feat in enumerate(mlvl_feats):
    #         B, TN, GC, H, W = feat.shape  # [B, TN, GC, H, W]
    #         N, T, G, C = 6, TN // 6, 4, GC // 4
    #         feat = feat.reshape(B, T, N, G, C, H, W)

    #         if MSMV_CUDA:  # Our CUDA operator requires channel_last
    #             feat = feat.permute(0, 1, 3, 2, 5, 6, 4)  # [B, T, G, N, H, W, C]
    #             feat = feat.reshape(B*T*G, N, H, W, C)
    #         else:  # Torch's grid_sample requires channel_first
    #             feat = feat.permute(0, 1, 3, 4, 2, 5, 6)  # [B, T, G, C, N, H, W]
    #             feat = feat.reshape(B*T*G, C, N, H, W)

    #         mlvl_feats[lvl] = feat.contiguous()
    
    # sample_points = sample_points[:, :, :1, :, :, :] #RADAR NO TIME

    # data1 = open("sample_points_bev[..., 0:3].txt",'w',encoding="utf-8")
    # print(sample_points[..., 0:3],file=data1)
    # exit(0)

    B, Q, T, G, P, _ = sample_points.shape  # [B, Q, T, G, P, 3] torch.Size([8, 1290, 8, 4, 4, 3])
    #N = 1 #BEV dont need multi-view
    #T = 1 #RADAR NO TIME
    mlvl_feats = mlvl_feats[0] #radar b c h w

    #print_sample_and_bev(mlvl_feats, sample_points) 

    B, GC, H, W = mlvl_feats.shape
    G, C = 4, GC // 4
    mlvl_feats = mlvl_feats.reshape(B, 1, G, C, H, W) # OLD =>(B, 1, N, G, C, H, W)
    mlvl_feats = mlvl_feats.expand(B, T, G, C, H, W) # new 8 Frame radar
    #mlvl_feats = mlvl_feats.permute(0, 1, 2, 3, 4, 5)  # [B, T, G, C, H, W]
    mlvl_feats = mlvl_feats.reshape(B*T*G, C, H, W)
    #print(mlvl_feats.shape)
    #print("test")

    # bev_h = mlvl_feats.shape[2]
    # bev_w = mlvl_feats.shape[3]
    sample_points = sample_points.reshape(B, Q, T, G * P, 3)

    # get the projection matrix
    # lidar2img = lidar2img[:, :, None, None, :, :]  # [B, TN, 1, 1, 4, 4]
    # lidar2img = lidar2img.expand(B, T*N, Q, G * P, 4, 4)
    # lidar2img = lidar2img.reshape(B, T, N, Q, G*P, 4, 4)

    # expand the points
    # ones = torch.ones_like(sample_points[..., :1])
    # sample_points = torch.cat([sample_points, ones], dim=-1)  # [B, Q, GP, 4] #扩展维度 为了lidar2img?? [B, Q, T, GP, 4]
    #sample_points = sample_points[:, :, ..., None]     # OLD [B, Q, T, GP, 4] -> [B, Q, 1, T, GP, 4, 1]

    
    sample_points = sample_points.expand(B, Q, T, G * P, 3)  #OLD 扩展N 多视角
    sample_points_bev = sample_points.transpose(1, 2)   #[B, T, Q, GP, 3] OLD [B, T, N, Q, GP, 4, 1]

    # project 3d sampling points to N views
    #sample_points_cam = torch.matmul(lidar2img, sample_points).squeeze(-1)  # [B, T, N, Q, GP, 4]
    #sample_points_bev = sample_points.squeeze(-1)  # [B, T, N, Q, GP, 4]
 
    # homo coord -> pixel coord 将齐次坐标（homo）转换为像素坐标
    # homo = sample_points_cam[..., 2:3]
    # homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + eps)
    # sample_points_cam = sample_points_cam[..., 0:2] / homo_nonzero  # [B, T, N, Q, GP, 2]
    #sample_points_bev[..., 2:3] =1 OLD RADAR
    sample_points_bev = sample_points_bev[..., 0:2] # [B, T, Q, GP, 2]

    #print_sample_and_bev(mlvl_feats, gt=None, sample=sample_points_bev, name="sampling_locations")
    # normalize
    # sample_points_cam[..., 0] /= image_w
    # sample_points_cam[..., 1] /= image_h
    range_w = pc_range[3]#-pc_range[0]
    range_h = pc_range[4]#-pc_range[1]
    # temp_0 = sample_points_bev[..., 0].clone() #x
    # temp_1 = sample_points_bev[..., 1].clone() #y

    # sample_points_bev[..., 0] = temp_1 #y
    # sample_points_bev[..., 1] = temp_0 #x

    # sample_points_bev[..., 0] = -sample_points_bev[..., 0]  #-y
    # sample_points_bev[..., 1] = -sample_points_bev[..., 1]  #-x

    sample_points_bev[..., 0] /= range_w  #
    sample_points_bev[..., 1] /= range_h  #

    #


    # check if out of image
    # valid_mask = ((homo > eps) \
    #     & (sample_points_cam[..., 1:2] > 0.0)
    #     & (sample_points_cam[..., 1:2] < 1.0)
    #     & (sample_points_cam[..., 0:1] > 0.0)
    #     & (sample_points_cam[..., 0:1] < 1.0)
    # ).squeeze(-1).float()  # [B, T, N, Q, GP]

    # valid_mask = ((sample_points_bev[..., 1:2] > -1)
    #     & (sample_points_bev[..., 1:2] < 1)
    #     & (sample_points_bev[..., 0:1] > -1)
    #     & (sample_points_bev[..., 0:1] < 1)
    # ).squeeze(-1).float()  # [B, T, N, Q, GP]

    # for visualization only
    # if DUMP.enabled:
    #     torch.save(torch.cat([sample_points_cam, homo_nonzero], dim=-1).cpu(),
    #                '{}/sample_points_cam_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))
    #     torch.save(valid_mask.cpu(),
    #                '{}/sample_points_cam_valid_mask_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

    # data1 = open("sample_points_bev.txt",'w',encoding="utf-8")
    # print(sample_points_bev[..., 0:2],file=data1)
    # exit(0)


    #valid_mask = valid_mask.permute(0, 1, 2, 3)  # [B, T, Q, GP]
    #sample_points_bev = sample_points_bev.permute(0, 1, 2, 3, 4)  # [B, T, Q, GP, 2]

    # prepare batched indexing
    i_batch = torch.arange(B, dtype=torch.long, device=sample_points.device)
    i_query = torch.arange(Q, dtype=torch.long, device=sample_points.device)
    i_time = torch.arange(T, dtype=torch.long, device=sample_points.device)
    i_point = torch.arange(G * P, dtype=torch.long, device=sample_points.device)
    i_batch = i_batch.view(B, 1, 1, 1).expand(B, T, Q, G * P)
    i_time = i_time.view(1, T, 1, 1).expand(B, T, Q, G * P)
    i_query = i_query.view(1, 1, Q, 1).expand(B, T, Q, G * P)
    i_point = i_point.view(1, 1, 1, G * P).expand(B, T, Q, G * P)
    
    # we only keep at most one valid sampling point, see https://zhuanlan.zhihu.com/p/654821380
    #i_view = torch.argmax(valid_mask, dim=-1)[..., None]  # [B, T, Q, GP, 1]

    # index the only one sampling point and its valid flag
    sample_points_bev = sample_points_bev[i_batch, i_time, i_query, i_point, :]  # [B, T, Q, GP, 2]
    #valid_mask = valid_mask[i_batch, i_time, i_query, i_point]  # [B, Q, GP]

    # treat the view index as a new axis for grid_sample and normalize the view index to [0, 1]
    #sample_points_cam = torch.cat([sample_points_cam, i_view[..., None].float() / (N - 1)], dim=-1)

    # reorganize the tensor to stack T and G to the batch dim for better parallelism
    # sample_points_cam = sample_points_cam.reshape(B, T, Q, G, P, 1, 3)
    # sample_points_cam = sample_points_cam.permute(0, 1, 3, 2, 4, 5, 6)  # [B, T, G, Q, P, 1, 3]
    # sample_points_cam = sample_points_cam.reshape(B*T*G, Q, P, 3)
    sample_points_bev = sample_points_bev.reshape(B, T, Q, G, P, 2)
    sample_points_bev = sample_points_bev.permute(0, 1, 3, 2, 4, 5)  # [B, T, G, Q, P, 3]
    sample_points_bev = sample_points_bev.reshape(B*T*G, Q, P, 2)
    # reorganize the tensor to stack T and G to the batch dim for better parallelism
    # B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels
    scale_weights = scale_weights.reshape(B, Q, G, T, P, -1)
    scale_weights = scale_weights.permute(0, 2, 3, 1, 4, 5)
    scale_weights = scale_weights.reshape(B*G*T, Q, P, -1)

    # multi-scale multi-view grid sample
    final = bev_sampling_pytorch(mlvl_feats, sample_points_bev, scale_weights)

    # reorganize the sampled features
    C = final.shape[2]  # [BTG, Q, C, P]
    final = final.reshape(B, T, G, Q, C, P)
    final = final.permute(0, 3, 2, 1, 5, 4) #[B, Q, G, T, P, C]
    final = final.flatten(3, 4)  # [B, Q, G, TP, C]
    
    """     
    B: batch size
    Q: num of queries
    T: num of frames
    G: num of groups (we follow the group sampling mechanism of AdaMixer)
    P: num of sampling points per frame per group
    N: num of views (six for nuScenes)
    L: num of layers of feature pyramid (typically it is 4: C2, C3, C4, C5) 
    """

    return final