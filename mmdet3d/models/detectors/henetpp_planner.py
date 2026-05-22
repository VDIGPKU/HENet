from mmdet3d.core import bbox3d2result
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img
import torch
import torchvision
from mmdet.models import DETECTORS
from torch import nn
import numpy as np
from .. import builder
from ...datasets.metric_planning_stp3 import PlanningMetric
from .henetpp import HenetppRC
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmdet.models.utils.builder import TRANSFORMER
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

@DETECTORS.register_module
class HenetppRC_planner(HenetppRC):

    def __init__(self, neck_det=None,
                 data_aug=None,
                 pts_bbox_head=None,
                 motion_encoder=None,
                 motion_decoder=None,
                 train_cfg=None, test_cfg=None,
                 stop_prev_grad=0,
                 loss_plan_reg=None,
                 num_fut_steps=None,
                 **kwargs):

        super().__init__(neck_det=neck_det,
                         data_aug=data_aug,
                         pts_bbox_head=pts_bbox_head,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg,
                         stop_prev_grad=stop_prev_grad,
                         **kwargs)
        self.planning_metric = None

        if motion_encoder is not None:
            self.motion_encoder = builder.build_transformer(motion_encoder)
            assert self.motion_encoder is not None, "motion_encoder build failed!"

        if motion_decoder is not None:
            self.motion_decoder = builder.build_transformer(motion_decoder)
            assert self.motion_decoder is not None, "motion_decoder build failed!"

        self.ego_size = torch.tensor([4.084, 1.730, 1.562])
        self.num_fut_steps = num_fut_steps
        self.ego_volocity_params = nn.Parameter(torch.zeros(1, 1, 2))  # 让 vx, vy 可学习
        self.positional_encoding_proj = nn.Linear(10, 256)
        self.bbox_proj = nn.Sequential(
            nn.Linear(276, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.traj_pre_project = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        self.occ_projection = nn.Sequential(
            nn.Linear(53, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.traj_post_project = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12)
        )

        # self.traj_gru = nn.GRU(input_size=512, hidden_size=128, batch_first=True, bidirectional=False)

        # self.traj_pred_head = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 6 * 2)
        # )

        # freeze perception part
        # exclude_keys = [
        #     "motion_encoder", "motion_decoder", "ego_volocity_params", "positional_encoding_proj", "bbox_proj",
        #     "traj_pre_project",
        #     "occ_projection", "traj_post_project"
        #     "traj_gru", "traj_pred_head"
        # ]
        # for name, param in self.named_parameters():
        #     if any(key in name for key in exclude_keys):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

    def loss_single(self, voxel_semantics, mask_camera, preds):
        """
        Args:
            voxel_semantics: [B, X, Y, Z]
            mask_camera: [B, X, Y, Z]
            preds: [B, X, Y, Z, num_classes]
        """
        loss_ = dict()
        voxel_semantics = voxel_semantics.long()

        # ------------------ 原始分辨率 loss ------------------
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)

            voxel_semantics_flat = voxel_semantics.reshape(-1)
            preds_flat = preds.reshape(-1, self.num_classes)
            mask_camera_flat = mask_camera.reshape(-1)

            num_total_samples = mask_camera_flat.sum()
            loss_occ = self.loss_occ(preds_flat, voxel_semantics_flat, mask_camera_flat, avg_factor=num_total_samples)
        else:
            voxel_semantics_flat = voxel_semantics.reshape(-1)
            preds_flat = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds_flat, voxel_semantics_flat)

        loss_['loss_occ'] = loss_occ

        # ------------------ 降采样版本 loss ------------------

        # Preds: [B, X, Y, Z, C] -> [B, C, X, Y, Z]
        preds_down = preds.permute(0, 4, 1, 2, 3)
        preds_down = F.avg_pool3d(preds_down.float(), kernel_size=4, stride=4)  # [B, C, X/4, Y/4, Z/4]
        preds_down = preds_down.permute(0, 2, 3, 4, 1).contiguous()  # [B, X/4, Y/4, Z/4, C]

        # voxel_semantics: [B, X, Y, Z] → [B, 1, X, Y, Z] → avg_pool → [B, X/4, Y/4, Z/4]
        voxel_semantics_down = voxel_semantics.unsqueeze(1).float()
        voxel_semantics_down = F.avg_pool3d(voxel_semantics_down, kernel_size=4, stride=4)
        voxel_semantics_down = voxel_semantics_down.squeeze(1).long()

        # mask_camera: same as above
        mask_camera_down = mask_camera.unsqueeze(1).float()
        mask_camera_down = F.avg_pool3d(mask_camera_down, kernel_size=4, stride=4)
        mask_camera_down = (mask_camera_down > 0.5).to(torch.int32).squeeze(1)  # binarize back

        # Flatten
        voxel_semantics_down = voxel_semantics_down.reshape(-1)
        preds_down = preds_down.reshape(-1, self.num_classes)
        mask_camera_down = mask_camera_down.reshape(-1)

        # Loss on downsampled version
        if self.use_mask:
            num_total_samples_down = mask_camera_down.sum()
            loss_occ_down = self.loss_occ(preds_down, voxel_semantics_down, mask_camera_down, avg_factor=num_total_samples_down)
        else:
            loss_occ_down = self.loss_occ(preds_down, voxel_semantics_down)

        # loss_['loss_occ_down'] = loss_occ_down * 0.01

        # import ipdb; ipdb.set_trace()

        return loss_


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
        # print(type(img_inputs), type(img_metas), type(points))  
        # img_inputs: list, len = 7 (history and current, multiview)
        # print(len(img_inputs), img_inputs[0].shape)
        '''
        points: None
        img_inputs: list
        img_metas: list of dicts, dicts of file metadata  少了一个时间戳，不知道有无影响
        '''
        """
        [ok] comment useless losses
        [ok] rewrite agent prediction logic
        """
        radar = kwargs.get('radar', None)
        B = len(radar)
        # import ipdb; ipdb.set_trace()
        vision_only = True
        device = img.device
        if vision_only:
            fixed_tensor = torch.load('/data/bevperception/data/nuscenes/fixed_tensor.pt', map_location=device)
            fixed_tensor = fixed_tensor.to(device)
            radar = [fixed_tensor for _ in range(B)]
            kwargs['radar'] = radar  # cover the gt radar with fixed radar

        if self.ret_2d_feat:  # False
            img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats, feat_2d = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas, **kwargs)
            feat_2d_mf = self.neck_det(feat_2d) + 'cached feat_2d'
        else:  # 走的这边 True
            # import ipdb; ipdb.set_trace()
            img_feats, _, depth, _, radar_feats, _ = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']  # √
        
        # img_feats, depth, _ = self.extract_img_feat(img, img_metas, **kwargs)
        # import ipdb;ipdb.set_trace()
        # print(f"depth min: {depth.min()}, max: {depth.max()}")

        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)  # not for now 
        losses['loss_depth'] = loss_depth  # not for now 
        # import ipdb; ipdb.set_trace()
        # print(img_feats[0].shape, radar_feats[0].shape)
        # print(bev_feat_list[0].shape, prev_radar_feats[0].shape)
        # radar_feats_up = torch.nn.functional.interpolate(radar_feats[0], scale_factor=2, mode='bilinear')
        fusion_feats = self.reduc_conv(torch.cat((img_feats[0], radar_feats[0]), dim=1))

        occ_pred = self.final_conv(fusion_feats).permute(0, 4, 3, 2, 1)
        # import ipdb; ipdb.set_trace()
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17

        # 先不算 occ 的 loss，就按之前的 infer
        # loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)  # not for now 
        # import ipdb; ipdb.set_trace()
        # losses.update(loss_occ)  # not for now

        if not self.ret_2d_feat:
            feat_2d_mf = self.extract_feat_mf(img, img_metas)
        # import ipdb; ipdb.set_trace()
        for i in range(len(img_metas)):
            img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i]
            img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]

        det_outs, bbox_feats = self.pts_bbox_head(feat_2d_mf, img_metas, **kwargs)

        loss_inputs = [gt_bboxes_3d, gt_labels_3d, det_outs]
        loss_det = self.pts_bbox_head.loss(*loss_inputs)  # not for now
        # import ipdb; ipdb.set_trace()
        losses.update(loss_det)  # not for now

        # print("shape of bbox_feats: ", bbox_feats.shape)
        # import ipdb;ipdb.set_trace()
        # we have fusion_feats, occ_pred, bbox_feats and det_outs, we can use them as the input of the motion planner.
        # ---
        planning = True
        if planning:
            # 如果存在：kwargs['ego_his_trajs']
            if 'ego_his_trajs' in kwargs:
                ego_his_trajs = kwargs['ego_his_trajs']  # [B, T_his=2, 2]
            else:
                # when we don't have ego history, we use some other info to replace it.
                ego_vel = kwargs['ego_vel']; ego_acc = kwargs['ego_acc']; command = kwargs['command']
                import ipdb; ipdb.set_trace()
                ego_his_trajs = torch.zeros(1, 2, 2, device=device)
            gt_ego_fut_trajs = kwargs['ego_fut_trajs']
            # import ipdb; ipdb.set_trace()
            # gt_ego_fut_trajs = gt_ego_fut_trajs.cumsum(dim=1)  # [B, T_fut, 2]
            B = ego_his_trajs.shape[0]

            # ----------------- Detection Result Filtering -----------------
            # Extract top-k high confidence proposals from last decoder layer
            det_layer = det_outs['all_cls_scores'][-1], det_outs['all_bbox_preds'][-1]  # ([B, 900, 10], [B, 900, 10])
            max_scores, _ = det_layer[0].max(dim=-1)  # Class-agnostic confidence [B, 900]
            _, top_indices = max_scores.topk(k=200, dim=1)  # Select top 200 proposals [B=1, 200]

            # Gather corresponding features using broadcasted indices
            topk_cls_scores = det_layer[0].gather(1, top_indices.unsqueeze(-1).expand(-1, -1, 10))  # [B, 200, 10]
            cls_confidences = torch.softmax(topk_cls_scores, dim=-1)  # [B, 200, 10]
            topk_bbox_preds = det_layer[1].gather(1, top_indices.unsqueeze(-1).expand(-1, -1,
                                                                                    10))  # [B, 200, 10] # [cx, cy, w, l, cz, h, sin, cos, vx, vy]

            # ----------------- Agent Query Construction -----------------
            # 900 valid feats and topk feats
            original_bbox_feats = bbox_feats[:, -900:, :]  # [B, 900, C]
            tokk_bbox_feats = original_bbox_feats.gather(
                1,
                top_indices.unsqueeze(-1).expand(-1, -1, original_bbox_feats.size(-1))
            )  # [B, 200, 256]

            topk_agent_box = torch.cat([topk_bbox_preds, cls_confidences], dim=-1)  # [B,200,20]
            topk_agent_query = torch.cat([topk_agent_box, tokk_bbox_feats], dim=-1)  # [B, 200, 20+256=276]
            # combined_bbox_features --> combined_bbox_features_reshaped [B, 200, 256]
            combined_bbox_features_reshaped = self.bbox_proj(topk_agent_query)  # [B, 200, 256]

            # ----------------- Ego Query Construction and Concatenation -----------------
            flatten_trajs = ego_his_trajs.reshape(B, -1)  # emb
            ego_his_feats = self.traj_pre_project(flatten_trajs).unsqueeze(1)  # [B, 1, 256]
            instance_query = torch.cat([ego_his_feats, combined_bbox_features_reshaped], dim=1)  # [B, 201, 256]
            instance_query = instance_query.permute(1, 0, 2)  # [L=201, B, C]

            # import ipdb; ipdb.set_trace()
            # TODO: adding positional encoding to instance_query (agent) with the bbox information
            # TODO: adding positional encoding to instance_query (ego) with similar bbox information
            # ------------------ Positional Encoding Construction and Concatenation ------------------
            agent_instance_pos = topk_bbox_preds
            # ego (Renault Zoe) size: [4.084, 1.730, 1.562]
            # Ego Instance Positional Encoding
            ego_instance_pos = torch.zeros(B, 1, 10, device=topk_bbox_preds.device)  # 初始化为 0
            ego_instance_pos[:, :, :3] = torch.tensor([0.0, 0.0, 0.0], device=topk_bbox_preds.device)  # 位置 (0,0,0)
            ego_instance_pos[:, :, 3:6] = self.ego_size  # 车辆尺寸 [w, l, h]
            ego_instance_pos[:, :, 6] = 0.0  # 朝向角度
            ego_instance_pos[:, :, 7] = 1.0
            ego_instance_pos[:, :, 8:] = self.ego_volocity_params.expand(B, -1, -1)  # 速度参数，TODO: 改成IMU获取
            # 拼接 Agent 和 Ego 的位置编码
            instance_pos = torch.cat([agent_instance_pos, ego_instance_pos], dim=1).permute(1, 0, 2)  # [201, B, 10]
            # import ipdb; ipdb.set_trace()

            # 线性投影到 256 维（加的话必须维数一致）
            instance_pos_emb = self.positional_encoding_proj(instance_pos)  # [201, B, 256]

            # ==================== Feature Fusion & Dimension Adjustment ====================
            # Adjust dimensions for fusion features and occupancy prediction
            fusion_feats_reshaped = fusion_feats.permute(0, 4, 3, 2, 1)  # [B, X, Y, Z, C1=32]
            occ_pred_reshaped = occ_pred  # Preserve original shape [B, X, Y, Z, C2=18]
            # Concatenate along channel dimension (C1 + C2 = 50)
            combined = torch.cat([fusion_feats_reshaped, occ_pred_reshaped], dim=-1)  # [B, 200, 200, 16, 50]

            # ==================== 3D Average Pooling for Downsampling ====================
            # Reorder dimensions to [B, C, X, Y, Z] for PyTorch AvgPool3d input
            combined_permuted = combined.permute(0, 4, 1, 2, 3)  # [B, 50, 200, 200, 16]

            # Apply 3D MAX pooling with kernel_size=4 on all spatial dimensions
            max_pool_3d = nn.MaxPool3d(kernel_size=4, stride=4)  # Reduces each dimension by 4x[1,3]  # maxpool!!!!!!! √
            combined_down = max_pool_3d(combined_permuted)  # Output: [B, C=50, 50, 50, 4]
            combined_down_permuted = combined_down.permute(0, 2, 3, 4, 1)  # [B, 50, 50, 4, 50]

            # ==================== Flattening & Concatenate Positional Encoding ================
            B, X, Y, Z, C = combined_down_permuted.shape  # [B, 50, 50, 4, 50]
            maxpooled_flat = combined_down_permuted.reshape(B, -1, C).permute(1, 0, 2)  # [10,000, B, 50]
            # 生成3D网格坐标
            grid_x = torch.arange(X, device=combined_down.device).view(-1, 1, 1).expand(X, Y, Z)
            grid_y = torch.arange(Y, device=combined_down.device).view(1, -1, 1).expand(X, Y, Z)
            grid_z = torch.arange(Z, device=combined_down.device).view(1, 1, -1).expand(X, Y, Z)
            # 将坐标展平
            pos_encoding = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)  # [10,000, 3]
            pos_encoding = pos_encoding.unsqueeze(1).expand(-1, B, -1)  # [10,000, B, 3]
            occ_flattened = torch.cat([maxpooled_flat, pos_encoding], dim=-1)  # [10,000, B, 53]
            # Project features to target dimension
            # import ipdb; ipdb.set_trace()
            occ_pred_proj = self.occ_projection(occ_flattened)  # [10,000, B, 256]

            block_num = 3

            for i in range(block_num):
                # ==================== Motion Planning Encoder ===========================
                motion_hs = self.motion_encoder(  # motion_hs: [201, B, 256]
                    query=instance_query,  # (L = 201, B, C = 256)
                    key=instance_query,  # (L = 201, B, C = 256)
                    value=instance_query,
                    query_pos=instance_pos_emb,  # <--INSTANCE 10 [201, B, 256]
                    key_pos=instance_pos_emb  # <--INSTANCE 10 [201, B, 256]
                )
                # ==================== Motion Planning Decoder ========================
                traj_embeds = self.motion_decoder(  # EgoFutureTransformerDecoder --> (201, B, 256)
                    query=motion_hs,  # [201, B, 256]
                    key=occ_pred_proj,  # [10,000, B, 256]
                    value=occ_pred_proj,  # [10,000, B, 256]
                    query_pos=instance_pos_emb,  # <--INSTANCE 10 [201, B, 256]
                    key_pos=None  # concat with k, v already.
                )
                instance_query = motion_hs
                # ----- check middle results -----
                process_supervision = True
                if process_supervision:
                    if i == 0:
                        traj_embeds_0 = traj_embeds
                    elif i == 1:
                        traj_embeds_1 = traj_embeds

            # import ipdb; ipdb.set_trace()
            def reshape_traj(tensor):
                B, L, _ = tensor.shape
                # 将12维拆分为6个时间步，每个时间步含(x,y)坐标
                return tensor.view(B, L, 6, 2)  # 12 = 6 * 2

            # ------ final output processing ------
            traj_embeds = traj_embeds.permute(1, 0, 2)  # [B, 201, 256]
            traj_coords = self.traj_post_project(traj_embeds)  # [B, 201, 12]  --> MLP!!!! √
            agent_traj_output = traj_coords[:, 1:, :]  # [B, 200, 12]  --> loss
            ego_traj_output = traj_coords[:, :1, :]  # [B, 1, 12]

            agent_traj_output = reshape_traj(agent_traj_output)  # [B, 200, 6, 2]
            ego_traj_output = reshape_traj(ego_traj_output)  # [B, 1, 6, 2]
            ego_traj_output = ego_traj_output.squeeze(dim=1)  # [B, 6, 2]

            gt_agent_fut_trajs = kwargs['gt_fut_trajs_abs']  # [B, 200, T_fut, 2]

            # plan with Transformer
            # import ipdb; ipdb.set_trace()
            
            # gru_input = {
            #     'traj_embed': traj_embeds[:, :1, :], # [B, 1, 256]
            #     'his_embed': ego_his_feats, # [B, 1, 256]
            # }
            # gru_input = torch.cat([gru_input['traj_embed'], gru_input['his_embed']], dim=-1)  # [B, 1, 512]

            # out, _ = self.traj_gru(gru_input)  # [B, 1, 128]
            # out = out.squeeze(1)        # [B, 128]

            # gru_pred = self.traj_pred_head(out)  # [B, 12]
            # gru_pred = gru_pred.view(-1, 6, 2)       # [B, 6, 2]

            # import ipdb; ipdb.set_trace()
            # ==================== Motion Losses ====================
            loss_motion = self.motion_loss(ego_traj_output, gt_ego_fut_trajs)
            # gru_motion_loss = self.motion_loss(gru_pred, gt_ego_fut_trajs)  # [B, 6, 2] vs [B, 200, T_fut, 2]

            # 计算 loss（L2 损失）
            agent_motion_loss = self.agent_motion_loss(agent_traj_output, gt_agent_fut_trajs)

            losses['motion_loss'] = loss_motion  # not for now
            # losses['gru_motion_loss'] = gru_motion_loss
            losses['agent_motion_loss'] = agent_motion_loss  # not for now

        return losses  # loss is a dictionary containing all the losses, later for back-prop altogether.

    def simple_test(self,  # simple_test is for both perception and planning, planning can be optional though
                    points,
                    img_metas,
                    img_input=None,
                    gt_masks_bev=None,
                    gt_bboxes_3d=None,
                    gt_labels_3d=None,
                    rescale=False,
                    radar=None,
                    img=None,
                    return_planning_metric=True,
                    **kwargs):
        """Test function without augmentation."""
        img = [img] if img is None else img
        img = img[0]        

        if self.ret_2d_feat:
            img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats, feat_2d = self.extract_feat(
                points, img=img_input, img_metas=img_metas, radar=radar[0], **kwargs)
            feat_2d_mf = self.neck_det(feat_2d) + 'cached self.neck_det(feat_2d)'
        else:
            img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats = self.extract_feat(
                points, img=img_input, img_metas=img_metas, radar=radar[0], **kwargs)
        # … 前半部分保持不变，计算 fusion_feats, occ_pred, feat_2d_mf, outs, bbox_feats …
        fusion_feats = self.reduc_conv(torch.cat((img_feats[0], radar_feats[0]), dim=1))
        occ_pred = self.final_conv(fusion_feats).permute(0, 4, 3, 2, 1)
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        feat_2d_mf = self.extract_feat_mf(img, img_metas)
        det_outs, bbox_feats = self.pts_bbox_head(feat_2d_mf, img_metas, **kwargs)
        bbox_list = self.pts_bbox_head.get_bboxes(det_outs, img_metas[0], rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        # 轨迹推理
        ego_his_trajs = kwargs['ego_his_trajs']  # [B, T_his, 2]
        ego_his_trajs = ego_his_trajs[0]
        traj_res = self.forward_test_traj(  # --> dict keys: 'ego', 'agent'
            fusion_feats=fusion_feats,
            occ_pred=occ_pred,
            det_outs=det_outs,
            bbox_feats=bbox_feats,
            ego_his_trajs=ego_his_trajs,
            topk=200  # 根据需要选取 top-k（agent）
        )

        bbox_list = self.pts_bbox_head.get_bboxes(det_outs, img_metas[0], rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        res_dict = {  # perception only --> if eval perception
            'pts_bbox': bbox_results[0],
            'pts_occ': occ_res
        }
        # import ipdb; ipdb.set_trace()
        ego_fut_pred = traj_res['ego']
        # gru_ego_fut_pred = traj_res['ego_gru']
        # Pred
        ego_fut_pred = torch.cumsum(ego_fut_pred, dim=1)  # still [B, T_fut, 2]
        # gru_ego_fut_pred = torch.cumsum(gru_ego_fut_pred, dim=1)
        ego_fut_trajs = kwargs['ego_fut_trajs']
        ego_fut_trajs = ego_fut_trajs[0]
        # GT
        ego_fut_trajs = torch.cumsum(ego_fut_trajs, dim=1)  # still [B, T_fut, 2]

        # import ipdb; ipdb.set_trace()
        gt_agent_feats = kwargs['gt_attr_labels'][0]
        gt_bbox = gt_bboxes_3d[0][0]
        gt_boxes = kwargs['gt_boxes'][0].squeeze(0)
        # import ipdb; ipdb.set_trace()
        fut_valid_flag = kwargs['fut_valid_flag']
        # if gt_bbox.tensor.shape[0] != gt_agent_feats[0].shape[0]:
        #     import ipdb; ipdb.set_trace()
        # with these results we calculate the `metric_dict_planner_stp3`
        metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
            # this returns a dict that includes ADE, FDE (at 1s, 2s, and 3s, 6 values)
            # pred_ego_fut_trajs=gru_ego_fut_pred,
            pred_ego_fut_trajs=ego_fut_pred,
            gt_ego_fut_trajs=ego_fut_trajs,
            gt_agent_boxes=gt_boxes,
            gt_agent_feats=gt_agent_feats,
            fut_valid_flag=fut_valid_flag
        )
        # import ipdb; ipdb.set_trace()
        # print("metric_dict_planner_stp3: ", metric_dict_planner_stp3)
        if return_planning_metric:
            res_dict.update(metric_dict_planner_stp3)
            return [res_dict]

        return [res_dict]

    def forward_test_traj(self,
                          # this includes agents and ego's trajectories. So this is where we can get the traj we want for eval
                          fusion_feats,
                          occ_pred,
                          det_outs,
                          bbox_feats,
                          ego_his_trajs,
                          topk: int = 200):
        """
        推理时的轨迹预测，复用训练时 forward_train 的流程。
        fusion_feats: [B, C1, X, Y, Z]
        occ_pred:     [B, X, Y, Z, C2]
        det_outs:     dict 包含 'all_cls_scores', 'all_bbox_preds'
        bbox_feats:   [B, N, C]
        ego_his_trajs:[B, T_his, 2]
        """
        # import ipdb; ipdb.set_trace()
        B = 1
        # ----------------- Detection Result Filtering -----------------
        # Extract top-k high confidence proposals from last decoder layer
        det_layer = det_outs['all_cls_scores'][-1], det_outs['all_bbox_preds'][-1]  # ([B, 900, 10], [B, 900, 10])
        max_scores, _ = det_layer[0].max(dim=-1)  # Class-agnostic confidence [B, 900]
        _, top_indices = max_scores.topk(k=200, dim=1)  # Select top 200 proposals [B=1, 200]

        # Gather corresponding features using broadcasted indices
        topk_cls_scores = det_layer[0].gather(1, top_indices.unsqueeze(-1).expand(-1, -1, 10))  # [B, 200, 10]
        cls_confidences = torch.softmax(topk_cls_scores, dim=-1)  # [B, 200, 10]
        topk_bbox_preds = det_layer[1].gather(1, top_indices.unsqueeze(-1).expand(-1, -1,
                                                                                  10))  # [B, 200, 10] # [cx, cy, w, l, cz, h, sin, cos, vx, vy]

        # ----------------- Agent Query Construction -----------------
        # 900 valid feats and topk feats
        original_bbox_feats = bbox_feats[:, -900:, :]  # [B, 900, C]
        tokk_bbox_feats = original_bbox_feats.gather(
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, original_bbox_feats.size(-1))
        )  # [B, 200, 256]

        topk_agent_box = torch.cat([topk_bbox_preds, cls_confidences], dim=-1)  # [B,200,20]
        topk_agent_query = torch.cat([topk_agent_box, tokk_bbox_feats], dim=-1)  # [B, 200, 20+256=276]
        # combined_bbox_features --> combined_bbox_features_reshaped [B, 200, 256]
        combined_bbox_features_reshaped = self.bbox_proj(topk_agent_query)  # [B, 200, 256]

        # ----------------- Ego Query Construction and Concatenation -----------------
        flatten_trajs = ego_his_trajs.reshape(B, -1)  # emb
        ego_his_feats = self.traj_pre_project(flatten_trajs).unsqueeze(1)  # [B, 1, 256]
        instance_query = torch.cat([ego_his_feats, combined_bbox_features_reshaped], dim=1)  # [B, 201, 256]
        instance_query = instance_query.permute(1, 0, 2)  # [L=201, B, C]

        # import ipdb; ipdb.set_trace()
        # TODO: adding positional encoding to instance_query (agent) with the bbox information
        # TODO: adding positional encoding to instance_query (ego) with similar bbox information
        # ------------------ Positional Encoding Construction and Concatenation ------------------
        agent_instance_pos = topk_bbox_preds
        # ego (Renault Zoe) size: [4.084, 1.730, 1.562]
        # Ego Instance Positional Encoding
        ego_instance_pos = torch.zeros(B, 1, 10, device=topk_bbox_preds.device)  # 初始化为 0
        ego_instance_pos[:, :, :3] = torch.tensor([0.0, 0.0, 0.0], device=topk_bbox_preds.device)  # 位置 (0,0,0)
        ego_instance_pos[:, :, 3:6] = self.ego_size  # 车辆尺寸 [w, l, h]
        ego_instance_pos[:, :, 6] = 0.0  # 朝向角度
        ego_instance_pos[:, :, 7] = 1.0
        ego_instance_pos[:, :, 8:] = self.ego_volocity_params.expand(B, -1, -1)  # 速度参数，TODO: 改成IMU获取
        # 拼接 Agent 和 Ego 的位置编码
        instance_pos = torch.cat([agent_instance_pos, ego_instance_pos], dim=1).permute(1, 0, 2)  # [201, B, 10]
        # import ipdb; ipdb.set_trace()

        # 线性投影到 256 维（加的话必须维数一致）
        instance_pos_emb = self.positional_encoding_proj(instance_pos)  # [201, B, 256]

        # ==================== Feature Fusion & Dimension Adjustment ====================
        # Adjust dimensions for fusion features and occupancy prediction
        fusion_feats_reshaped = fusion_feats.permute(0, 4, 3, 2, 1)  # [B, X, Y, Z, C1=32]
        occ_pred_reshaped = occ_pred  # Preserve original shape [B, X, Y, Z, C2=18]
        # import ipdb; ipdb.set_trace()
        # Concatenate along channel dimension (C1 + C2 = 50)
        combined = torch.cat([fusion_feats_reshaped, occ_pred_reshaped], dim=-1)  # [B, 200, 200, 16, 50]

        # ==================== 3D Average Pooling for Downsampling ====================
        # Reorder dimensions to [B, C, X, Y, Z] for PyTorch AvgPool3d input
        combined_permuted = combined.permute(0, 4, 1, 2, 3)  # [B, 50, 200, 200, 16]

        # Apply 3D MAX pooling with kernel_size=4 on all spatial dimensions
        max_pool_3d = nn.MaxPool3d(kernel_size=4, stride=4)  # Reduces each dimension by 4x[1,3]  # maxpool!!!!!!! √
        combined_down = max_pool_3d(combined_permuted)  # Output: [B, C=50, 50, 50, 4]
        combined_down_permuted = combined_down.permute(0, 2, 3, 4, 1)  # [B, 50, 50, 4, 50]

        # ==================== Flattening & Concatenate Positional Encoding ================
        B, X, Y, Z, C = combined_down_permuted.shape  # [B, 50, 50, 4, 50]
        maxpooled_flat = combined_down_permuted.reshape(B, -1, C).permute(1, 0, 2)  # [10,000, B, 50]
        # 生成3D网格坐标
        grid_x = torch.arange(X, device=combined_down.device).view(-1, 1, 1).expand(X, Y, Z)
        grid_y = torch.arange(Y, device=combined_down.device).view(1, -1, 1).expand(X, Y, Z)
        grid_z = torch.arange(Z, device=combined_down.device).view(1, 1, -1).expand(X, Y, Z)
        # 将坐标展平
        pos_encoding = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)  # [10,000, 3]
        pos_encoding = pos_encoding.unsqueeze(1).expand(-1, B, -1)  # [10,000, B, 3]
        occ_flattened = torch.cat([maxpooled_flat, pos_encoding], dim=-1)  # [10,000, B, 53]
        # Project features to target dimension
        # import ipdb; ipdb.set_trace()
        occ_pred_proj = self.occ_projection(occ_flattened)  # [10,000, B, 256]

        block_num = 3

        for i in range(block_num):
            # ==================== Motion Planning Encoder ===========================
            motion_hs = self.motion_encoder(  # motion_hs: [201, B, 256]
                query=instance_query,  # (L = 201, B, C = 256)
                key=instance_query,  # (L = 201, B, C = 256)
                value=instance_query,
                query_pos=instance_pos_emb,  # <--INSTANCE 10 [201, B, 256]
                key_pos=instance_pos_emb  # <--INSTANCE 10 [201, B, 256]
            )
            # ==================== Motion Planning Decoder ========================
            traj_embeds = self.motion_decoder(  # EgoFutureTransformerDecoder --> (201, B, 256)
                query=motion_hs,  # [201, B, 256]
                key=occ_pred_proj,  # [10,000, B, 256]
                value=occ_pred_proj,  # [10,000, B, 256]
                query_pos=instance_pos_emb,  # <--INSTANCE 10 [201, B, 256]
                key_pos=None  # concat with k, v already.
            )
            instance_query = motion_hs

        traj_embeds = traj_embeds.permute(1, 0, 2)  # [B, 201, 256]
        traj_coords = self.traj_post_project(traj_embeds)  # [B, 201, 12]  --> MLP!!!! √
        agent_traj_output = traj_coords[:, 1:, :]  # [B, 200, 12]  --> loss
        ego_traj_output = traj_coords[:, :1, :]  # [B, 1, 12]

        # import ipdb; ipdb.set_trace()
        def reshape_traj(tensor):
            B, L, _ = tensor.shape
            # 将12维拆分为6个时间步，每个时间步含(x,y)坐标
            return tensor.view(B, L, 6, 2)  # 12 = 6 * 2

        agent_traj_output = reshape_traj(agent_traj_output)  # [B, 200, 6, 2]
        ego_traj_output = reshape_traj(ego_traj_output)  # [B, 1, 6, 2]
        ego_traj_output = ego_traj_output.squeeze(dim=1)  # [B, 6, 2]


        # gru_input = {
        #     'traj_embed': traj_embeds[:, :1, :], # [B, 1, 256]
        #     'his_embed': ego_his_feats, # [B, 1, 256]
        # }
        # gru_input = torch.cat([gru_input['traj_embed'], gru_input['his_embed']], dim=-1)  # [B, 1, 512]

        # out, _ = self.traj_gru(gru_input)  # [B, 1, 128]
        # out = out.squeeze(1)        # [B, 128]

        # gru_pred = self.traj_pred_head(out)  # [B, 12]
        # gru_pred = gru_pred.view(-1, 6, 2)       # [B, 6, 2]

        return {
            'ego': ego_traj_output,
            # 'ego_gru': gru_pred,  # [B, 6, 2]
            'agents': agent_traj_output,
        }


    def compute_planner_metric_stp3(
            self,
            pred_ego_fut_trajs,
            gt_ego_fut_trajs,
            gt_agent_boxes,
            gt_agent_feats,
            fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3."""
        metric_dict = {'fut_valid_flag': fut_valid_flag}
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        occupancy_for_uniad, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(occupancy_for_uniad, pedestrian)

        for i in range(future_second):
            if fut_valid_flag[0]:
                cur_time = (i + 1) * 2
                traj_L2_ade = self.planning_metric.compute_L2_ade(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                traj_L2_fde = self.planning_metric.compute_L2_fde(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                _, obj_coll_uniad = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy_for_uniad)
                _, obj_coll_vad = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                # VAD&STP3 take average L2 and Col, UniAD takes final L2 and Col
                metric_dict['plan_L2_vad_{}s'.format(i + 1)] = traj_L2_ade
                metric_dict['plan_L2_uniad_{}s'.format(i + 1)] = traj_L2_fde
                metric_dict['plan_col_vad_{}s'.format(i + 1)] = obj_coll_vad.mean().item()
                metric_dict['plan_col_uniad_{}s'.format(i + 1)] = obj_coll_uniad[-1].item()
            else:
                # VAD&STP3 set these case to 0, UniAD skips these case
                metric_dict['plan_L2_vad_{}s'.format(i + 1)] = 0.0
                metric_dict['plan_L2_uniad_{}s'.format(i + 1)] = -1.0
                metric_dict['plan_col_vad_{}s'.format(i + 1)] = 0.0
                metric_dict['plan_col_uniad_{}s'.format(i + 1)] = -1.0

        return metric_dict

    def motion_loss(self, traj_output, gt_ego_fut_trajs, loss_type='l1'):
        """
        Compute the loss between predicted and ground truth trajectories.

        Args:
            traj_output (Tensor): Predicted trajectory, shape [B, T, 2].
            gt_ego_fut_trajs (Tensor): Ground truth trajectory, shape [B, T, 2].
            loss_type (str): Type of loss function ('l1' or 'l2'). Default is 'l1'.

        Returns:
            loss_motion (Tensor): Loss value.
        """
        # Ensure the shapes match
        assert traj_output.shape == gt_ego_fut_trajs.shape, \
            f"Shape mismatch: traj_output {traj_output.shape}, gt_ego_fut_trajs {gt_ego_fut_trajs.shape}"

        # Select the loss function
        if loss_type == 'l1':
            loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        # Compute the loss
        loss_motion = loss_fn(traj_output, gt_ego_fut_trajs)
        return loss_motion


    def agent_motion_loss(self, pred_trajs: torch.Tensor, gt_trajs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_trajs: Tensor of shape [B, N=200, T, 2] — predicted agent trajectories
            gt_trajs:   Tensor of shape [B, N=200, T, 2] — ground-truth trajectories (padded)

        Returns:
            Scalar loss averaged over valid ground-truth agents
        """
        import torch
        from scipy.optimize import linear_sum_assignment
        pred_trajs = pred_trajs.float()
        gt_trajs = gt_trajs.float()
        device = pred_trajs.device
        B, N, T, _ = pred_trajs.shape
        total_loss = 0.0
        num_valid = 0

        for b in range(B):
            pred = pred_trajs[b]  # [N, T, 2]
            gt = gt_trajs[b]      # [N, T, 2]
            
            # 有效 agent 掩码：[N]
            valid_mask = (gt.abs().sum(dim=-1).sum(dim=-1) > 0)  # [N]
            gt_valid = gt[valid_mask]  # [A, T, 2]
            A = gt_valid.shape[0]
            if A == 0:
                continue

            # Flatten 后计算 pairwise cost matrix
            gt_flat = gt_valid.view(A, -1)  # [A, T*2]
            pred_flat = pred.view(N, -1)    # [N, T*2]
            cost_matrix = torch.cdist(gt_flat, pred_flat, p=2)  # [A, N]
            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

            # 匹配后计算 L2 loss
            matched_pred = pred[col_ind]        # [A, T, 2]
            matched_gt = gt_valid[row_ind]      # [A, T, 2]
            # loss = ((matched_pred - matched_gt) ** 2).sum(dim=-1).mean(dim=-1).sum()  # scalar
            loss = self.motion_loss(matched_pred, matched_gt)
            total_loss += loss
            num_valid += A

        if num_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss / num_valid


    def plan_col_loss(
        self,
        pred,
        target,
        agent_fut_preds,
        x_dis_thresh=1.5,
        y_dis_thresh=3.0,
        dis_thresh=3.0
    ):
        """Planning ego-agent collsion constraint.

        Args:
            pred (torch.Tensor): ego_fut_preds, [B, fut_ts, 2].
            target (torch.Tensor): agent_preds, [B, num_agent, 2].  # the centre of the bboxes in xy-axis.
            agent_fut_preds (Tensor): [B, num_agent, fut_ts, 2].
            weight (torch.Tensor): [B, fut_ts, 2].
            x_dis_thresh (float, optional): distance threshold between ego and other agents in x-axis.
            y_dis_thresh (float, optional): distance threshold between ego and other agents in y-axis.
            dis_thresh (float, optional): distance threshold to filter distant agents.

        Returns:
            torch.Tensor: Calculated loss [B, fut_mode, fut_ts, 2]
        """
        B, A, L = agent_fut_preds.shape
        agent_fut_preds = agent_fut_preds.view(1, A, 6, 2)
        pred = pred.cumsum(dim=-2)
        agent_fut_preds = agent_fut_preds.cumsum(dim=-2)
        target = target[:, :, None, :] + agent_fut_preds  # 
        # filter distant agents from ego vehicle
        dist = torch.linalg.norm(pred[:, None, :, :] - target, dim=-1)
        dist_mask = dist > dis_thresh
        target[dist_mask] = 1e6

        # [B, num_agent, fut_ts]
        x_dist = torch.abs(pred[:, None, :, 0] - target[..., 0])
        y_dist = torch.abs(pred[:, None, :, 1] - target[..., 1])
        x_min_idxs = torch.argmin(x_dist, dim=1).tolist()
        y_min_idxs = torch.argmin(y_dist, dim=1).tolist()
        batch_idxs = [[i] for i in range(y_dist.shape[0])]
        ts_idxs = [[i for i in range(y_dist.shape[-1])] for j in range(y_dist.shape[0])]

        # [B, fut_ts]
        x_min_dist = x_dist[batch_idxs, x_min_idxs, ts_idxs]
        y_min_dist = y_dist[batch_idxs, y_min_idxs, ts_idxs]
        x_loss = x_min_dist
        safe_idx = x_loss > x_dis_thresh
        unsafe_idx = x_loss <= x_dis_thresh
        x_loss[safe_idx] = 0
        x_loss[unsafe_idx] = x_dis_thresh - x_loss[unsafe_idx]
        y_loss = y_min_dist
        safe_idx = y_loss > y_dis_thresh
        unsafe_idx = y_loss <= y_dis_thresh
        y_loss[safe_idx] = 0
        y_loss[unsafe_idx] = y_dis_thresh - y_loss[unsafe_idx]
        loss = torch.cat([x_loss.unsqueeze(-1), y_loss.unsqueeze(-1)], dim=-1)

        return loss


@DETECTORS.register_module
class HenetppRC_planner_closed(HenetppRC_planner):

    def __init__(self, neck_det=None,
                 data_aug=None,
                 pts_bbox_head=None,
                 motion_encoder=None,
                 motion_decoder=None,
                 train_cfg=None, test_cfg=None,
                 stop_prev_grad=0,
                 loss_plan_reg=None,
                 num_fut_steps=None,
                 **kwargs):

        super().__init__(neck_det=neck_det,
                         data_aug=data_aug,
                         pts_bbox_head=pts_bbox_head,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg,
                         stop_prev_grad=stop_prev_grad,
                         **kwargs)
        self.planning_metric = None

        if motion_encoder is not None:
            self.motion_encoder = builder.build_transformer(motion_encoder)
            assert self.motion_encoder is not None, "motion_encoder build failed!"

        if motion_decoder is not None:
            self.motion_decoder = builder.build_transformer(motion_decoder)
            assert self.motion_decoder is not None, "motion_decoder build failed!"

        self.ego_size = torch.tensor([4.084, 1.730, 1.562])
        self.num_fut_steps = num_fut_steps
        self.ego_volocity_params = nn.Parameter(torch.zeros(1, 1, 2))  # 让 vx, vy 可学习
        self.positional_encoding_proj = nn.Linear(10, 256)
        self.bbox_proj = nn.Sequential(
            nn.Linear(276, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.traj_pre_project = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        self.occ_projection = nn.Sequential(
            nn.Linear(53, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.traj_post_project = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12)
        )

        # self.traj_gru = nn.GRU(input_size=512, hidden_size=128, batch_first=True, bidirectional=False)

        # self.traj_pred_head = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 6 * 2)
        # )

        # freeze planning + occupancy part
        exclude_keys = [
            # planning part
            "motion_encoder", "motion_decoder", "ego_volocity_params", "positional_encoding_proj", "bbox_proj",
            "traj_pre_project", "occ_projection", "traj_post_project",
            # occ branch
            "reduc_conv", "final_conv", "predicter"
            # 如果以后要解冻 occ 分支再训练，可以把这三行删掉
        ]

        # train perception part only
        for name, param in self.named_parameters():
            if any(key in name for key in exclude_keys):
                param.requires_grad = False
            else:
                param.requires_grad = True


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,  # 形状不对（改了）
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function."""
        # import ipdb;ipdb.set_trace()
        # print(gt_labels_3d[0].dtype, gt_labels_3d[0])
        # print(type(img_inputs), type(img_metas), type(points))  
        # img_inputs: list, len = 7
        # print(len(img_inputs), img_inputs[0].shape)
        '''
        points: None
        img_inputs: list
        img_metas: list of dicts, dicts of file metadata
        '''
        """
        [ok] comment useless losses
        [ok] rewrite agent prediction logic
        """
        # import ipdb; ipdb.set_trace()
        B = len(gt_bboxes_3d)
        # import ipdb; ipdb.set_trace()
        vision_only = True
        device = img.device
        if vision_only:
            fixed_tensor = torch.load('/data/bevperception/data/nuscenes/fixed_tensor.pt', map_location=device)
            fixed_tensor = fixed_tensor.to(device)
            radar = [fixed_tensor for _ in range(B)]
            # import ipdb; ipdb.set_trace()
            kwargs['radar'] = radar  # fake radar for vision only

        bda = torch.tensor(  # fixed bda for training
            [[[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]]], device=device
        )
        img_inputs.append(bda)
        if self.ret_2d_feat:  # False
            img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats, feat_2d = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas, **kwargs)
            feat_2d_mf = self.neck_det(feat_2d) + 'cached feat_2d'
        else:  # 走的这边 True
            img_metas[0]['img_timestamp'] = [0.0] * 48
            img_feats, _, depth, _, radar_feats, _ = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        
        # img_feats, depth, _ = self.extract_img_feat(img, img_metas, **kwargs)
        # import ipdb;ipdb.set_trace()
        # print(f"depth min: {depth.min()}, max: {depth.max()}")

        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)  # not for now 
        losses['loss_depth'] = loss_depth  # not for now 
        # import ipdb; ipdb.set_trace()
        # print(img_feats[0].shape, radar_feats[0].shape)
        # print(bev_feat_list[0].shape, prev_radar_feats[0].shape)
        # radar_feats_up = torch.nn.functional.interpolate(radar_feats[0], scale_factor=2, mode='bilinear')


        # -------- train with occupancy supervision --------
        train_with_occ = False  # not for now
        if train_with_occ:
            fusion_feats = self.reduc_conv(torch.cat((img_feats[0], radar_feats[0]), dim=1))
            occ_pred = self.final_conv(fusion_feats).permute(0, 4, 3, 2, 1)
            if self.use_predicter:
                occ_pred = self.predicter(occ_pred)
            voxel_semantics = kwargs['voxel_semantics']
            mask_camera = kwargs['mask_camera']
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
            loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)  # not for now 
            losses.update(loss_occ)  # not for now

        # when we don't use occupancy, just no need to forward the occ branch

        if not self.ret_2d_feat:
            feat_2d_mf = self.extract_feat_mf(img, img_metas)
        for i in range(len(img_metas)):
            img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i]
            img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]

        det_outs, bbox_feats = self.pts_bbox_head(feat_2d_mf, img_metas, **kwargs)

        loss_inputs = [gt_bboxes_3d, gt_labels_3d, det_outs]
        loss_det = self.pts_bbox_head.loss(*loss_inputs)  # not for now
        # import ipdb; ipdb.set_trace()
        losses.update(loss_det)  # not for now
        # lidar2img = img_metas[0]['lidar2img']
        # to cpu
        # img_ori = img.cpu()  # [B, 48, 3, 384, 704]
        # save: bboxes + img
        # visualize_bboxes_from_TN(img=img, gt_bboxes_3d=gt_bboxes_3d, lidar2img=img_metas[0]['lidar2img'],fps=1)
        # import ipdb; ipdb.set_trace()

        # print("shape of bbox_feats: ", bbox_feats.shape)
        # import ipdb;ipdb.set_trace()
        # we have fusion_feats, occ_pred, bbox_feats and det_outs, we can use them as the input of the motion planner.
        # ---
        # 如果存在：kwargs['ego_his_trajs']
        planning = False  # we train 
        if planning:
            if 'ego_his_trajs' in kwargs:
                ego_his_trajs = kwargs['ego_his_trajs']  # [B, T_his=2, 2]
            else:
                # when we don't have ego history, we use some other info to replace it.
                ego_vel = kwargs['ego_vel']; ego_acc = kwargs['ego_acc']; command = kwargs['command']
                import ipdb; ipdb.set_trace()
            gt_ego_fut_trajs = kwargs['ego_fut_trajs']
            # import ipdb; ipdb.set_trace()
            # gt_ego_fut_trajs = gt_ego_fut_trajs.cumsum(dim=1)  # [B, T_fut, 2]
            B = ego_his_trajs.shape[0]

            # ----------------- Detection Result Filtering -----------------
            # Extract top-k high confidence proposals from last decoder layer
            det_layer = det_outs['all_cls_scores'][-1], det_outs['all_bbox_preds'][-1]  # ([B, 900, 10], [B, 900, 10])
            max_scores, _ = det_layer[0].max(dim=-1)  # Class-agnostic confidence [B, 900]
            _, top_indices = max_scores.topk(k=200, dim=1)  # Select top 200 proposals [B=1, 200]

            # Gather corresponding features using broadcasted indices
            topk_cls_scores = det_layer[0].gather(1, top_indices.unsqueeze(-1).expand(-1, -1, 10))  # [B, 200, 10]
            cls_confidences = torch.softmax(topk_cls_scores, dim=-1)  # [B, 200, 10]
            topk_bbox_preds = det_layer[1].gather(1, top_indices.unsqueeze(-1).expand(-1, -1,
                                                                                    10))  # [B, 200, 10] # [cx, cy, w, l, cz, h, sin, cos, vx, vy]

            # ----------------- Agent Query Construction -----------------
            # 900 valid feats and topk feats
            original_bbox_feats = bbox_feats[:, -900:, :]  # [B, 900, C]
            tokk_bbox_feats = original_bbox_feats.gather(
                1,
                top_indices.unsqueeze(-1).expand(-1, -1, original_bbox_feats.size(-1))
            )  # [B, 200, 256]

            topk_agent_box = torch.cat([topk_bbox_preds, cls_confidences], dim=-1)  # [B,200,20]
            topk_agent_query = torch.cat([topk_agent_box, tokk_bbox_feats], dim=-1)  # [B, 200, 20+256=276]
            # combined_bbox_features --> combined_bbox_features_reshaped [B, 200, 256]
            combined_bbox_features_reshaped = self.bbox_proj(topk_agent_query)  # [B, 200, 256]

            # ----------------- Ego Query Construction and Concatenation -----------------
            flatten_trajs = ego_his_trajs.reshape(B, -1)  # emb
            ego_his_feats = self.traj_pre_project(flatten_trajs).unsqueeze(1)  # [B, 1, 256]
            instance_query = torch.cat([ego_his_feats, combined_bbox_features_reshaped], dim=1)  # [B, 201, 256]
            instance_query = instance_query.permute(1, 0, 2)  # [L=201, B, C]

            # import ipdb; ipdb.set_trace()
            # TODO: adding positional encoding to instance_query (agent) with the bbox information
            # TODO: adding positional encoding to instance_query (ego) with similar bbox information
            # ------------------ Positional Encoding Construction and Concatenation ------------------
            agent_instance_pos = topk_bbox_preds
            # ego (Renault Zoe) size: [4.084, 1.730, 1.562]
            # Ego Instance Positional Encoding
            ego_instance_pos = torch.zeros(B, 1, 10, device=topk_bbox_preds.device)  # 初始化为 0
            ego_instance_pos[:, :, :3] = torch.tensor([0.0, 0.0, 0.0], device=topk_bbox_preds.device)  # 位置 (0,0,0)
            ego_instance_pos[:, :, 3:6] = self.ego_size  # 车辆尺寸 [w, l, h]
            ego_instance_pos[:, :, 6] = 0.0  # 朝向角度
            ego_instance_pos[:, :, 7] = 1.0
            ego_instance_pos[:, :, 8:] = self.ego_volocity_params.expand(B, -1, -1)  # 速度参数，TODO: 改成IMU获取
            # 拼接 Agent 和 Ego 的位置编码
            instance_pos = torch.cat([agent_instance_pos, ego_instance_pos], dim=1).permute(1, 0, 2)  # [201, B, 10]
            # import ipdb; ipdb.set_trace()

            # 线性投影到 256 维（加的话必须维数一致）
            instance_pos_emb = self.positional_encoding_proj(instance_pos)  # [201, B, 256]

            # ==================== Feature Fusion & Dimension Adjustment ====================
            # Adjust dimensions for fusion features and occupancy prediction
            fusion_feats_reshaped = fusion_feats.permute(0, 4, 3, 2, 1)  # [B, X, Y, Z, C1=32]
            occ_pred_reshaped = occ_pred  # Preserve original shape [B, X, Y, Z, C2=18]
            # Concatenate along channel dimension (C1 + C2 = 50)
            combined = torch.cat([fusion_feats_reshaped, occ_pred_reshaped], dim=-1)  # [B, 200, 200, 16, 50]

            # ==================== 3D Average Pooling for Downsampling ====================
            # Reorder dimensions to [B, C, X, Y, Z] for PyTorch AvgPool3d input
            combined_permuted = combined.permute(0, 4, 1, 2, 3)  # [B, 50, 200, 200, 16]

            # Apply 3D MAX pooling with kernel_size=4 on all spatial dimensions
            max_pool_3d = nn.MaxPool3d(kernel_size=4, stride=4)  # Reduces each dimension by 4x[1,3]  # maxpool!!!!!!! √
            combined_down = max_pool_3d(combined_permuted)  # Output: [B, C=50, 50, 50, 4]
            combined_down_permuted = combined_down.permute(0, 2, 3, 4, 1)  # [B, 50, 50, 4, 50]

            # ==================== Flattening & Concatenate Positional Encoding ================
            B, X, Y, Z, C = combined_down_permuted.shape  # [B, 50, 50, 4, 50]
            maxpooled_flat = combined_down_permuted.reshape(B, -1, C).permute(1, 0, 2)  # [10,000, B, 50]
            # 生成3D网格坐标
            grid_x = torch.arange(X, device=combined_down.device).view(-1, 1, 1).expand(X, Y, Z)
            grid_y = torch.arange(Y, device=combined_down.device).view(1, -1, 1).expand(X, Y, Z)
            grid_z = torch.arange(Z, device=combined_down.device).view(1, 1, -1).expand(X, Y, Z)
            # 将坐标展平
            pos_encoding = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)  # [10,000, 3]
            pos_encoding = pos_encoding.unsqueeze(1).expand(-1, B, -1)  # [10,000, B, 3]
            occ_flattened = torch.cat([maxpooled_flat, pos_encoding], dim=-1)  # [10,000, B, 53]
            # Project features to target dimension
            # import ipdb; ipdb.set_trace()
            occ_pred_proj = self.occ_projection(occ_flattened)  # [10,000, B, 256]

            block_num = 3

            for i in range(block_num):
                # ==================== Motion Planning Encoder ===========================
                motion_hs = self.motion_encoder(  # motion_hs: [201, B, 256]
                    query=instance_query,  # (L = 201, B, C = 256)
                    key=instance_query,  # (L = 201, B, C = 256)
                    value=instance_query,
                    query_pos=instance_pos_emb,  # <--INSTANCE 10 [201, B, 256]
                    key_pos=instance_pos_emb  # <--INSTANCE 10 [201, B, 256]
                )
                # ==================== Motion Planning Decoder ========================
                traj_embeds = self.motion_decoder(  # EgoFutureTransformerDecoder --> (201, B, 256)
                    query=motion_hs,  # [201, B, 256]
                    key=occ_pred_proj,  # [10,000, B, 256]
                    value=occ_pred_proj,  # [10,000, B, 256]
                    query_pos=instance_pos_emb,  # <--INSTANCE 10 [201, B, 256]
                    key_pos=None  # concat with k, v already.
                )
                instance_query = motion_hs
                # ----- check middle results -----
                process_supervision = True
                if process_supervision:
                    if i == 0:
                        traj_embeds_0 = traj_embeds
                    elif i == 1:
                        traj_embeds_1 = traj_embeds

            # import ipdb; ipdb.set_trace()
            def reshape_traj(tensor):
                B, L, _ = tensor.shape
                # 将12维拆分为6个时间步，每个时间步含(x,y)坐标
                return tensor.view(B, L, 6, 2)  # 12 = 6 * 2

            # ------ final output processing ------
            traj_embeds = traj_embeds.permute(1, 0, 2)  # [B, 201, 256]
            traj_coords = self.traj_post_project(traj_embeds)  # [B, 201, 12]  --> MLP!!!! √
            agent_traj_output = traj_coords[:, 1:, :]  # [B, 200, 12]  --> loss
            ego_traj_output = traj_coords[:, :1, :]  # [B, 1, 12]

            agent_traj_output = reshape_traj(agent_traj_output)  # [B, 200, 6, 2]
            ego_traj_output = reshape_traj(ego_traj_output)  # [B, 1, 6, 2]
            ego_traj_output = ego_traj_output.squeeze(dim=1)  # [B, 6, 2]

            gt_agent_fut_trajs = kwargs['gt_fut_trajs_abs']  # [B, 200, T_fut, 2]

            # plan with Transformer
            # import ipdb; ipdb.set_trace()
            
            # gru_input = {
            #     'traj_embed': traj_embeds[:, :1, :], # [B, 1, 256]
            #     'his_embed': ego_his_feats, # [B, 1, 256]
            # }
            # gru_input = torch.cat([gru_input['traj_embed'], gru_input['his_embed']], dim=-1)  # [B, 1, 512]

            # out, _ = self.traj_gru(gru_input)  # [B, 1, 128]
            # out = out.squeeze(1)        # [B, 128]

            # gru_pred = self.traj_pred_head(out)  # [B, 12]
            # gru_pred = gru_pred.view(-1, 6, 2)       # [B, 6, 2]

            # import ipdb; ipdb.set_trace()
            # ==================== Motion Losses ====================
            loss_motion = self.motion_loss(ego_traj_output, gt_ego_fut_trajs)
            # gru_motion_loss = self.motion_loss(gru_pred, gt_ego_fut_trajs)  # [B, 6, 2] vs [B, 200, T_fut, 2]

            # 计算 loss（L2 损失）
            agent_motion_loss = self.agent_motion_loss(agent_traj_output, gt_agent_fut_trajs)

            losses['motion_loss'] = loss_motion  # not for now
            # losses['gru_motion_loss'] = gru_motion_loss
            losses['agent_motion_loss'] = agent_motion_loss  # not for now

        return losses  # loss is a dictionary containing all the losses, later for back-prop altogether.

    def simple_test(self,  # simple_test is for both perception and planning, planning can be optional though
                    points,
                    img_metas,
                    img_input=None,
                    gt_masks_bev=None,
                    gt_bboxes_3d=None,
                    gt_labels_3d=None,
                    rescale=False,
                    radar=None,
                    img=None,
                    return_planning_metric=True,
                    **kwargs):
        """Test function without augmentation."""
        img = [img] if img is None else img
        img = img[0]        

        vision_only = True
        if vision_only:
            device = radar[0][0].device
            fixed_tensor = torch.load('/data/bevperception/data/nuscenes/fixed_tensor.pt', map_location=device)
            fixed_tensor = fixed_tensor.to(device)
            radar = [[fixed_tensor]]


        if self.ret_2d_feat:
            img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats, feat_2d = self.extract_feat(
                points, img=img_input, img_metas=img_metas, radar=radar[0], **kwargs)
            feat_2d_mf = self.neck_det(feat_2d) + 'cached self.neck_det(feat_2d)'
        else:
            img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats = self.extract_feat(
                points, img=img_input, img_metas=img_metas, radar=radar[0], **kwargs)
        # … 前半部分保持不变，计算 fusion_feats, occ_pred, feat_2d_mf, outs, bbox_feats …
        fusion_feats = self.reduc_conv(torch.cat((img_feats[0], radar_feats[0]), dim=1))
        occ_pred = self.final_conv(fusion_feats).permute(0, 4, 3, 2, 1)
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        feat_2d_mf = self.extract_feat_mf(img, img_metas)
        det_outs, bbox_feats = self.pts_bbox_head(feat_2d_mf, img_metas, **kwargs)
        bbox_list = self.pts_bbox_head.get_bboxes(det_outs, img_metas[0], rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        # 轨迹推理
        ego_his_trajs = kwargs['ego_his_trajs']  # [B, T_his, 2]
        ego_his_trajs = ego_his_trajs[0]
        traj_res = self.forward_test_traj(  # --> dict keys: 'ego', 'agent'
            fusion_feats=fusion_feats,
            occ_pred=occ_pred,
            det_outs=det_outs,
            bbox_feats=bbox_feats,
            ego_his_trajs=ego_his_trajs,
            topk=200  # 根据需要选取 top-k（agent）
        )

        bbox_list = self.pts_bbox_head.get_bboxes(det_outs, img_metas[0], rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        res_dict = {  # perception only --> if eval perception
            'pts_bbox': bbox_results[0],
            'pts_occ': occ_res
        }
        # import ipdb; ipdb.set_trace()
        ego_fut_pred = traj_res['ego']
        # gru_ego_fut_pred = traj_res['ego_gru']
        # Pred
        ego_fut_pred = torch.cumsum(ego_fut_pred, dim=1)  # still [B, T_fut, 2]
        # gru_ego_fut_pred = torch.cumsum(gru_ego_fut_pred, dim=1)
        ego_fut_trajs = kwargs['ego_fut_trajs']
        ego_fut_trajs = ego_fut_trajs[0]
        # GT
        ego_fut_trajs = torch.cumsum(ego_fut_trajs, dim=1)  # still [B, T_fut, 2]

        # import ipdb; ipdb.set_trace()
        gt_agent_feats = kwargs['gt_attr_labels'][0]
        gt_bbox = gt_bboxes_3d[0][0]
        gt_boxes = kwargs['gt_boxes'][0].squeeze(0)
        # import ipdb; ipdb.set_trace()
        fut_valid_flag = kwargs['fut_valid_flag']
        # if gt_bbox.tensor.shape[0] != gt_agent_feats[0].shape[0]:
        #     import ipdb; ipdb.set_trace()
        # with these results we calculate the `metric_dict_planner_stp3`
        metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
            # this returns a dict that includes ADE, FDE (at 1s, 2s, and 3s, 6 values)
            # pred_ego_fut_trajs=gru_ego_fut_pred,
            pred_ego_fut_trajs=ego_fut_pred,
            gt_ego_fut_trajs=ego_fut_trajs,
            gt_agent_boxes=gt_boxes,
            gt_agent_feats=gt_agent_feats,
            fut_valid_flag=fut_valid_flag
        )
        # import ipdb; ipdb.set_trace()
        # print("metric_dict_planner_stp3: ", metric_dict_planner_stp3)
        if return_planning_metric:
            res_dict.update(metric_dict_planner_stp3)
            return [res_dict]

        return [res_dict]

    def forward_test_traj(self,
                          # this includes agents and ego's trajectories. So this is where we can get the traj we want for eval
                          fusion_feats,
                          occ_pred,
                          det_outs,
                          bbox_feats,
                          ego_his_trajs,
                          topk: int = 200):
        """
        推理时的轨迹预测，复用训练时 forward_train 的流程。
        fusion_feats: [B, C1, X, Y, Z]
        occ_pred:     [B, X, Y, Z, C2]
        det_outs:     dict 包含 'all_cls_scores', 'all_bbox_preds'
        bbox_feats:   [B, N, C]
        ego_his_trajs:[B, T_his, 2]
        """
        # import ipdb; ipdb.set_trace()
        B = 1
        # ----------------- Detection Result Filtering -----------------
        # Extract top-k high confidence proposals from last decoder layer
        det_layer = det_outs['all_cls_scores'][-1], det_outs['all_bbox_preds'][-1]  # ([B, 900, 10], [B, 900, 10])
        max_scores, _ = det_layer[0].max(dim=-1)  # Class-agnostic confidence [B, 900]
        _, top_indices = max_scores.topk(k=200, dim=1)  # Select top 200 proposals [B=1, 200]

        # Gather corresponding features using broadcasted indices
        topk_cls_scores = det_layer[0].gather(1, top_indices.unsqueeze(-1).expand(-1, -1, 10))  # [B, 200, 10]
        cls_confidences = torch.softmax(topk_cls_scores, dim=-1)  # [B, 200, 10]
        topk_bbox_preds = det_layer[1].gather(1, top_indices.unsqueeze(-1).expand(-1, -1,
                                                                                  10))  # [B, 200, 10] # [cx, cy, w, l, cz, h, sin, cos, vx, vy]

        # ----------------- Agent Query Construction -----------------
        # 900 valid feats and topk feats
        original_bbox_feats = bbox_feats[:, -900:, :]  # [B, 900, C]
        tokk_bbox_feats = original_bbox_feats.gather(
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, original_bbox_feats.size(-1))
        )  # [B, 200, 256]

        topk_agent_box = torch.cat([topk_bbox_preds, cls_confidences], dim=-1)  # [B,200,20]
        topk_agent_query = torch.cat([topk_agent_box, tokk_bbox_feats], dim=-1)  # [B, 200, 20+256=276]
        # combined_bbox_features --> combined_bbox_features_reshaped [B, 200, 256]
        combined_bbox_features_reshaped = self.bbox_proj(topk_agent_query)  # [B, 200, 256]

        # ----------------- Ego Query Construction and Concatenation -----------------
        flatten_trajs = ego_his_trajs.reshape(B, -1)  # emb
        ego_his_feats = self.traj_pre_project(flatten_trajs).unsqueeze(1)  # [B, 1, 256]
        instance_query = torch.cat([ego_his_feats, combined_bbox_features_reshaped], dim=1)  # [B, 201, 256]
        instance_query = instance_query.permute(1, 0, 2)  # [L=201, B, C]

        # import ipdb; ipdb.set_trace()
        # TODO: adding positional encoding to instance_query (agent) with the bbox information
        # TODO: adding positional encoding to instance_query (ego) with similar bbox information
        # ------------------ Positional Encoding Construction and Concatenation ------------------
        agent_instance_pos = topk_bbox_preds
        # ego (Renault Zoe) size: [4.084, 1.730, 1.562]
        # Ego Instance Positional Encoding
        ego_instance_pos = torch.zeros(B, 1, 10, device=topk_bbox_preds.device)  # 初始化为 0
        ego_instance_pos[:, :, :3] = torch.tensor([0.0, 0.0, 0.0], device=topk_bbox_preds.device)  # 位置 (0,0,0)
        ego_instance_pos[:, :, 3:6] = self.ego_size  # 车辆尺寸 [w, l, h]
        ego_instance_pos[:, :, 6] = 0.0  # 朝向角度
        ego_instance_pos[:, :, 7] = 1.0
        ego_instance_pos[:, :, 8:] = self.ego_volocity_params.expand(B, -1, -1)  # 速度参数，TODO: 改成IMU获取
        # 拼接 Agent 和 Ego 的位置编码
        instance_pos = torch.cat([agent_instance_pos, ego_instance_pos], dim=1).permute(1, 0, 2)  # [201, B, 10]
        # import ipdb; ipdb.set_trace()

        # 线性投影到 256 维（加的话必须维数一致）
        instance_pos_emb = self.positional_encoding_proj(instance_pos)  # [201, B, 256]

        # ==================== Feature Fusion & Dimension Adjustment ====================
        # Adjust dimensions for fusion features and occupancy prediction
        fusion_feats_reshaped = fusion_feats.permute(0, 4, 3, 2, 1)  # [B, X, Y, Z, C1=32]
        occ_pred_reshaped = occ_pred  # Preserve original shape [B, X, Y, Z, C2=18]
        # import ipdb; ipdb.set_trace()
        # Concatenate along channel dimension (C1 + C2 = 50)
        combined = torch.cat([fusion_feats_reshaped, occ_pred_reshaped], dim=-1)  # [B, 200, 200, 16, 50]

        # ==================== 3D Average Pooling for Downsampling ====================
        # Reorder dimensions to [B, C, X, Y, Z] for PyTorch AvgPool3d input
        combined_permuted = combined.permute(0, 4, 1, 2, 3)  # [B, 50, 200, 200, 16]

        # Apply 3D MAX pooling with kernel_size=4 on all spatial dimensions
        max_pool_3d = nn.MaxPool3d(kernel_size=4, stride=4)  # Reduces each dimension by 4x[1,3]  # maxpool!!!!!!! √
        combined_down = max_pool_3d(combined_permuted)  # Output: [B, C=50, 50, 50, 4]
        combined_down_permuted = combined_down.permute(0, 2, 3, 4, 1)  # [B, 50, 50, 4, 50]

        # ==================== Flattening & Concatenate Positional Encoding ================
        B, X, Y, Z, C = combined_down_permuted.shape  # [B, 50, 50, 4, 50]
        maxpooled_flat = combined_down_permuted.reshape(B, -1, C).permute(1, 0, 2)  # [10,000, B, 50]
        # 生成3D网格坐标
        grid_x = torch.arange(X, device=combined_down.device).view(-1, 1, 1).expand(X, Y, Z)
        grid_y = torch.arange(Y, device=combined_down.device).view(1, -1, 1).expand(X, Y, Z)
        grid_z = torch.arange(Z, device=combined_down.device).view(1, 1, -1).expand(X, Y, Z)
        # 将坐标展平
        pos_encoding = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)  # [10,000, 3]
        pos_encoding = pos_encoding.unsqueeze(1).expand(-1, B, -1)  # [10,000, B, 3]
        occ_flattened = torch.cat([maxpooled_flat, pos_encoding], dim=-1)  # [10,000, B, 53]
        # Project features to target dimension
        # import ipdb; ipdb.set_trace()
        occ_pred_proj = self.occ_projection(occ_flattened)  # [10,000, B, 256]

        block_num = 3

        for i in range(block_num):
            # ==================== Motion Planning Encoder ===========================
            motion_hs = self.motion_encoder(  # motion_hs: [201, B, 256]
                query=instance_query,  # (L = 201, B, C = 256)
                key=instance_query,  # (L = 201, B, C = 256)
                value=instance_query,
                query_pos=instance_pos_emb,  # <--INSTANCE 10 [201, B, 256]
                key_pos=instance_pos_emb  # <--INSTANCE 10 [201, B, 256]
            )
            # ==================== Motion Planning Decoder ========================
            traj_embeds = self.motion_decoder(  # EgoFutureTransformerDecoder --> (201, B, 256)
                query=motion_hs,  # [201, B, 256]
                key=occ_pred_proj,  # [10,000, B, 256]
                value=occ_pred_proj,  # [10,000, B, 256]
                query_pos=instance_pos_emb,  # <--INSTANCE 10 [201, B, 256]
                key_pos=None  # concat with k, v already.
            )
            instance_query = motion_hs

        traj_embeds = traj_embeds.permute(1, 0, 2)  # [B, 201, 256]
        traj_coords = self.traj_post_project(traj_embeds)  # [B, 201, 12]  --> MLP!!!! √
        agent_traj_output = traj_coords[:, 1:, :]  # [B, 200, 12]  --> loss
        ego_traj_output = traj_coords[:, :1, :]  # [B, 1, 12]

        # import ipdb; ipdb.set_trace()
        def reshape_traj(tensor):
            B, L, _ = tensor.shape
            # 将12维拆分为6个时间步，每个时间步含(x,y)坐标
            return tensor.view(B, L, 6, 2)  # 12 = 6 * 2

        agent_traj_output = reshape_traj(agent_traj_output)  # [B, 200, 6, 2]
        ego_traj_output = reshape_traj(ego_traj_output)  # [B, 1, 6, 2]
        ego_traj_output = ego_traj_output.squeeze(dim=1)  # [B, 6, 2]


        # gru_input = {
        #     'traj_embed': traj_embeds[:, :1, :], # [B, 1, 256]
        #     'his_embed': ego_his_feats, # [B, 1, 256]
        # }
        # gru_input = torch.cat([gru_input['traj_embed'], gru_input['his_embed']], dim=-1)  # [B, 1, 512]

        # out, _ = self.traj_gru(gru_input)  # [B, 1, 128]
        # out = out.squeeze(1)        # [B, 128]

        # gru_pred = self.traj_pred_head(out)  # [B, 12]
        # gru_pred = gru_pred.view(-1, 6, 2)       # [B, 6, 2]

        return {
            'ego': ego_traj_output,
            # 'ego_gru': gru_pred,  # [B, 6, 2]
            'agents': agent_traj_output,
        }

    def motion_loss(self, traj_output, gt_ego_fut_trajs, loss_type='l1'):
        """
        Compute the loss between predicted and ground truth trajectories.

        Args:
            traj_output (Tensor): Predicted trajectory, shape [B, T, 2].
            gt_ego_fut_trajs (Tensor): Ground truth trajectory, shape [B, T, 2].
            loss_type (str): Type of loss function ('l1' or 'l2'). Default is 'l1'.

        Returns:
            loss_motion (Tensor): Loss value.
        """
        # Ensure the shapes match
        assert traj_output.shape == gt_ego_fut_trajs.shape, \
            f"Shape mismatch: traj_output {traj_output.shape}, gt_ego_fut_trajs {gt_ego_fut_trajs.shape}"

        # Select the loss function
        if loss_type == 'l1':
            loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        # Compute the loss
        loss_motion = loss_fn(traj_output, gt_ego_fut_trajs)
        return loss_motion

    def agent_motion_loss(self, pred_trajs: torch.Tensor, gt_trajs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_trajs: Tensor of shape [B, N=200, T, 2] — predicted agent trajectories
            gt_trajs:   Tensor of shape [B, N=200, T, 2] — ground-truth trajectories (padded)

        Returns:
            Scalar loss averaged over valid ground-truth agents
        """
        import torch
        from scipy.optimize import linear_sum_assignment
        pred_trajs = pred_trajs.float()
        gt_trajs = gt_trajs.float()
        device = pred_trajs.device
        B, N, T, _ = pred_trajs.shape
        total_loss = 0.0
        num_valid = 0

        for b in range(B):
            pred = pred_trajs[b]  # [N, T, 2]
            gt = gt_trajs[b]      # [N, T, 2]
            
            # 有效 agent 掩码：[N]
            valid_mask = (gt.abs().sum(dim=-1).sum(dim=-1) > 0)  # [N]
            gt_valid = gt[valid_mask]  # [A, T, 2]
            A = gt_valid.shape[0]
            if A == 0:
                continue

            # Flatten 后计算 pairwise cost matrix
            gt_flat = gt_valid.view(A, -1)  # [A, T*2]
            pred_flat = pred.view(N, -1)    # [N, T*2]
            cost_matrix = torch.cdist(gt_flat, pred_flat, p=2)  # [A, N]
            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

            # 匹配后计算 L2 loss
            matched_pred = pred[col_ind]        # [A, T, 2]
            matched_gt = gt_valid[row_ind]      # [A, T, 2]
            # loss = ((matched_pred - matched_gt) ** 2).sum(dim=-1).mean(dim=-1).sum()  # scalar
            loss = self.motion_loss(matched_pred, matched_gt)
            total_loss += loss
            num_valid += A

        if num_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss / num_valid


@TRANSFORMER.register_module()
class CustomTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in motion transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default: `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(CustomTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                key_padding_mask=None,
                *args,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        intermediate = []
        for lid, layer in enumerate(self.layers):
            query = layer(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                key_padding_mask=key_padding_mask,
                *args,
                **kwargs)

            if self.return_intermediate:
                intermediate.append(query)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return query
    
def visualize_bboxes_from_TN(img,
                            gt_bboxes_3d,
                            lidar2img,
                            views=6,
                            frames_per_view=8,
                            order='time_first',   # 'view_first' or 'time_first'
                            grid=(3, 2),          # rows, cols, rows*cols must == views
                            save_path="vis_output.mp4",
                            fps=2,
                            bbox_idx=0,
                            color=(0,255,0),
                            thickness=1):
    """
    Visualize 3D bboxes on multi-view frames stored as img: [1, T*N, 3, H, W]

    Args:
        img: torch.Tensor [1, T*N, 3, H, W] (0~255, uint8/float)
        gt_bboxes_3d: iterable (e.g., list) or container where gt_bboxes_3d[bbox_idx] gives LiDARInstance3DBoxes for the frame to visualize
        lidar2img: torch.Tensor [1, T*N, 4, 4] (per-image lidar->img projection)
        views: int, number of camera views (N)
        frames_per_view: int, number of frames per view (T)
        order: 'view_first' (default) if data ordered as [v0_f0, v0_f1,..., v1_f0,...],
               'time_first' if ordered as [t0_v0, t0_v1,..., t1_v0,...]
        grid: tuple(rows, cols) to place the N views into a tile (rows * cols must == views)
        save_path: output video path
        fps: video fps
        bbox_idx: index into gt_bboxes_3d to pick which frame's bboxes to draw (default 0)
        color, thickness: draw params passed to your draw function
    """
    assert img.shape[0] == 1, "Only supports batch=1"
    TN = img.shape[1]
    N = views
    T = frames_per_view
    assert TN == N * T, f"TN ({TN}) != views*frames_per_view ({N}*{T})"
    rows, cols = grid
    assert rows * cols == N, f"grid {grid} does not match views={N}"

    # bring to cpu
    imgs = img[0].detach().cpu()             # [TN, 3, H, W]
    l2i = lidar2img[0].detach().cpu()        # [TN, 4, 4]

    _, C, H, W = imgs.shape

    # reshape to [T, N, C, H, W]
    if order == 'view_first':
        # current order: for each view v: its T frames are contiguous
        imgs_tn = imgs.view(N, T, C, H, W).permute(1, 0, 2, 3, 4).contiguous()
        l2i_tn = l2i.view(N, T, 4, 4).permute(1, 0, 2, 3).contiguous()
    elif order == 'time_first':
        # current order: for each time t: N views contiguous
        imgs_tn = imgs.view(T, N, C, H, W).contiguous()
        l2i_tn = l2i.view(T, N, 4, 4).contiguous()
    else:
        raise ValueError("order must be 'view_first' or 'time_first'")

    # choose bbox (gt_bboxes_3d is indexed as in your code: gt_bboxes_3d[bbox_idx])
    bboxes3d = gt_bboxes_3d[bbox_idx]

    frames_out = []
    for t in range(T):
        view_imgs = []
        for v in range(N):
            # img for view v at time t -> H,W,3 numpy
            raw_img = imgs_tn[t, v].permute(1, 2, 0).numpy()   # H, W, C
            if raw_img.dtype != np.uint8:
                raw_img = np.clip(raw_img, 0, 255).astype(np.uint8)

            cam2img = l2i_tn[t, v].numpy()  # [4,4]

            # draw (uses your draw_lidar_bbox3d_on_img)
            vis = draw_lidar_bbox3d_on_img(
                bboxes3d,
                raw_img,
                cam2img,
                img_metas=None,
                color=color,
                thickness=thickness
            )
            view_imgs.append(vis)

        # tile views into grid
        # ensure all images same H,W
        for im in view_imgs:
            assert im.shape[0] == H and im.shape[1] == W, "Image size mismatch"

        rows_imgs = []
        for r in range(rows):
            row_slice = view_imgs[r * cols:(r + 1) * cols]
            row_cat = np.concatenate(row_slice, axis=1)   # concat horizontally
            rows_imgs.append(row_cat)
        grid_img = np.concatenate(rows_imgs, axis=0)     # concat vertically

        frames_out.append(grid_img)

    # stack to [T, H_grid, W_grid, 3] and save
    frames_arr = np.stack(frames_out, axis=0)   # uint8
    if frames_arr.dtype != np.uint8:
        frames_arr = np.clip(frames_arr, 0, 255).astype(np.uint8)

    # 只取第一帧
    first_frame = frames_arr[0]  # [H, W, C]

    # 转换为PIL图像并保存
    from PIL import Image
    first_frame_pil = Image.fromarray(first_frame.astype(np.uint8))
    first_frame_pil.save(save_path.replace('.mp4', '.png'))  # 修改文件扩展名
    print(f"Saved first frame to {save_path.replace('.mp4', '.png')}, frame_size={first_frame.shape}")

    # frames_tensor = torch.from_numpy(frames_arr)    # [T, H, W, C]
    # torchvision.io.write_video(save_path, frames_tensor, fps=fps)
    
    # print(f"Saved video to {save_path}, frames={T}, grid_size={(rows,H,cols,W)}")
