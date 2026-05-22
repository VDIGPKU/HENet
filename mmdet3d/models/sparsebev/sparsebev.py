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
from .. import builder
from collections import defaultdict
from mmdet3d.ops.csrc.wrapper import msmv_sampling, TRTMSMVSampling


@DETECTORS.register_module()
class SparseBEV(MVXTwoStageDetector):
    def __init__(self,
                 data_aug=None,
                 stop_prev_grad=0,
                 longterm_model=None,
                 **kwargs):

        super(SparseBEV, self).__init__(**kwargs)
        self.data_aug = data_aug
        self.stop_prev_grad = stop_prev_grad
        self.color_aug = GpuPhotoMetricDistortion()
        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = True
        if longterm_model is not None:
            self.longterm_model = builder.build_detector(longterm_model)

        self.memory = {}
        self.queue = queue.Queue()

    # @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img):
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_feat(self, img, img_metas):
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

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feat = cast_tensor_type(img_feat, torch.half, torch.float32)
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          **kwargs):
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
        outs = self.pts_bbox_head(pts_feats, img_metas, **kwargs)
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
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      img_lt=None,
                      img_metas_lt=None,
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

        img_feats = self.extract_feat(img, img_metas)
        if img_lt is not None:
            img_feats_lt = self.longterm_model.extract_feat(img_lt, img_metas_lt)
        else:
            img_feats_lt = None

        for i in range(len(img_metas)):
            img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i]
            img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]

        losses = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore,
                                        img_feats_lt=img_feats_lt, img_metas_lt=img_metas_lt, **kwargs)

        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def simple_test_pts(self, x, img_metas, rescale=False, **kwargs):
        outs = self.pts_bbox_head(x, img_metas, **kwargs)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas[0], rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results
    
    def simple_test(self, img_metas, img=None, rescale=False, **kwargs):
        if 'openad' in kwargs and kwargs['openad']:
            world_size = -1
        else:
            world_size = get_dist_info()[1]
        if world_size == 1:  # online
            return self.simple_test_online(img_metas, img, rescale, **kwargs)
        else:  # offline
            return self.simple_test_offline(img_metas, img, rescale, **kwargs)

    def simple_test_offline(self, img_metas, img=None, rescale=False, img_lt=None, img_metas_lt=None, **kwargs):

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        if img_lt is not None:
            img_lt = img_lt[0]
            img_metas_lt = img_metas_lt[0]
            img_feats_lt = self.longterm_model.extract_feat(img_lt, img_metas_lt)
        else:
            img_feats_lt = None

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale,
                                        img_feats_lt=img_feats_lt, img_metas_lt=img_metas_lt, **kwargs)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        # print(len(bbox_list), bbox_list[0])
        return bbox_list

    def simple_test_online(self, img_metas, img=None, rescale=False, **kwargs):
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
                img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
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
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list


@DETECTORS.register_module()
class SparseBEVTRT(SparseBEV):
    def __init__(self,
                 **kwargs):

        super(SparseBEVTRT, self).__init__(**kwargs)

    def forward(self, img=None, lidar2img=None, img_timestamp=None, len_img_filenames=None, feat_prev_1=None, feat_prev_2=None, feat_prev_3=None, feat_prev_4=None):
        len_img_filenames = int(len_img_filenames)
        # len_img_filenames 不变，onnx会把他转化为constant
        # 如果在这里修改了lidar2img和img_timestamp, onnx也会把他们转化成constant，但这是不对的，因此我们要保留他的tensor格式

        # img = torch.stack(img)
        return self.simple_test(img, lidar2img, img_timestamp, len_img_filenames, feat_prev_1, feat_prev_2, feat_prev_3, feat_prev_4)
    
    def simple_test(self, img=None, lidar2img=None, img_timestamp=None, len_img_filenames=None, feat_prev_1=None, feat_prev_2=None, feat_prev_3=None, feat_prev_4=None, rescale=False):
        self.fp16_enabled = False
        # img_metas[0] is a key, img.shape = [1, 6, 3, 256, 704])
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
            if i == 0: # 可以在后续验证一下是不是这样 这里默认第一个是最新的frame
                # extract feature and put into memory
                # img_feats_curr is a list with 4 tensors
                img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
                # img_feats_curr_ret = [img_feats_curr[0].clone().detach(), img_feats_curr[1].clone().detach(), img_feats_curr[2].clone().detach(), img_feats_curr[3].clone().detach()]
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
        cls_scores, bbox_preds = self.pts_bbox_head.forward_trt(img_feats, img_metas)  
        return cls_scores, bbox_preds, img_feats_curr_ret[0], img_feats_curr_ret[1], img_feats_curr_ret[2], img_feats_curr_ret[3]

    def simple_test_debug(self, img=None, lidar2img=None, img_timestamp=None, len_img_filenames=None, feat_prev_1=None, feat_prev_2=None, feat_prev_3=None, feat_prev_4=None, rescale=False):

        self.fp16_enabled = False
        # img_metas[0] is a key, img.shape = [1, 6, 3, 256, 704])
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
            if i == 0: # 可以在后续验证一下是不是这样 这里默认第一个是最新的frame
                # extract feature and put into memory
                # img_feats_curr is a list with 4 tensors
                img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
                # img_feats_curr_ret = [img_feats_curr[0].clone().detach(), img_feats_curr[1].clone().detach(), img_feats_curr[2].clone().detach(), img_feats_curr[3].clone().detach()]
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
        img_feats = cast_tensor_type(img_feats, torch.half, torch.float32)
        # run detector
        sampled_feat = self.pts_bbox_head.forward_trt_debug(img_feats, img_metas)  
        return sampled_feat, img_feats_curr_ret[0], img_feats_curr_ret[1], img_feats_curr_ret[2], img_feats_curr_ret[3]
        # return cls_scores, bbox_preds, img_feats_curr_ret[0], img_feats_curr_ret[1], img_feats_curr_ret[2], img_feats_curr_ret[3]


@DETECTORS.register_module()
class SparseBEVDEBUGTRT(SparseBEV):
    def __init__(self,
                 **kwargs):

        super(SparseBEVDEBUGTRT, self).__init__(**kwargs)
    
    def forward(self, img=None, lidar2img=None, img_timestamp=None, len_img_filenames=None, feat_prev_1=None, feat_prev_2=None, feat_prev_3=None, feat_prev_4=None):
        len_img_filenames = int(len_img_filenames)
        # len_img_filenames 不变，onnx会把他转化为constant
        # 如果在这里修改了lidar2img和img_timestamp, onnx也会把他们转化成constant，但这是不对的，因此我们要保留他的tensor格式

        # img = torch.stack(img)
        return self.simple_test(img, lidar2img, img_timestamp, len_img_filenames, feat_prev_1, feat_prev_2, feat_prev_3, feat_prev_4)

    def simple_test(self, img=None, lidar2img=None, img_timestamp=None, len_img_filenames=None, feat_prev_1=None, feat_prev_2=None, feat_prev_3=None, feat_prev_4=None, rescale=False):
        # img_metas[0] is a key, img.shape = [1, 6, 3, 256, 704])
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
            if i == 0: # 可以在后续验证一下是不是这样 这里默认第一个是最新的frame
                # extract feature and put into memory
                # img_feats_curr is a list with 4 tensors
                img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
                # img_feats_curr_ret = [img_feats_curr[0].clone().detach(), img_feats_curr[1].clone().detach(), img_feats_curr[2].clone().detach(), img_feats_curr[3].clone().detach()]
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
        cls_score, bbox_pred = self.pts_bbox_head.forward_trt_debug(img_feats, img_metas)  
        return cls_score, bbox_pred
