# Copyright (c) OpenMMLab. All rights reserved.
import random
from numpy import random as randomm
import warnings
import mmcv
import math

import cv2
import numpy as np
from mmcv import is_tuple_of
from mmcv.utils import build_from_cfg

from mmdet3d.core.voxel import VoxelGenerator
from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes, box_np_ops)
from mmdet3d.datasets.pipelines.compose import Compose
from mmdet.datasets.pipelines import RandomCrop, RandomFlip, Rotate
from ..builder import OBJECTSAMPLERS, PIPELINES
from .data_augment_utils import noise_per_object_v3_

from PIL import Image
import torch

@PIPELINES.register_module()
class RandomTransformImage_Changan(object):
    def __init__(self, ida_aug_conf=None, training=True):
        self.ida_aug_conf = ida_aug_conf
        self.training = training

    def __call__(self, results):
        resize, resize_dims, crop, flip, rotate, resize_front, resize_dims_front, crop_front = self.sample_augmentation()
        results['flip_dx']=0
        results['flip_dy']=0 #radar
        if len(results['lidar2img']) == len(results['img']):
            for i in range(len(results['img'])):
                img = Image.fromarray(np.uint8(results['img'][i]))

                # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
                if i%6 == 0: # front image
                    img, ida_mat = self.img_transform(
                        img,
                        resize=resize_front,
                        resize_dims=resize_dims_front,
                        crop=crop_front,
                        flip=flip,
                        rotate=rotate,
                    )
                else:
                    img, ida_mat = self.img_transform(
                        img,
                        resize=resize,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate,
                    )
                results['img'][i] = np.array(img).astype(np.uint8)
                results['lidar2img'][i] = ida_mat @ results['lidar2img'][i]

        elif len(results['img']) == 6:
            for i in range(len(results['img'])):
                img = Image.fromarray(np.uint8(results['img'][i]))

                # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
                if i%6 == 0: # front image
                    img, ida_mat = self.img_transform(
                        img,
                        resize=resize_front,
                        resize_dims=resize_dims_front,
                        crop=crop_front,
                        flip=flip,
                        rotate=rotate,
                    )
                else:
                    img, ida_mat = self.img_transform(
                        img,
                        resize=resize,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate,
                    )
                results['img'][i] = np.array(img).astype(np.uint8)

            for i in range(len(results['lidar2img'])):
                results['lidar2img'][i] = ida_mat @ results['lidar2img'][i]

        else:
            raise ValueError()

        results['ori_shape'] = [img.shape for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['pad_shape'] = [img.shape for img in results['img']]

        return results

    def img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        """
        https://github.com/Megvii-BaseDetection/BEVStereo/blob/master/dataset/nusc_mv_det_dataset.py#L48
        """

        def get_rot(h):
            return torch.Tensor([
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ])

        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)

        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])

        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b

        A = get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b

        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b

        ida_mat = torch.eye(4)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran

        return img, ida_mat.numpy()

    def sample_augmentation(self):
        """
        https://github.com/Megvii-BaseDetection/BEVStereo/blob/master/dataset/nusc_mv_det_dataset.py#L247
        """
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']
        
        H_fonrt, W_front = self.ida_aug_conf['H_fonrt'], self.ida_aug_conf['W_front']

        

        if self.training:
            resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.ida_aug_conf['rot_lim'])

            # resize for image front
            additional_resize = max(H / H_fonrt, W / W_front)
            resize_front = additional_resize * resize
            resize_dims_front = (int(W_front * resize_front), int(H_fonrt * resize_front))
            newW, newH = resize_dims_front
            crop_h = int((1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            
            crop_front = (crop_w, crop_h, crop_w + fW, crop_h + fH)

        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

            # resize for image front
            resize_front = max(fH / H_fonrt, fW / W_front)
            resize_dims_front = (int(W_front * resize), int(H_fonrt * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop_front = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        

        return resize, resize_dims, crop, flip, rotate, resize_front, resize_dims_front, crop_front
        