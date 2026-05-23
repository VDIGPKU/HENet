import os
import mmcv
import numpy as np
import open3d.io as o3dio
import torch
from PIL import Image
from pyquaternion import Quaternion
import math

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes,CameraInstance3DBoxes
from ..builder import PIPELINES
import cv2
from numpy.linalg import inv
from mmcv.runner import get_dist_info

from typing import Any, Dict, Tuple
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from nuscenes.utils.data_classes import RadarPointCloud
from ...core.points.radar_points import RadarPoints
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from functools import reduce
from mmdet3d.core.bbox import limit_period

import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms

def compose_lidar2img(ego2global_translation_curr,
                      ego2global_rotation_curr,
                      lidar2ego_translation_curr,
                      lidar2ego_rotation_curr,
                      sensor2global_translation_past,
                      sensor2global_rotation_past,
                      cam_intrinsic_past):
    R = sensor2global_rotation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T = sensor2global_translation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T -= ego2global_translation_curr @ (
                inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T) + lidar2ego_translation_curr @ inv(
        lidar2ego_rotation_curr).T

    lidar2cam_r = inv(R.T)
    lidar2cam_t = T @ lidar2cam_r.T

    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[:cam_intrinsic_past.shape[0], :cam_intrinsic_past.shape[1]] = cam_intrinsic_past
    lidar2img = (viewpad @ lidar2cam_rt.T).astype(np.float32)

    return lidar2img

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_Changan(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img_list = [mmcv.imread(name, self.color_type) for name in filename]

        # img = np.stack(img_list, axis=-1)

        if self.to_float32:
            img_list = [img.astype(np.float32) for img in img_list]
            
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        # results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img'] = img_list
        # NOTE: useless
        # results['img_shape'] = img.shape
        # results['ori_shape'] = img.shape
        # results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 3
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweeps_Changan(object):
    def __init__(self,
                 sweeps_num=5,
                 color_type='color',
                 test_mode=False):
        self.sweeps_num = sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def load_offline(self, results):
        cam_types = ['cam_front', 'cam_front_left', 'cam_front_right', 
                     'cam_rear', 'cam_rear_left', 'cam_rear_right',
                   ]

        if 'lidar2img_mf' in results.keys():
            results['lidar2img'] = results['lidar2img_mf']

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            if self.test_mode:
                interval = self.test_interval
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]
            elif len(results['sweeps']['prev']) <= self.sweeps_num:
                pad_len = self.sweeps_num - len(results['sweeps']['prev'])
                choices = list(range(len(results['sweeps']['prev']))) + [len(results['sweeps']['prev']) - 1] * pad_len
            else:
                max_interval = len(results['sweeps']['prev']) // self.sweeps_num
                max_interval = min(max_interval, self.train_interval[1])
                min_interval = min(max_interval, self.train_interval[0])
                interval = np.random.randint(min_interval, max_interval + 1)
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results

    def load_online(self, results):
        # only used when measuring FPS
        assert self.test_mode
        assert self.test_interval == 6

        cam_types = ['cam_front', 'cam_front_left', 'cam_front_right', 
                     'cam_rear', 'cam_rear_left', 'cam_rear_right',
                   ]

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
        else:
            interval = self.test_interval
            choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    # skip loading history frames
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))

        return results

    def __call__(self, results):
        if self.sweeps_num == 0:
            return results

        world_size = get_dist_info()[1]
        # if world_size == 1 and self.test_mode:
        if False:
            return self.load_online(results)
        else:
            return self.load_offline(results)


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweeps_Changan_Deploy(object):
    def __init__(self,
                 sweeps_num=5,
                 color_type='color',
                 test_mode=False,
                 suffix=''):
        self.sweeps_num = sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode
        self.suffix = suffix

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def load_offline(self, results):
        cam_types = ['cam_front', 'cam_front_left', 'cam_front_right',
                     'cam_rear', 'cam_rear_left', 'cam_rear_right',
                   ]

        if 'lidar2img_mf' in results.keys():
            results['lidar2img'] = results['lidar2img_mf']

        # 不从sweep中得到，而是和prepareimageinputs一样的修改
        assert 'adjacent' + self.suffix in results
        cnt = 0
        for adj_info in results['adjacent' + self.suffix]:
            for sensor in cam_types:
                results['img'].append(mmcv.imread(adj_info['cams'][sensor]['data_path'], self.color_type))
                results['img_timestamp'].append(adj_info['cams'][sensor]['timestamp'] / 1e6)
                results['filename'].append(os.path.relpath(adj_info['cams'][sensor]['data_path']))
                # print('sensor2global_translation', adj_info[sensor]['sensor2global_translation'])
                # print('sensor2global_rotation', adj_info[sensor]['sensor2global_rotation'])
                # print('cam_intrinsic', adj_info[sensor]['cam_intrinsic'])
                # exit(0)
                results['lidar2img'].append(compose_lidar2img(
                    results['ego2global_translation'],
                    results['ego2global_rotation'],
                    results['lidar2ego_translation'],
                    results['lidar2ego_rotation'],
                    adj_info['cams'][sensor]['sensor2global_translation'],
                    adj_info['cams'][sensor]['sensor2global_rotation'],
                    adj_info['cams'][sensor]['cam_intrinsic'],
                ))
            cnt += 1
            if cnt == self.sweeps_num: # 只取sweeps_num个，不包括当前帧
                break
        return results

    def load_online(self, results):
        assert False

    def __call__(self, results):
        if self.sweeps_num == 0:
            return results

        world_size = get_dist_info()[1]
        # if world_size == 1 and self.test_mode:
        if False:
            return self.load_online(results)
        else:
            return self.load_offline(results)


@PIPELINES.register_module()
class LoadRadarPointsMultiSweeps_Changan(object):
    """Load radar points from multiple sweeps.
    This is usually used for nuScenes dataset to utilize previous sweeps.
    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                #  load_dim=18,
                 use_dim=[0, 1, 2, 3, 4],
                 sweeps_num=3, 
                 file_client_args=dict(backend='disk'),
                 max_num=300,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 
                 test_mode=False,
                 rote90=True,
                 ignore=[]):
        # self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.max_num = max_num
        self.test_mode = test_mode
        self.pc_range = pc_range
        self.ignore = ignore
        self.rote90 = rote90
        if len(ignore)>0:
            print(self.ignore)

    def _load_points(self, pts_filename):

        radar_obj = mmcv.load(pts_filename)
        radar_type = pts_filename.split('/')[-2]
        radar_obj = radar_obj[radar_type]
        keys = list(radar_obj.keys())
        assert len(keys) == 1
        radar_obj = radar_obj[keys[0]]
        points = []
        for radar in radar_obj:
            p = [radar['center_x'], radar['center_y'], radar['center_z'],
                 radar['class'], 
                 radar['confidence']/100, radar['obstacle_prob']/100, 
                 radar['motionstatus'],
                 radar['size_x'], radar['size_y'], radar['size_z'], radar['yaw'],
                 radar['velocity_lateral'], radar['velocity_longitudinal']
                ]
            p = np.array(p)
            points.append(p)
        points = np.stack(points)

        return points.astype(np.float32)

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.
        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.
        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.
                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        radars_dict = results['radar']
        # print(radars_dict.keys())

        points_sweep_list = []
        for key, sweeps in radars_dict.items():
            if key in self.ignore:
                # print(f'ignore {key}')
                continue
            if len(sweeps) < self.sweeps_num:
                idxes = list(range(len(sweeps)))
            else:
                idxes = list(range(self.sweeps_num))
            
            ts = sweeps[0]['timestamp'] * 1e-6
            for idx in idxes:
                sweep = sweeps[idx]
                if not os.path.isfile(sweep['data_path']):
                    continue
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep)

                timestamp = sweep['timestamp'] * 1e-6
                time_diff = ts - timestamp
                # print(time_diff)
                time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff

                # velocity in sensor frame
                velo = points_sweep[:, -2:]
                velo = np.concatenate(
                    (velo, np.zeros((velo.shape[0], 1))), 1)
                velo = velo @ sweep['sensor2lidar_rotation'].T
                velo = velo[:, :2]

                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                # print()
                points_sweep_ = np.concatenate(
                    [points_sweep[:, :-2], velo, time_diff], axis=1)
                points_sweep_list.append(points_sweep_)
        
        points = np.concatenate(points_sweep_list, axis=0)
        # print(points.shape)
        
        points = points[:, self.use_dim]
        
        # print(points.shape)

        points = RadarPoints(
            points, points_dim=points.shape[-1], attribute_dims=None
        )
        if(self.rote90):
            points.rotate(-math.pi/2) #
        results['radar'] = points
        return results
    
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class LoadRadarPointsMultiSweeps_Changan_Deploy(object):
    """Load radar points from multiple sweeps.
    This is usually used for nuScenes dataset to utilize previous sweeps.
    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                #  load_dim=18,
                 use_dim=[0, 1, 2, 3, 4],
                 sweeps_num=3,
                 file_client_args=dict(backend='disk'),
                 max_num=300,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 test_mode=False,
                 rote90=True,
                 ignore=[],
                 suffix=''):
        # self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.max_num = max_num
        self.test_mode = test_mode
        self.pc_range = pc_range
        self.ignore = ignore
        self.rote90 = rote90
        self.suffix = suffix
        if len(ignore)>0:
            print(self.ignore)

    def _load_points(self, pts_filename):

        radar_obj = mmcv.load(pts_filename)
        radar_type = pts_filename.split('/')[-2]
        radar_obj = radar_obj[radar_type]
        keys = list(radar_obj.keys())
        assert len(keys) == 1
        radar_obj = radar_obj[keys[0]]
        points = []
        for radar in radar_obj:
            p = [radar['center_x'], radar['center_y'], radar['center_z'],
                 radar['class'],
                 radar['confidence']/100, radar['obstacle_prob']/100,
                 radar['motionstatus'],
                 radar['size_x'], radar['size_y'], radar['size_z'], radar['yaw'],
                 radar['velocity_lateral'], radar['velocity_longitudinal']
                ]
            p = np.array(p)
            points.append(p)
        points = np.stack(points)

        return points.astype(np.float32)

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.
        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.
        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.
                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points_sweep_list = []

        radars_dict = results['radar']
        assert 'adjacent' + self.suffix in results
        # 当前帧和 过去帧合并
        all_radars = [radars_dict]
        for adj_info in results['adjacent' + self.suffix]:
            all_radars.append(adj_info['radars']) # 注意这里radars和radar的区别

        for key, sweeps in radars_dict.items():
            if key in self.ignore:
                # print(f'ignore {key}')
                continue
            ts = sweeps[0]['timestamp'] * 1e-6

            cnt = 0
            for cur_radars_dict in all_radars:
                cur_sweeps = cur_radars_dict[key]
                sweep = cur_sweeps[0]
                if not os.path.isfile(sweep['data_path']):
                    continue
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep)

                timestamp = sweep['timestamp'] * 1e-6
                time_diff = ts - timestamp
                # print(time_diff)
                time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff

                # velocity in sensor frame
                velo = points_sweep[:, -2:]
                velo = np.concatenate(
                    (velo, np.zeros((velo.shape[0], 1))), 1)
                velo = velo @ sweep['sensor2lidar_rotation'].T
                velo = velo[:, :2]

                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                # print()
                points_sweep_ = np.concatenate(
                    [points_sweep[:, :-2], velo, time_diff], axis=1)
                points_sweep_list.append(points_sweep_)
                cnt += 1
                if cnt == self.sweeps_num: # 只取sweeps_num个，包括了当前帧
                    break

        points = np.concatenate(points_sweep_list, axis=0)
        # print(points.shape)

        points = points[:, self.use_dim]

        # print(points.shape)

        points = RadarPoints(
            points, points_dim=points.shape[-1], attribute_dims=None
        )
        if(self.rote90):
            points.rotate(-math.pi/2) #
        results['radar'] = points
        return results


    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class PrepareImageInputs_Changan(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_config,
            is_train=False,
            sequential=False,
            ego_cam='CAM_FRONT',
            add_adj_bbox=False,
            with_stereo=False,
            with_future_pred=False,
            img_norm_cfg=None,
            suffix='',
            ignore=[],
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.ego_cam = ego_cam
        self.with_future_pred = with_future_pred
        self.add_adj_bbox = add_adj_bbox
        self.img_norm_cfg = img_norm_cfg
        self.with_stereo = with_stereo
        self.suffix = suffix
        self.ignore = ignore
        if len(ignore) > 0:
            print(self.ignore)

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sweep2key_transformation(self,
                                     cam_info,
                                     key_info,
                                     cam_name,
                                     ego_cam=None):
        if ego_cam is None:
            ego_cam = cam_name
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        sweepego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        sweepego2global = sweepego2global_rot.new_zeros((4, 4))
        sweepego2global[3, 3] = 1
        sweepego2global[:3, :3] = sweepego2global_rot
        sweepego2global[:3, -1] = sweepego2global_tran

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][ego_cam]['ego2global_rotation']
        keyego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global_tran = torch.Tensor(
            key_info['cams'][ego_cam]['ego2global_translation'])
        keyego2global = keyego2global_rot.new_zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = keyego2global.inverse()

        sweepego2keyego = global2keyego @ sweepego2global

        return sweepego2keyego

    def get_sensor_transforms(self, cam_info, cam_name):
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        imgs_ori = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        results['occ_filename'] = []
        canvas = []
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            results['img_file_paths'][cam_name] = filename  # for vis
            results['occ_filename'].append(filename)
            img_ori = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img_ori.height, W=img_ori.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img_ori, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            # print(cam_name, self.ignore)
            if cam_name in self.ignore:
                canvas.append(np.zeros_like(np.array(img)))
                imgs.append(torch.zeros_like(self.normalize_img(img)))
                imgs_ori.append(torch.zeros_like(np.array(img_ori)))  # for vis
            else:
                canvas.append(np.array(img))
                imgs.append(self.normalize_img(img))
                imgs_ori.append(np.array(img_ori))  # for vis

            if self.sequential:
                assert 'adjacent' + self.suffix in results
                for adj_info in results['adjacent' + self.suffix]:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    results['occ_filename'].append(filename_adj)
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    if cam_name in self.ignore:
                        imgs.append(torch.zeros_like(self.normalize_img(img_adjacent)))
                        imgs_ori.append(torch.zeros_like(np.array(img_adjacent)))  # for vis
                    else:
                        imgs.append(self.normalize_img(img_adjacent))
                        imgs_ori.append(np.array(img_adjacent))  # for vis
            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            for adj_info in results['adjacent' + self.suffix]:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])
                results['ego2global_rotation_quaternion'].append(adj_info['ego2global_rotation'])

                # align
                for cam_name in cam_names:
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)
            if self.add_adj_bbox:
                results['adjacent_bboxes'] = self.align_adj_bbox2keyego(results)

        imgs = torch.stack(imgs)
        imgs_ori = [torch.tensor(img) for img in imgs_ori]  # for vis
        imgs_ori = torch.stack(imgs_ori)  # for vis

        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        results['canvas' + self.suffix] = canvas
        results['img_shape'] = [(self.data_config['input_size'][0], self.data_config['input_size'][1]) for _ in
                                range(6)]
        results['imgs_ori'] = imgs_ori  # for vis
        return (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans)

    def __call__(self, results):
        results['img_file_paths'] = {}
        if self.add_adj_bbox:
            results['adjacent_bboxes'] = self.get_adjacent_bboxes(results)
        results['img_inputs' + self.suffix] = self.get_inputs(results)
        results['input_shape' + self.suffix] = self.data_config['input_size']  # new
        return results

    def get_adjacent_bboxes(self, results):
        adjacent_bboxes = list()
        for idx, adj_info in enumerate(results['adjacent']):
            if self.with_stereo and idx > 0:  # reference frame不用读gt
                break
            adjacent_bboxes.append(adj_info['ann_infos'])
        return adjacent_bboxes

    def align_adj_bbox2keyego(self, results):
        cam_name = self.choose_cams()[0]
        ret_list = []
        for idx, adj_info in enumerate(results['adjacent']):
            if self.with_stereo and idx > 0:  # reference frame不用读gt 也不用align
                break
            sweepego2keyego = self.get_sweep2key_transformation(adj_info,
                                                                results['curr'],
                                                                cam_name,
                                                                self.ego_cam)
            adj_bbox, adj_labels = results['adjacent_bboxes'][idx]
            adj_bbox = torch.Tensor(adj_bbox)
            adj_labels = torch.tensor(adj_labels)
            gt_bbox = adj_bbox
            if len(adj_bbox) == 0:
                adj_bbox = torch.zeros(0, 9)
                ret_list.append((adj_bbox, adj_labels))
                continue
            # center
            homo_sweep_center = torch.cat([gt_bbox[:, :3], torch.ones_like(gt_bbox[:, 0:1])], dim=-1)
            homo_key_center = (sweepego2keyego @ homo_sweep_center.t()).t()  # [4, N]
            # velo
            rot = sweepego2keyego[:3, :3]
            homo_sweep_velo = torch.cat([gt_bbox[:, 7:], torch.zeros_like(gt_bbox[:, 0:1])], dim=-1)
            homo_key_velo = (rot @ homo_sweep_velo.t()).t()

            # yaw
            def get_new_yaw(box_cam, extrinsic):
                corners = box_cam.corners
                cam2lidar_rt = torch.tensor(extrinsic)
                N = corners.shape[0]
                corners = corners.reshape(N * 8, 3)
                extended_xyz = torch.cat(
                    [corners, corners.new_ones(corners.size(0), 1)], dim=-1)
                corners = extended_xyz @ cam2lidar_rt.T
                corners = corners.reshape(N, 8, 4)[:, :, :3]
                yaw = np.arctan2(corners[:, 1, 1] - corners[:, 2, 1], corners[:, 1, 0] - corners[:, 2, 0])

                def limit_period(val, offset=0.5, period=np.pi):
                    """Limit the value into a period for periodic function.

                    Args:
                        val (np.ndarray): The value to be converted.
                        offset (float, optional): Offset to set the value range. \
                            Defaults to 0.5.
                        period (float, optional): Period of the value. Defaults to np.pi.

                    Returns:
                        torch.Tensor: Value in the range of \
                            [-offset * period, (1-offset) * period]
                    """
                    return val - np.floor(val / period + offset) * period

                return limit_period(yaw + (np.pi / 2), period=np.pi * 2)

            new_yaw_sweep = get_new_yaw(LiDARInstance3DBoxes(adj_bbox, box_dim=adj_bbox.shape[-1],
                                                             origin=(0.5, 0.5, 0.5)), sweepego2keyego).reshape(-1, 1)
            adj_bbox = torch.cat([homo_key_center[:, :3], gt_bbox[:, 3:6], new_yaw_sweep, homo_key_velo[:, :2]], dim=-1)
            ret_list.append((adj_bbox, adj_labels))

        return ret_list


@PIPELINES.register_module()
class LoadOccGTFromFile_Changan(object):
    def __init__(self,
                 mask_path=None,
                 thresh=0.2,
                 mask_level=1,
                 ):
        self.mask_path = mask_path
        self.thresh = thresh
        self.mask_level = mask_level

    def __call__(self, results):

        occ_path = results['pts_filename']
        occ_path = occ_path.replace('bev_pcd', 'gt_occ')
        occ_path = occ_path.replace('.pcd', '.npy')
        occ_mask_path = occ_path.replace('gt_occ', 'gt_occ_mask')
        # print(occ_path, occ_mask_path)
        semantics = np.load(occ_path)
        semantics = np.where(semantics == 0, 17, semantics).astype(np.uint8)
        mask_lidar = np.load(occ_mask_path)
        mask_lidar = (mask_lidar >= self.mask_level)
        mask_lidar = mask_lidar.astype(np.uint8)
        # import ipdb; ipdb.set_trace()

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_lidar

        if 'hop_load_all' in results and results['hop_load_all']:
            raise ValueError("changan pipeline do not support hop_load_all now")

        return results


@PIPELINES.register_module()
class LoadAnnotationsBEVDepth_Changan(object):

    def __init__(self, bda_aug_conf, classes, is_train=True, sequential=False, align_adj_bbox=False, with_hop=False,
                 is_val=True, use_centerpoint=False):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes
        self.sequential = sequential
        self.align_adj_bbox = align_adj_bbox
        self.with_hop = with_hop
        self.is_val = is_val  # if is_val then load bbox gt for seg gt
        self.use_centerpoint = use_centerpoint

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        fsr_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                    fsr_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                         6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                    fsr_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, fsr_mat, rot_mat, flip_mat, scale_mat

    def __call__(self, results):
        if self.use_centerpoint == True and self.is_train == True:
            gt_boxes, gt_labels = results['ann_info']['gt_bboxes_3d'].tensor, results['ann_info']['gt_labels_3d']
            # gt_boxes = torch.Tensor(gt_boxes)
            # gt_boxes = LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
            #                                 origin=(0.5, 0.5, 0.5))
            # torch.save(gt_boxes.corners, 'vis/vis_bbox/gt_boxes_changan_bevdet.bin')
            # torch.save(gt_labels, 'vis/vis_bbox/gt_labels_changan_bevdet.bin')
            # exit(0)
            gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        else:
            gt_boxes = torch.zeros(0, 9)
            gt_labels = torch.zeros(0, 1)

        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        results['rotate_bda'] = rotate_bda
        results['scale_bda'] = scale_bda
        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy  # save

        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot, rm, fm, sm = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                                           flip_dx, flip_dy)
        bda_mat[:3, :3] = rm
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels

        if 'points' in results:
            points = results['points']
            lidar2ego = results['lidar2ego']
            # points.rotate(lidar2ego[:3, :3].T)
            # points.tensor[:, :3] = points.tensor[:, :3] + lidar2ego[:3, 3]
            points.tensor[:, :3] = (bda_rot @ points.tensor[:, :3].unsqueeze(-1)).squeeze(-1)
            results['points'] = points

        if 'img_inputs' in results:
            imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
            post_rots, post_trans = results['img_inputs'][4:]
            results['img_inputs'] = (imgs, sensor2egos, ego2globals, intrins, post_rots,
                                     post_trans, bda_rot)
            ego2img_rts = []
            if not self.sequential and self.with_hop:
                sensor2keyegos = self.get_sensor2keyego_transformation(sensor2egos, ego2globals)
                for sensor2keyego, intrin, post_rot, post_tran in zip(
                        sensor2keyegos, intrins, post_rots, post_trans):
                    rot = sensor2keyego[:3, :3]
                    tran = sensor2keyego[:3, 3]
                    viewpad = torch.eye(3).to(imgs.device)
                    viewpad[:post_rot.shape[0], :post_rot.shape[1]] = \
                        post_rot @ intrin[:post_rot.shape[0], :post_rot.shape[1]]
                    viewpad[:post_tran.shape[0], 2] += post_tran
                    intrinsic = viewpad

                    # need type float
                    ego2img_r = intrinsic.float() @ torch.linalg.inv(rot.float()) @ torch.linalg.inv(bda_rot.float())
                    ego2img_t = -intrinsic.float() @ torch.linalg.inv(rot.float()) @ tran.float()
                    ego2img_rt = torch.eye(4).to(imgs.device)
                    ego2img_rt[:3, :3] = ego2img_r
                    ego2img_rt[:3, 3] = ego2img_t
                    '''
                    X_{3d} = bda * (rots * (intrinsic)^(-1) * X_{img} + trans)
                    bda^(-1) * X_{3d} = rots * (intrinsic)^(-1) * X_{img} + trans
                    bda^(-1) * X_{3d} - trans = rots * (intrinsic)^(-1) * X_{img}
                    intrinsic * rots^(-1) * (bda^(-1) * X_{3d} - trans) = X_{img}
                    intrinsic * rots^(-1) * bda^(-1) * X_{3d} - intrinsic * rots^(-1) * trans = X_{img}
                    rotate = intrinsic * rots^(-1) * bda^(-1)
                    translation = - intrinsic * rots^(-1) * trans
                    '''
                    ego2img_rts.append(ego2img_rt)
                ego2img_rts = torch.stack(ego2img_rts, dim=0)
            if self.align_adj_bbox:
                results = self.align_adj_bbox_bda(results, rotate_bda, scale_bda,
                                                  flip_dx, flip_dy)
            results['lidar2img'] = np.asarray(ego2img_rts)

            if 'img_inputs_lt' in results.keys():
                imgs_lt, rots_lt, trans_lt, intrins_lt = results['img_inputs_lt'][:4]
                post_rots_lt, post_trans_lt = results['img_inputs_lt'][4:]
                results['img_inputs_lt'] = (imgs_lt, rots_lt, trans_lt, intrins_lt,
                                            post_rots_lt, post_trans_lt, bda_rot)

        results["bda_r"] = bda_mat
        results["bda_f"] = (flip_dx, flip_dy)
        results["bda_s"] = scale_bda
        # print('LLLLL 2392  ++LoadAnnotationsBEVDepth', len(results['lidar2img']))
        return results

    def get_sensor2keyego_transformation(self, sensor2egos, ego2globals):
        # sensor2ego -> sweep sensor to sweep ego
        # ego2globals -> sweep ego to global
        sensor2keyegos = []
        keyego2global = ego2globals[0]  # assert key ego is frame 0 with CAM_FRONT
        global2keyego = torch.inverse(keyego2global.double())
        for sensor2ego, ego2global in zip(sensor2egos, ego2globals):
            # calculate the transformation from sweep sensor to key ego
            sensor2keyego = global2keyego @ ego2global.double() @ sensor2ego.double()
            sensor2keyegos.append(sensor2keyego)
        return sensor2keyegos

    def align_adj_bbox_bda(self, results, rotate_bda, scale_bda, flip_dx, flip_dy):
        for adjacent_bboxes in results['adjacent_bboxes']:
            adj_bbox, adj_label = adjacent_bboxes
            gt_boxes = adj_bbox
            if len(gt_boxes) == 0:
                gt_boxes = torch.zeros(0, 9)
            gt_boxes, _, _, _, _ = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                                      flip_dx, flip_dy)
            if not 'adj_gt_3d' in results.keys():
                adj_bboxes_3d = \
                    LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                         origin=(0.5, 0.5, 0.5))
                adj_labels_3d = adj_label
                results['adj_gt_3d'] = [[adj_bboxes_3d, adj_labels_3d]]
            else:
                adj_bboxes_3d = \
                    LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                         origin=(0.5, 0.5, 0.5))
                results['adj_gt_3d'].append([
                    adj_bboxes_3d, adj_label
                ])
        return results


@PIPELINES.register_module()
class LoadPointsFromFile_Changan(object):
    def __init__(self, coord_type):
        self.coord_type = coord_type

    def __call__(self, results):
        pcd_path = results['pts_filename']
        pcd = o3dio.read_point_cloud(pcd_path)
        points = np.array(pcd.points)

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1])

        results['points'] = points
        return results

