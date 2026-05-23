# Copyright (c) OpenMMLab. All rights reserved.
import os
from os import path as osp
import time
import copy
import tempfile
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
import torch

from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .occ_metrics import Metric_mIoU, Metric_FScore
from .ray import generate_rays, generate_rays_nframe
from .vad_custom_nuscenes_eval import NuScenesEval_custom
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes

nusc_class_nums = torch.Tensor([
    2854504, 7291443, 141614, 4239939, 32248552,
    1583610, 364372, 2346381, 582961, 4829021,
    14073691, 191019309, 6249651, 55095657,
    58484771, 193834360, 131378779
])
dynamic_class = [0, 1, 3, 4, 5, 7, 9, 10]


def load_depth(img_file_path, gt_path):
    file_name = os.path.split(img_file_path)[-1]
    cam_depth = np.fromfile(os.path.join(gt_path, f'{file_name}.bin'),
                            dtype=np.float32,
                            count=-1).reshape(-1, 3)

    coords = cam_depth[:, :2].astype(np.int16)
    depth_label = cam_depth[:, 2]
    return coords, depth_label


def load_seg_label(img_file_path, gt_path, img_size=[900, 1600], mode='lidarseg'):
    if mode == 'lidarseg':  # proj lidarseg to img
        coor, seg_label = load_depth(img_file_path, gt_path)
        seg_map = np.zeros(img_size)
        seg_map[coor[:, 1], coor[:, 0]] = seg_label
    else:
        file_name = os.path.join(gt_path, f'{os.path.split(img_file_path)[-1]}.npy')
        seg_map = np.load(file_name)
    return seg_map


def get_sensor_transforms(cam_info, cam_name):
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


@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):
    def __init__(self,
                 use_rays=False,
                 load_traj=False,
                 load_others=False,
                 semantic_gt_path=None,
                 depth_gt_path=None,
                 aux_frames=[-1, 1],
                 max_ray_nums=0,
                 wrs_use_batch=False,
                 load_adj_occ_labels=False,
                 hop_target_frame=-1,
                 hop_load_all=False,
                 det_info_file=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_rays = use_rays
        self.load_traj = load_traj
        self.load_others = load_others
        self.semantic_gt_path = semantic_gt_path
        self.depth_gt_path = depth_gt_path
        self.aux_frames = aux_frames
        self.max_ray_nums = max_ray_nums

        if wrs_use_batch:  # compute with batch data
            self.WRS_balance_weight = None
        else:  # compute with total dataset
            self.WRS_balance_weight = torch.exp(0.005 * (nusc_class_nums.max() / nusc_class_nums - 1))

        self.dynamic_class = torch.tensor(dynamic_class)

        self.load_adj_occ_labels = load_adj_occ_labels
        self.hop_target_frame = hop_target_frame
        self.hop_load_all = hop_load_all

        self.det_info_file = det_info_file
        if det_info_file is not None:
            # load annotations
            if hasattr(self.file_client, 'get_local_path'):
                with self.file_client.get_local_path(det_info_file) as local_path:
                    self.det_infos = self.load_annotations(open(local_path, 'rb'))
            else:
                self.det_infos = self.load_annotations(self.ann_file)

    def get_rays(self, index):
        info = self.data_infos[index]

        sensor2egos = []
        ego2globals = []
        intrins = []
        coors = []
        label_depths = []
        label_segs = []
        time_ids = {}
        idx = 0

        for time_id in [0] + self.aux_frames:
            time_ids[time_id] = []
            select_id = max(index + time_id, 0)

            if select_id >= len(self.data_infos) or self.data_infos[select_id]['scene_token'] != info['scene_token']:
                select_id = index  # out of sequence
            info = self.data_infos[select_id]

            for cam_name in info['cams'].keys():
                intrin = torch.Tensor(info['cams'][cam_name]['cam_intrinsic'])
                sensor2ego, ego2global = get_sensor_transforms(info, cam_name)
                img_file_path = info['cams'][cam_name]['data_path']

                # load seg/depth GT of rays
                seg_map = load_seg_label(img_file_path, self.semantic_gt_path)
                coor, label_depth = load_depth(img_file_path, self.depth_gt_path)
                label_seg = seg_map[coor[:, 1], coor[:, 0]]

                sensor2egos.append(sensor2ego)
                ego2globals.append(ego2global)
                intrins.append(intrin)
                coors.append(torch.Tensor(coor))
                label_depths.append(torch.Tensor(label_depth))
                label_segs.append(torch.Tensor(label_seg))
                time_ids[time_id].append(idx)
                idx += 1

        T, N = len(self.aux_frames) + 1, len(info['cams'].keys())
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        sensor2egos = sensor2egos.view(T, N, 4, 4)
        ego2globals = ego2globals.view(T, N, 4, 4)

        # calculate the transformation from adjacent_sensor to key_ego
        keyego2global = ego2globals[0, :, ...].unsqueeze(0)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()
        sensor2keyegos = sensor2keyegos.view(T * N, 4, 4)

        # generate rays for all frames
        rays = generate_rays(
            coors, label_depths, label_segs, sensor2keyegos, intrins,
            max_ray_nums=self.max_ray_nums,
            time_ids=time_ids,
            dynamic_class=self.dynamic_class,
            balance_weight=self.WRS_balance_weight)
        return rays

    def get_data_info(self, index):
        info = self.data_infos[index]
        # import ipdb; ipdb.set_trace()
        if self.det_info_file is not None:
            sweeps_prev, sweeps_next = self.collect_sweeps_det(index)
        else:
            sweeps_prev, sweeps_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation = Quaternion(lidar2ego_rotation).rotation_matrix

        # standard protocol modified from SECOND.Pytorch
        if self.include_location:
            input_dict = dict(
                sample_idx=info['token'],
                pts_filename=info['lidar_path'],
                # sweeps=info['sweeps'],
                timestamp=info['timestamp'] / 1e6,
                location=info["location"],
                scene_token=info['scene_token'],
                scene_name=info['scene_name'] if 'scene_name' in info.keys() else '',
                frame_idx=info['frame_idx'] if 'frame_idx' in info.keys() else -1,
                sweeps={'prev': sweeps_prev, 'next': sweeps_next},
                ego2global_translation=ego2global_translation,
                ego2global_rotation=ego2global_rotation,
                lidar2ego_translation=lidar2ego_translation,
                lidar2ego_rotation=lidar2ego_rotation,
            )
        else:
            input_dict = dict(
                sample_idx=info['token'],
                pts_filename=info['lidar_path'],
                # sweeps=info['sweeps'],
                timestamp=info['timestamp'] / 1e6,
                scene_token=info['scene_token'],
                scene_name=info['scene_name'] if 'scene_name' in info.keys() else '',
                frame_idx=info['frame_idx'] if 'frame_idx' in info.keys() else -1,
                sweeps={'prev': sweeps_prev, 'next': sweeps_next},
                ego2global_translation=ego2global_translation,
                ego2global_rotation=ego2global_rotation,
                lidar2ego_translation=lidar2ego_translation,
                lidar2ego_rotation=lidar2ego_rotation,
            )

        if 'radars' in info:
            input_dict['radar'] = info['radars']

        # ego to global transform
        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = info["ego2global_translation"]
        input_dict["ego2global"] = ego2global

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        input_dict["lidar2ego"] = lidar2ego
        # import ipdb; ipdb.set_trace()
        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']

        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                img_timestamps = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(os.path.relpath(cam_info['data_path']))
                    img_timestamps.append(cam_info['timestamp'] / 1e6)
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.
                    shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        img_timestamp=img_timestamps,
                        lidar2img=lidar2img_rts,
                    ))

                if not self.test_mode:
                    annos = self.get_ann_info(index)
                    input_dict['ann_info'] = annos

            elif self.img_info_prototype == 'mmcv+bevdet4d':
                image_paths = []
                img_timestamps = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(os.path.relpath(cam_info['data_path']))
                    img_timestamps.append(cam_info['timestamp'] / 1e6)
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.
                    shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        img_timestamp=img_timestamps,
                        lidar2img_mf=lidar2img_rts,
                    ))

                if not self.test_mode:
                    annos = self.get_ann_info(index)
                    input_dict['ann_info'] = annos

                assert 'bevdet' in self.img_info_prototype
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))
                    if self.multi_adj_frame_id_cfg_longterm is not None:
                        info_adj_list_lt = self.get_adj_info_lt(info, index)
                        input_dict.update(dict(adjacent_lt=info_adj_list_lt))

            else:
                assert 'bevdet' in self.img_info_prototype  # bev 自己写的里多帧的模型作为baseline，eta会不准）
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))
                    if self.multi_adj_frame_id_cfg_longterm is not None:
                        info_adj_list_lt = self.get_adj_info_lt(info, index)
                        input_dict.update(dict(adjacent_lt=info_adj_list_lt))
        else:
            if not self.test_mode:
                annos = self.get_ann_info(index)
                input_dict['ann_info'] = annos

        # occ info
        input_dict['with_gt'] = self.data_infos[index]['with_gt'] if 'with_gt' in self.data_infos[index] else True
        if 'occ_path' in self.data_infos[index]:
            input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']

        if self.hop_load_all:
            adj_occ_path_list = []
            for i in self.aux_frames:
                # print(i)
                # new_index = index + i
                cur_data_info = self.data_infos[max(0, index + i)]
                cur_occ_path = cur_data_info['occ_path']
                adj_occ_path_list.append(cur_occ_path)
            input_dict['hop_load_all'] = True
            input_dict['hop_all_path'] = {"adj_path": adj_occ_path_list}

            input_dict['with_target_occ'] = False
            return input_dict

        if self.load_adj_occ_labels:
            adj_occ_gt_path = self.load_adj_occ_gt_path(index=index, aux_frames=self.aux_frames)
            input_dict['target_occ_gt_path'] = adj_occ_gt_path[self.hop_target_frame]
            input_dict['with_target_occ'] = True
        # generate rays for rendering supervision
        if self.use_rays:
            rays_info = self.get_rays(index)
            input_dict['rays'] = rays_info
        else:
            input_dict['rays'] = torch.zeros((1))

        return input_dict

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det, self.with_velocity)
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        'car',
                        'construction_vehicle',
                        'bus',
                        'truck',
                        'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def evaluate(self, results,
                 runner=None,
                 show_dir=None,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 **eval_kwargs):

        metrics = {}

        if "pts_seg" in results[0]:
            metrics.update(self.evaluate_map(results))

        if "pts_bbox" in results[0]:
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

            if isinstance(result_files, dict):
                for name in result_names:
                    if name == 'pts_seg' or 'gt' in name or 'pts_occ' in name:
                        continue
                    print("Evaluating bboxes of {}".format(name))
                    ret_dict = self._evaluate_single(result_files[name])
                    metrics.update(ret_dict)
            elif isinstance(result_files, str):
                metrics.update(self._evaluate_single(result_files))

            if tmp_dir is not None:
                tmp_dir.cleanup()

            if show or out_dir:
                self.show(results, out_dir, show=show, pipeline=pipeline)

        if "pts_occ" in results[0]:
            self.occ_eval_metrics = Metric_mIoU(
                num_classes=18,
                use_lidar_mask=False,
                use_image_mask=True)

            print('\nStarting Evaluation...')
            for index, result in enumerate(tqdm(results)):
                occ_pred = result['pts_occ']
                info = self.data_infos[index]
                occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
                gt_semantics = occ_gt['semantics']
                mask_lidar = occ_gt['mask_lidar'].astype(bool)
                mask_camera = occ_gt['mask_camera'].astype(bool)
                self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)
            metrics.update({'occ_count_miou': self.occ_eval_metrics.count_miou()})

        return metrics

    def load_adj_occ_gt_path(self, index=-1, aux_frames=[-3, -2, -1]):
        adj_occ_gt_path = []
        for i in aux_frames:
            select_id = index + i
            occ_gt_path = self.data_infos[select_id]['occ_path']
            adj_occ_gt_path.append(occ_gt_path)

        return adj_occ_gt_path

    def collect_sweeps(self, index, into_past=60, into_future=60):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            curr_index = curr_index - 1

        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def collect_sweeps_det(self, index, into_past=60, into_future=60):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.det_infos[curr_index]['sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.det_infos[curr_index - 1]['cams'])
            curr_index = curr_index - 1

        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.det_infos):
                break
            curr_sweeps = self.det_infos[curr_index]['sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.det_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next


def output_to_nusc_box(detection, with_velocity=True):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # our LiDAR coordinate system -> nuScenes box coordinate system
    nus_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if with_velocity:
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (0, 0, 0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list


@DATASETS.register_module()
class NuScenesDatasetOccpancyPlanner(NuScenesDatasetOccpancy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.with_velocity = True
        self.with_attr = True

    def get_data_info(self, index):
        info = self.data_infos[index]
        # import ipdb; ipdb.set_trace()
        if self.det_info_file is not None:
            sweeps_prev, sweeps_next = self.collect_sweeps_det(index)
        else:
            sweeps_prev, sweeps_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation = Quaternion(lidar2ego_rotation).rotation_matrix

        # standard protocol modified from SECOND.Pytorch
        if self.include_location:
            input_dict = dict(
                sample_idx=info['token'],
                pts_filename=info['lidar_path'],
                # sweeps=info['sweeps'],
                timestamp=info['timestamp'] / 1e6,
                location=info["location"],
                scene_token=info['scene_token'],
                scene_name=info['scene_name'] if 'scene_name' in info.keys() else '',
                frame_idx=info['frame_idx'] if 'frame_idx' in info.keys() else -1,
                sweeps={'prev': sweeps_prev, 'next': sweeps_next},
                ego2global_translation=ego2global_translation,
                ego2global_rotation=ego2global_rotation,
                lidar2ego_translation=lidar2ego_translation,
                lidar2ego_rotation=lidar2ego_rotation,
            )
        else:
            input_dict = dict(
                sample_idx=info['token'],
                pts_filename=info['lidar_path'],
                # sweeps=info['sweeps'],
                timestamp=info['timestamp'] / 1e6,
                scene_token=info['scene_token'],
                scene_name=info['scene_name'] if 'scene_name' in info.keys() else '',
                frame_idx=info['frame_idx'] if 'frame_idx' in info.keys() else -1,
                sweeps={'prev': sweeps_prev, 'next': sweeps_next},
                ego2global_translation=ego2global_translation,
                ego2global_rotation=ego2global_rotation,
                lidar2ego_translation=lidar2ego_translation,
                lidar2ego_rotation=lidar2ego_rotation,
            )

        if 'radars' in info:
            input_dict['radar'] = info['radars']

        # ego to global transform
        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = info["ego2global_translation"]
        input_dict["ego2global"] = ego2global

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        input_dict["lidar2ego"] = lidar2ego
        # import ipdb; ipdb.set_trace()
        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']

        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                img_timestamps = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(os.path.relpath(cam_info['data_path']))
                    img_timestamps.append(cam_info['timestamp'] / 1e6)
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.
                    shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        img_timestamp=img_timestamps,
                        lidar2img=lidar2img_rts,
                    ))

                if not self.test_mode:
                    annos = self.get_ann_info(index)
                    input_dict['ann_info'] = annos

            elif self.img_info_prototype == 'mmcv+bevdet4d':
                image_paths = []
                img_timestamps = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(os.path.relpath(cam_info['data_path']))
                    img_timestamps.append(cam_info['timestamp'] / 1e6)
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.
                    shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        img_timestamp=img_timestamps,
                        lidar2img_mf=lidar2img_rts,
                    ))

                if not self.test_mode:
                    annos = self.get_ann_info(index)
                    input_dict['ann_info'] = annos

                assert 'bevdet' in self.img_info_prototype
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))
                    if self.multi_adj_frame_id_cfg_longterm is not None:
                        info_adj_list_lt = self.get_adj_info_lt(info, index)
                        input_dict.update(dict(adjacent_lt=info_adj_list_lt))

            else:
                assert 'bevdet' in self.img_info_prototype  # bev 自己写的里多帧的模型作为baseline，eta会不准）
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))
                    if self.multi_adj_frame_id_cfg_longterm is not None:
                        info_adj_list_lt = self.get_adj_info_lt(info, index)
                        input_dict.update(dict(adjacent_lt=info_adj_list_lt))
        else:
            if not self.test_mode:
                annos = self.get_ann_info(index)
                input_dict['ann_info'] = annos

        # occ info
        input_dict['with_gt'] = self.data_infos[index]['with_gt'] if 'with_gt' in self.data_infos[index] else True
        if 'occ_path' in self.data_infos[index]:
            input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']

        if self.hop_load_all:
            adj_occ_path_list = []
            for i in self.aux_frames:
                # print(i)
                # new_index = index + i
                cur_data_info = self.data_infos[max(0, index + i)]
                cur_occ_path = cur_data_info['occ_path']
                adj_occ_path_list.append(cur_occ_path)
            input_dict['hop_load_all'] = True
            input_dict['hop_all_path'] = {"adj_path": adj_occ_path_list}

            input_dict['with_target_occ'] = False
            return input_dict

        if self.load_adj_occ_labels:
            adj_occ_gt_path = self.load_adj_occ_gt_path(index=index, aux_frames=self.aux_frames)
            input_dict['target_occ_gt_path'] = adj_occ_gt_path[self.hop_target_frame]
            input_dict['with_target_occ'] = True
        # generate rays for rendering supervision
        if self.use_rays:
            rays_info = self.get_rays(index)
            input_dict['rays'] = rays_info
        else:
            input_dict['rays'] = torch.zeros((1))

        if self.load_traj:  # loading traj from info file
            input_dict['ego_his_trajs'] = self.det_infos[index]['gt_ego_his_trajs']
            input_dict['ego_fut_trajs'] = self.det_infos[index]['gt_ego_fut_trajs']
            input_dict['ego_fut_masks'] = self.det_infos[index]['gt_ego_fut_masks']

            gt_agent_fut_trajs = self.det_infos[index]['gt_agent_fut_trajs']  # [A, 12]
            A, _ = gt_agent_fut_trajs.shape

            gt_agent_fut_trajs = torch.tensor(gt_agent_fut_trajs)  # [A, 12]
            gt_agent_fut_trajs = gt_agent_fut_trajs.view(A, 6, 2)  # reshape 成 (A, 6, 2)


            # import ipdb; ipdb.set_trace()

            # 初始位置：从 gt_boxes 取前两个维度 (x, y)，shape [A, 2]
            gt_boxes = self.det_infos[index]['gt_boxes']
            initial_xy = torch.tensor(gt_boxes[:, :2])  # [A, 2]

            # 计算累积和（轨迹位移 → 轨迹坐标）
            abs_trajs = torch.cumsum(gt_agent_fut_trajs, dim=1) + initial_xy.unsqueeze(1)  # [A, 6, 2]

            # 创建 0 padding 的结果
            padded_trajs = torch.zeros((200, 6, 2), dtype=abs_trajs.dtype)
            padded_trajs[:A] = abs_trajs

            # import matplotlib.pyplot as plt
            # # 可视化 agent future trajectories
            # fig, ax = plt.subplots(figsize=(6, 6))
            # for traj in abs_trajs:
            #     traj_np = traj.cpu().numpy()  # [6, 2]
            #     ax.plot(traj_np[:, 0], traj_np[:, 1], marker='o', linestyle='-', alpha=0.7)
            # ax.set_title('Agent Future Trajectories')
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.axis('equal')

            # save_dir = '/data/bevperception'
            # os.makedirs(save_dir, exist_ok=True)
            # save_path = os.path.join(save_dir, f'agent_fut_trajs_{index}.png')
            # plt.savefig(save_path)
            # plt.close(fig)


            input_dict['gt_fut_trajs_abs'] = padded_trajs  # [200, 6, 2]

            # import ipdb; ipdb.set_trace()

            
        if self.load_others:
            input_dict['fut_valid_flag'] = self.det_infos[index]['fut_valid_flag']
            input_dict['gt_ego_fut_cmd'] = self.det_infos[index]['gt_ego_fut_cmd']
            input_dict['gt_ego_lcf_feat'] = self.det_infos[index]['gt_ego_lcf_feat']

            mask = self.det_infos[index]['valid_flag']
            input_dict['gt_boxes'] = self.det_infos[index]['gt_boxes'][mask]
            gt_fut_trajs = self.det_infos[index]['gt_agent_fut_trajs'][mask]
            input_dict['gt_fut_trajs'] = gt_fut_trajs
            gt_fut_masks = self.det_infos[index]['gt_agent_fut_masks'][mask]
            gt_fut_goal = self.det_infos[index]['gt_agent_fut_goal'][mask]
            gt_lcf_feat = self.det_infos[index]['gt_agent_lcf_feat'][mask]
            gt_fut_yaw = self.det_infos[index]['gt_agent_fut_yaw'][mask]
            attr_labels = np.concatenate(
                [gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None], gt_lcf_feat, gt_fut_yaw], axis=-1
            ).astype(np.float32)
            input_dict['gt_attr_labels'] = attr_labels
        # print(f'dict keys: {input_dict.keys()}')
        # import ipdb; ipdb.set_trace()
        return input_dict

    def to_scalar(self, tensor_or_float):
        if isinstance(tensor_or_float, torch.Tensor):
            return tensor_or_float.detach().cpu().item()
        return tensor_or_float

    def safe_divide(self, sum_, count):
        return sum_ / count if count > 0 else float('nan')

    def evaluate(self, results, logger=None, **kwargs):
        """评估自车轨迹的1s/2s/3s ADE/FDE和碰撞率指标"""

        time_horizons = [1, 2, 3]
        metrics = {
            'L2_vad': {t: {'sum': 0.0, 'count': 0} for t in time_horizons},
            'L2_uniad': {t: {'sum': 0.0, 'count': 0} for t in time_horizons},
            'col_vad': {t: {'sum': 0.0, 'count': 0} for t in time_horizons},
            'col_uniad': {t: {'sum': 0.0, 'count': 0} for t in time_horizons}
        }

        print('\nEvaluate Planning...')
        prog_bar = mmcv.ProgressBar(len(results))

        for idx, res in enumerate(results):

            l2_vad = [self.to_scalar(res['plan_L2_vad_1s']), self.to_scalar(res['plan_L2_vad_2s']),
                      self.to_scalar(res['plan_L2_vad_3s'])]
            col_vad = [self.to_scalar(res['plan_col_vad_1s']), self.to_scalar(res['plan_col_vad_2s']),
                       self.to_scalar(res['plan_col_vad_3s'])]
            l2_uniad = [self.to_scalar(res['plan_L2_uniad_1s']), self.to_scalar(res['plan_L2_uniad_2s']),
                        self.to_scalar(res['plan_L2_uniad_3s'])]
            col_uniad = [self.to_scalar(res['plan_col_uniad_1s']), self.to_scalar(res['plan_col_uniad_2s']),
                         self.to_scalar(res['plan_col_uniad_3s'])]

            for t in time_horizons:
                metrics['L2_vad'][t]['sum'] += l2_vad[t-1]
                metrics['L2_vad'][t]['count'] += 1
                metrics['col_vad'][t]['sum'] += col_vad[t-1]
                metrics['col_vad'][t]['count'] += 1
                # VAD&STP3 set invalid case to 0, UniAD skips invalid case
                if l2_uniad[0] > -0.5:
                    metrics['L2_uniad'][t]['sum'] += l2_uniad[t-1]
                    metrics['L2_uniad'][t]['count'] += 1
                    metrics['col_uniad'][t]['sum'] += col_uniad[t-1]
                    metrics['col_uniad'][t]['count'] += 1

            prog_bar.update()

        result_dict = {'plan_num_samples': metrics['L2_vad'][1]['count'],
                       'plan_num_vaild_samples': metrics['L2_uniad'][1]['count']}

        for t in time_horizons:
            result_dict[f'plan_l2_vad_{t}s'] = self.safe_divide(metrics['L2_vad'][t]['sum'],
                                                                metrics['L2_vad'][t]['count'])
            result_dict[f'plan_col_vad_{t}s'] = self.safe_divide(metrics['col_vad'][t]['sum'],
                                                                 metrics['col_vad'][t]['count'])
            result_dict[f'plan_l2_uniad_{t}s'] = self.safe_divide(metrics['L2_uniad'][t]['sum'],
                                                                  metrics['L2_uniad'][t]['count'])
            result_dict[f'plan_col_uniad_{t}s'] = self.safe_divide(metrics['col_uniad'][t]['sum'],
                                                                   metrics['col_uniad'][t]['count'])

        result_dict['plan_l2_vad_mean'] = (result_dict['plan_l2_vad_1s'] + result_dict['plan_l2_vad_2s'] +
                                           result_dict['plan_l2_vad_3s']) / 3
        result_dict['plan_col_vad_mean'] = (result_dict['plan_col_vad_1s'] + result_dict['plan_col_vad_2s'] +
                                            result_dict['plan_col_vad_3s']) / 3
        result_dict['plan_l2_uniad_mean'] = (result_dict['plan_l2_uniad_1s'] + result_dict['plan_l2_uniad_2s'] +
                                             result_dict['plan_l2_uniad_3s']) / 3
        result_dict['plan_col_uniad_mean'] = (result_dict['plan_col_uniad_1s'] + result_dict['plan_col_uniad_2s'] +
                                              result_dict['plan_col_uniad_3s']) / 3

        print('\n-------- PLANNING METRICs (follow UniAD) --------')
        for t in time_horizons:
            print(f"{t}s    L2(FDE)={result_dict[f'plan_l2_uniad_{t}s']:.2f}  COL={result_dict[f'plan_col_uniad_{t}s']:.2%}")
        print(f"MEAN  L2(FDE)={result_dict['plan_l2_uniad_mean']:.2f}  COL={result_dict['plan_col_uniad_mean']:.2%}")

        print('\n-------- PLANNING METRICs (follow VAD/ST-P3) --------')
        for t in time_horizons:
            print(f"{t}s    L2(ADE)={result_dict[f'plan_l2_vad_{t}s']:.2f}  COL={result_dict[f'plan_col_vad_{t}s']:.2%}")
        print(f"MEAN  L2(ADE)={result_dict['plan_l2_vad_mean']:.2f}  COL={result_dict['plan_col_vad_mean']:.2%}")

        return result_dict

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        if isinstance(results, dict):
            # print(f'results must be a list, but get dict, keys={results.keys()}')
            # assert isinstance(results, list)
            results = results['bbox_results']
        assert isinstance(results, list)
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                if name == 'metric_results':
                    continue
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         map_metric='chamfer',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        detail = dict()
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root,
                             verbose=False)

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.custom_eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']

        from .map_utils.mean_ap import eval_map
        from .map_utils.mean_ap import format_res_gt_by_classes
        result_path = osp.abspath(result_path)

        print('Formating results & gts by classes')
        pred_results = mmcv.load(result_path)
        map_results = pred_results['map_results']
        gt_anns = mmcv.load(self.map_ann_file)
        map_annotations = gt_anns['GTs']
        cls_gens, cls_gts = format_res_gt_by_classes(result_path,
                                                     map_results,
                                                     map_annotations,
                                                     cls_names=self.MAPCLASSES,
                                                     num_pred_pts_per_instance=self.fixed_num,
                                                     eval_use_same_gt_sample_num_flag=self.eval_use_same_gt_sample_num_flag,
                                                     pc_range=self.pc_range)
        map_metrics = map_metric if isinstance(map_metric, list) else [map_metric]
        allowed_metrics = ['chamfer', 'iou']
        for metric in map_metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        for metric in map_metrics:
            print('-*' * 10 + f'use metric:{metric}' + '-*' * 10)
            if metric == 'chamfer':
                thresholds = [0.5, 1.0, 1.5]
            elif metric == 'iou':
                thresholds = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            cls_aps = np.zeros((len(thresholds), self.NUM_MAPCLASSES))
            for i, thr in enumerate(thresholds):
                print('-*' * 10 + f'threshhold:{thr}' + '-*' * 10)
                mAP, cls_ap = eval_map(
                    map_results,
                    map_annotations,
                    cls_gens,
                    cls_gts,
                    threshold=thr,
                    cls_names=self.MAPCLASSES,
                    logger=logger,
                    num_pred_pts_per_instance=self.fixed_num,
                    pc_range=self.pc_range,
                    metric=metric)
                for j in range(self.NUM_MAPCLASSES):
                    cls_aps[i, j] = cls_ap[j]['ap']
            for i, name in enumerate(self.MAPCLASSES):
                print('{}: {}'.format(name, cls_aps.mean(0)[i]))
                detail['NuscMap_{}/{}_AP'.format(metric, name)] = cls_aps.mean(0)[i]
            print('map: {}'.format(cls_aps.mean(0).mean()))
            detail['NuscMap_{}/mAP'.format(metric)] = cls_aps.mean(0).mean()
            for i, name in enumerate(self.MAPCLASSES):
                for j, thr in enumerate(thresholds):
                    if metric == 'chamfer':
                        detail['NuscMap_{}/{}_AP_thr_{}'.format(metric, name, thr)] = cls_aps[j][i]
                    elif metric == 'iou':
                        if thr == 0.5 or thr == 0.75:
                            detail['NuscMap_{}/{}_AP_thr_{}'.format(metric, name, thr)] = cls_aps[j][i]

        return detail
