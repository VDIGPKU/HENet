import copy
import numpy as np
import os
from os import path as osp
import torch
import random
import json, pickle
import tempfile
import cv2
import laspy

from .builder import DATASETS

# from mmcv.utils import save_tensor
from mmcv.parallel import DataContainer as DC

from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes

from mmcv.fileio.io import load, dump
from mmcv.utils import track_iter_progress, mkdir_or_exist

from mmdet3d.core.bbox.structures.nuscenes_box import CustomNuscenesBox

from mmdet.datasets.pipelines import to_tensor

from .custom_3d import Custom3DDataset
from .pipelines import Compose

from .nuscenes_styled_eval_utils import DetectionMetrics, EvalBoxes, DetectionBox, center_distance, accumulate, DetectionMetricDataList, calc_ap, calc_tp, quaternion_yaw

from prettytable import PrettyTable
from .B2D_e2e_dataset import B2D_E2E_Dataset


import random
import math
import os
from os import path as osp
import cv2
import tempfile
import copy
import prettytable
import pickle
import json

# np.set_printoptions(suppress=True)
from torch.utils.data import Dataset
import pyquaternion
from pyquaternion import Quaternion
from shapely.geometry import LineString

from .evaluation.detection.nuscenes_styled_eval_utils import (
    DetectionMetrics, 
    EvalBoxes, 
    DetectionBox,
    center_distance,
    accumulate,
    DetectionMetricDataList,
    calc_ap, 
    calc_tp, 
    quaternion_yaw,
)


import mmcv
from mmcv.utils import print_log
# from mmdet.datasets import DATASETS
# from mmdet.datasets.pipelines import Compose
from .utils_b2d import (
    draw_lidar_bbox3d_on_img,
    draw_lidar_bbox3d_on_bev,
)


NameMapping = {
    # =================vehicle=================
    # bicycle
    'vehicle.bh.crossbike': 'bicycle',
    "vehicle.diamondback.century": 'bicycle',
    "vehicle.gazelle.omafiets": 'bicycle',
    # car
    "vehicle.audi.etron": 'car',
    "vehicle.chevrolet.impala": 'car',
    "vehicle.dodge.charger_2020": 'car',
    "vehicle.dodge.charger_police": 'car',
    "vehicle.dodge.charger_police_2020": 'car',
    "vehicle.lincoln.mkz_2017": 'car',
    "vehicle.lincoln.mkz_2020": 'car',
    "vehicle.mini.cooper_s_2021": 'car',
    "vehicle.mercedes.coupe_2020": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.nissan.patrol_2021": 'car',
    "vehicle.audi.tt": 'car',
    "vehicle.audi.etron": 'car',
    "vehicle.ford.crown": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.tesla.model3": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": 'car',
    # bus
    # van
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": "van",
    "vehicle.ford.ambulance": "van",
    # truck
    "vehicle.carlamotors.firetruck": 'truck',
    # =========================================

    # =================traffic sign============
    # traffic.speed_limit
    "traffic.speed_limit.30": 'traffic_sign',
    "traffic.speed_limit.40": 'traffic_sign',
    "traffic.speed_limit.50": 'traffic_sign',
    "traffic.speed_limit.60": 'traffic_sign',
    "traffic.speed_limit.90": 'traffic_sign',
    "traffic.speed_limit.120": 'traffic_sign',

    "traffic.stop": 'traffic_sign',
    "traffic.yield": 'traffic_sign',
    "traffic.traffic_light": 'traffic_light',
    # =========================================

    # ===================Construction===========
    "static.prop.warningconstruction": 'traffic_cone',
    "static.prop.warningaccident": 'traffic_cone',
    "static.prop.trafficwarning": "traffic_cone",

    # ===================Construction===========
    "static.prop.constructioncone": 'traffic_cone',

    # =================pedestrian==============
    "walker.pedestrian.0001": 'pedestrian',
    "walker.pedestrian.0003": 'pedestrian',
    "walker.pedestrian.0004": 'pedestrian',
    "walker.pedestrian.0005": 'pedestrian',
    "walker.pedestrian.0007": 'pedestrian',
    "walker.pedestrian.0010": 'pedestrian',
    "walker.pedestrian.0013": 'pedestrian',
    "walker.pedestrian.0014": 'pedestrian',
    "walker.pedestrian.0015": 'pedestrian',
    "walker.pedestrian.0016": 'pedestrian',
    "walker.pedestrian.0017": 'pedestrian',
    "walker.pedestrian.0018": 'pedestrian',
    "walker.pedestrian.0019": 'pedestrian',
    "walker.pedestrian.0020": 'pedestrian',
    "walker.pedestrian.0021": 'pedestrian',
    "walker.pedestrian.0022": 'pedestrian',
    "walker.pedestrian.0025": 'pedestrian',
    "walker.pedestrian.0027": 'pedestrian',
    "walker.pedestrian.0030": 'pedestrian',
    "walker.pedestrian.0031": 'pedestrian',
    "walker.pedestrian.0032": 'pedestrian',
    "walker.pedestrian.0034": 'pedestrian',
    "walker.pedestrian.0035": 'pedestrian',
    "walker.pedestrian.0041": 'pedestrian',
    "walker.pedestrian.0042": 'pedestrian',
    "walker.pedestrian.0046": 'pedestrian',
    "walker.pedestrian.0047": 'pedestrian',

    # ==========================================
    "static.prop.dirtdebris01": 'others',
    "static.prop.dirtdebris02": 'others',
}

@DATASETS.register_module()
class B2D_Occ_Dataset(Custom3DDataset):
    def __init__(self, queue_length=4, bev_size=(200, 200),overlap_test=False,with_velocity=True,sample_interval=5,name_mapping=NameMapping, eval_cfg = None, map_root =None,map_file=None,past_frames=4, future_frames=4,predict_frames=12,planning_frames=6,patch_size = [102.4, 102.4],point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] ,occ_receptive_field=3,occ_n_future=6,occ_filter_invalid_sample=False,occ_filter_by_valid_flag=False,eval_mod=None,multi_adj_frame_id_cfg=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.queue_length = queue_length
        self.bev_size = (200, 200)
        self.overlap_test = overlap_test
        self.with_velocity = with_velocity
        self.NameMapping  = name_mapping
        self.eval_cfg  = eval_cfg
        self.sample_interval = sample_interval
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.predict_frames = predict_frames
        self.planning_frames = planning_frames
        self.map_root = map_root
        self.map_file = map_file
        self.point_cloud_range = np.array(point_cloud_range)
        self.patch_size = patch_size
        self.occ_receptive_field = occ_receptive_field  # past + current
        self.occ_n_future = occ_n_future  # future only
        self.occ_filter_invalid_sample = occ_filter_invalid_sample
        self.occ_filter_by_valid_flag = occ_filter_by_valid_flag
        self.occ_only_total_frames = 7  # NOTE: hardcode, not influenced by planning   
        self.eval_mod = eval_mod     
        self.map_element_class = {'Broken':0, 'Solid':1, 'SolidSolid':2,'Center':3,'TrafficLight':4,'StopSign':5}

        with open(self.map_file,'rb') as f: 
            self.map_infos = pickle.load(f)
        
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg

    def invert_pose(self, pose):
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
        return inv_pose

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        # import ipdb; ipdb.set_trace()
        return example
    
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]

        for i in range(len(info['gt_names'])):
            if info['gt_names'][i] in self.NameMapping.keys():
                info['gt_names'][i] = self.NameMapping[info['gt_names'][i]]


        gt_masks,gt_labels,gt_bboxes = self.get_map_info(index)


        input_dict = dict(
            folder=info['folder'],
            scene_token=info['folder'],
            frame_idx=info['frame_idx'],
            ego_yaw=np.nan_to_num(info['ego_yaw'],nan=np.pi/2),
            ego_translation=info['ego_translation'],
            sensors=info['sensors'],
            world2lidar=info['sensors']['LIDAR_TOP']['world2lidar'],
            gt_ids=info['gt_ids'],
            gt_boxes=info['gt_boxes'],
            gt_names=info['gt_names'],
            ego_vel = info['ego_vel'],
            ego_accel = info['ego_accel'],
            ego_rotation_rate = info['ego_rotation_rate'],
            npc2world = info['npc2world'],
            gt_lane_labels=gt_labels,
            gt_lane_bboxes=gt_bboxes,
            gt_lane_masks=gt_masks,
            timestamp=info['frame_idx']/10

        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
            for sensor_type, cam_info in info['sensors'].items():
                if not 'CAM' in sensor_type:
                    continue
                image_paths.append(osp.join(self.data_root,cam_info['data_path']))
                # obtain lidar to image transformation matrix
                cam2ego = cam_info['cam2ego']
                intrinsic = cam_info['intrinsic']
                intrinsic_pad = np.eye(4)
                intrinsic_pad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2cam = self.invert_pose(cam2ego) @ lidar2ego
                lidar2img = intrinsic_pad @ lidar2cam
                lidar2img_rts.append(lidar2img)
                cam_intrinsics.append(intrinsic_pad)
                lidar2cam_rts.append(lidar2cam)
            ego2world = np.eye(4)
            ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=input_dict['ego_yaw']).rotation_matrix
            ego2world[0:3,3] = input_dict['ego_translation']
            lidar2global = ego2world @ lidar2ego
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    l2g_r_mat=lidar2global[0:3,0:3],
                    l2g_t=lidar2global[0:3,3]

                ))
        
        # curr: should include ['cams'][cam_name]['data_path'] and ['sensors'][cam_name]['intrinsic']
        # put cam info to ['curr']
        # cams:
        input_dict['cams'] = self.get_cams(info)  # add cam info to input_dict['cams']

        # curr 指向目前整个 input_dict
        input_dict['curr'] = input_dict

        annos = self.get_ann_info(index)  # bbox
        input_dict['ann_info'] = annos
        # import ipdb; ipdb.set_trace()
        # ego_his_trajs = 没有
        ego_fut_trajs = annos['gt_sdc_fut_traj']
        ego_fut_masks = annos['gt_sdc_fut_traj_mask']
        gt_fut_trajs_abs = annos['gt_fut_traj']

        gt_boxes = input_dict['gt_boxes']  # (n, 9)
        offsets = gt_boxes[:, :2]  # (n, 2)
        # 先把相对位移转换成累计位移
        cum_displacement = np.cumsum(gt_fut_trajs_abs, axis=1)  # (16, 12, 2)

        # 把初始绝对位置加到累计位移上，得到绝对轨迹
        gt_fut_trajs_abs = cum_displacement + offsets[:, None, :]
        # import ipdb; ipdb.set_trace()
        padded_array = np.zeros((200, 12, 2), dtype=gt_fut_trajs_abs.dtype)
        n = gt_fut_trajs_abs.shape[0]
        padded_array[:n, :, :] = gt_fut_trajs_abs

        input_dict['ego_fut_trajs'] = ego_fut_trajs
        input_dict['ego_fut_masks'] = ego_fut_masks
        input_dict['gt_fut_trajs_abs'] = padded_array  # agents

        # gt_boxes = input_dict['ann_info']['gt_bboxes_3d']
        gt_boxes = input_dict['ann_info']['gt_bboxes_3d'].tensor.cpu().numpy()
        input_dict['gt_bboxes_3d'] = gt_boxes

        gt_labels = input_dict['ann_info']['gt_labels_3d']
        input_dict['gt_labels_3d'] = gt_labels

        input_dict['ann_infos'] = [gt_boxes, gt_labels]
        # import matplotlib.pyplot as plt

        # plt.figure(figsize=(6, 6))
        # plt.plot(ego_fut_trajs[0, :, 0], ego_fut_trajs[0, :, 1], marker='o', linestyle='-')
        # plt.title("GT SDC Future Trajectory")
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.axis('equal')
        # plt.grid(True)

        # output_path = "/data/ego_fut_trajs.png"
        # plt.savefig(output_path)
        """
        """
        # def get_box_corners(x, y, x_size, y_size, yaw):
        #     """返回一个box的4个角点，按顺时针顺序"""
        #     # 以中心为原点的四个角
        #     dx = x_size / 2
        #     dy = y_size / 2
        #     corners = np.array([
        #         [ dx,  dy],
        #         [-dx,  dy],
        #         [-dx, -dy],
        #         [ dx, -dy],
        #     ])
        #     # 旋转
        #     rot = np.array([[np.cos(yaw), -np.sin(yaw)],
        #                     [np.sin(yaw),  np.cos(yaw)]])
        #     rotated = corners @ rot.T
        #     # 平移到全局坐标
        #     rotated[:, 0] += x
        #     rotated[:, 1] += y
        #     return rotated

        # plt.figure(figsize=(6, 6))
        # for box in gt_boxes:
        #     x, y, z, x_size, y_size, z_size, yaw = box[:7]
        #     corners = get_box_corners(x, y, x_size, y_size, yaw)
        #     # 闭合框
        #     corners = np.vstack([corners, corners[0]])
        #     plt.plot(corners[:, 0], corners[:, 1], '-b')
        #     plt.scatter(x, y, c='r', s=10)  # 中心点

        # plt.axis('equal')
        # plt.xlabel("X (m)")
        # plt.ylabel("Y (m)")
        # plt.title("BEV GT Boxes")
        # plt.grid(True)
        # plt.savefig("/data/bev_gt_boxes.png")
        """
        """


        # plt.figure(figsize=(6, 6))
        # for traj in gt_fut_trajs_abs:
        #     plt.plot(traj[:, 0], traj[:, 1], marker='o', linestyle='-')

        # plt.title("GT Future Trajectories (Abs)")
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.axis('equal')
        # plt.xlim([-100, 100])
        # plt.grid(True)

        # output_path_multi = "/data/gt_fut_trajs_abs.png"
        # plt.savefig(output_path_multi)

        yaw = input_dict['ego_yaw']
        rotation = list(Quaternion(axis=[0, 0, 1], radians=yaw))
        if yaw < 0:
            yaw += 2*np.pi
        yaw_in_degree = yaw / np.pi * 180 
        
        can_bus = np.zeros(18)
        can_bus[:3] = input_dict['ego_translation']
        can_bus[3:7] = rotation
        can_bus[7:10] = input_dict['ego_vel']
        can_bus[10:13] = input_dict['ego_accel']
        can_bus[13:16] = input_dict['ego_rotation_rate']
        can_bus[16] = yaw
        can_bus[17] = yaw_in_degree
        input_dict['can_bus'] = can_bus
        all_frames = []
        for adj_idx in range(index-self.occ_receptive_field+1,index+self.occ_n_future+1):
            if adj_idx<0 or adj_idx>=len(self.data_infos):
                all_frames.append(-1)
            elif self.data_infos[adj_idx]['folder'] != self.data_infos[index]['folder']:
                all_frames.append(-1)
            else: 
                all_frames.append(adj_idx)
            
        future_frames = all_frames[self.occ_receptive_field-1:]
        input_dict['occ_has_invalid_frame'] = (-1 in all_frames[:self.occ_only_total_frames])
        input_dict['occ_img_is_valid'] = np.array(all_frames) >= 0
        occ_future_ann_infos = []
        for future_frame in future_frames:
            if future_frame >= 0:
                occ_future_ann_infos.append(
                    self.get_ann_boxes_only(future_frame),
                )
            else:
                occ_future_ann_infos.append(None)
        input_dict['occ_future_ann_infos'] = occ_future_ann_infos

        input_dict.update(self.occ_get_transforms(future_frames))
        sdc_planning, sdc_planning_mask = self.get_ego_future_xy(index,self.sample_interval,self.planning_frames)
        input_dict['sdc_planning'] = sdc_planning
        input_dict['sdc_planning_mask'] = sdc_planning_mask
        command = info['command_near']
        if command < 0:
            command = 4
        command -= 1
        input_dict['command'] = command

        # Lidar
        # laz_path = info['lidar_path']
        # with laspy.open(laz_path) as f:
        #     laz = f.read()
        # # print(laz.point_format.dimension_names)
        # fields = ('X', 'Y', 'Z', 'classification')
        # data_list = [getattr(laz, field) for field in fields]
        # points = np.stack(data_list, axis=-1)  # shape: [num_points, len(fields)]

        # input_dict['lidar'] = points
        # radar = info['radar_data']
        # input_dict['radar'] = radar


        # History Info.
        info_adj_list = self.get_adj_info(info, index)
        input_dict.update(dict(adjacent=info_adj_list))
        # import ipdb; ipdb.set_trace()
        input_dict['sweeps'] = {'prev': [], 'next': []}  # empty sweeps

        return input_dict

    def get_cams(self, info):
        cams = {}  # a dict to store camera information
        cam_keys = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        for cam_name in cam_keys:
            cam_info = info['sensors'].get(cam_name)
            if cam_info is not None:
                # sensor2ego
                cam2ego = cam_info['cam2ego']       # 4x4
                R = cam2ego[:3, :3]
                t = cam2ego[:3, 3]
                q = Quaternion(matrix=R)
                sensor2ego_rotation = [q.w, q.x, q.y, q.z]
                sensor2ego_translation = t.tolist()

                # ego2global
                world2cam = cam_info['world2cam']
                ego2global_mat = np.linalg.inv(world2cam)
                R_g = ego2global_mat[:3, :3]
                t_g = ego2global_mat[:3, 3]

                U, _, Vt = np.linalg.svd(R_g)
                R_g_orth = U @ Vt

                q_g = Quaternion(matrix=R_g_orth)
                ego2global_rotation = [q_g.w, q_g.x, q_g.y, q_g.z]
                ego2global_translation = t_g.tolist()

                cams[cam_name] = {
                    'data_path': os.path.join('./data/bench2drive', cam_info['data_path']),
                    'cam_intrinsic': cam_info['intrinsic'],
                    'sensor2ego_rotation': sensor2ego_rotation,
                    'sensor2ego_translation': sensor2ego_translation,
                    'ego2global_rotation': ego2global_rotation,
                    'ego2global_translation': ego2global_translation
                }

        return cams
        
    def get_adj_info(self, info, index):
        info_adj_list = []
        adj_id_list = list(range(*self.multi_adj_frame_id_cfg))  # when multi_adj_frame_id_cfg=(1, 9, 1), len of list is 8
        assert self.multi_adj_frame_id_cfg[0] == 1
        assert self.multi_adj_frame_id_cfg[2] == 1
        adj_id_list.append(self.multi_adj_frame_id_cfg[1])  # len becomes 9
        for select_id in adj_id_list:
            select_id = min(max(index - select_id, 0), len(self.data_infos) - 1)
            if not self.data_infos[select_id]['folder'] == info['folder']:
                cams_dict = {}
                cams_dict['cams'] = self.get_cams(info)
                info_adj_list.append(cams_dict)
            else:
                cams_dict = {}
                cams_dict['cams'] = self.get_cams(self.data_infos[select_id])
                info_adj_list.append(cams_dict)

        return info_adj_list  # a list of dicts, each dict contains camera information for the corresponding frame (adjacent frames)


    def get_map_info(self, index):

        gt_masks = []
        gt_labels = []
        gt_bboxes = []

        ann_info = self.data_infos[index]
        town_name = ann_info['town_name']
        map_info = self.map_infos[town_name]
        lane_points = map_info['lane_points']
        lane_sample_points = map_info['lane_sample_points']
        lane_types = map_info['lane_types']
        trigger_volumes_points = map_info['trigger_volumes_points']
        trigger_volumes_sample_points = map_info['trigger_volumes_sample_points']
        trigger_volumes_types = map_info['trigger_volumes_types']
        world2lidar = np.array(ann_info['sensors']['LIDAR_TOP']['world2lidar'])
        ego_xy = np.linalg.inv(world2lidar)[0:2,3]

        #1st search
        max_distance = 100
        chosed_idx = []
        for idx in range(len(lane_sample_points)):
            single_sample_points = lane_sample_points[idx]
            distance = np.linalg.norm((single_sample_points[:,0:2]-ego_xy),axis=-1)
            if np.min(distance) < max_distance:
                chosed_idx.append(idx)

        for idx in chosed_idx:
            if not lane_types[idx] in self.map_element_class.keys():
                continue
            points = lane_points[idx]
            points = np.concatenate([points,np.ones((points.shape[0],1))],axis=-1)
            points_in_ego = (world2lidar @ points.T).T
            #print(points_in_ego)
            mask = (points_in_ego[:,0]>self.point_cloud_range[0]) & (points_in_ego[:,0]<self.point_cloud_range[3]) & (points_in_ego[:,1]>self.point_cloud_range[1]) & (points_in_ego[:,1]<self.point_cloud_range[4])
            points_in_ego_range = points_in_ego[mask,0:2]
            if len(points_in_ego_range) > 1:
                gt_mask = np.zeros(self.bev_size,dtype=np.uint8)
                normalized_points = np.zeros_like(points_in_ego_range)
                normalized_points[:,0] = (points_in_ego_range[:,0] + self.patch_size[0]/2)*(self.bev_size[0]/self.patch_size[0])
                normalized_points[:,1] = (points_in_ego_range[:,1] + self.patch_size[1]/2)*(self.bev_size[1]/self.patch_size[1])
                cv2.polylines(gt_mask, [normalized_points.astype(np.int32)], False, color=1, thickness=2)
                gt_label =  self.map_element_class[lane_types[idx]]
                gt_masks.append(gt_mask)
                gt_labels.append(gt_label)
                ys, xs = np.where(gt_mask==1)
                gt_bboxes.append([min(xs), min(ys), max(xs), max(ys)]) 

        for idx in range(len(trigger_volumes_points)):
            if not trigger_volumes_types[idx] in self.map_element_class.keys():
                continue
            points = trigger_volumes_points[idx]
            points = np.concatenate([points,np.ones((points.shape[0],1))],axis=-1)
            points_in_ego = (world2lidar @ points.T).T
            mask = (points_in_ego[:,0]>self.point_cloud_range[0]) & (points_in_ego[:,0]<self.point_cloud_range[3]) & (points_in_ego[:,1]>self.point_cloud_range[1]) & (points_in_ego[:,1]<self.point_cloud_range[4])
            points_in_ego_range = points_in_ego[mask,0:2]
            if mask.all():
                gt_mask = np.zeros(self.bev_size,dtype=np.uint8)
                normalized_points = np.zeros_like(points_in_ego_range)
                normalized_points[:,0] = (points_in_ego_range[:,0] + self.patch_size[0]/2)*(self.bev_size[0]/self.patch_size[0])
                normalized_points[:,1] = (points_in_ego_range[:,1] + self.patch_size[1]/2)*(self.bev_size[1]/self.patch_size[1])
                cv2.fillConvexPoly(gt_mask, normalized_points.astype(np.int32), color=1)
                gt_label = self.map_element_class[trigger_volumes_types[idx]]
                gt_masks.append(gt_mask)
                gt_labels.append(gt_label)
                ys, xs = np.where(gt_mask==1)
                gt_bboxes.append([min(xs), min(ys), max(xs), max(ys)]) 

        if len(gt_masks) == 0:
            gt_masks.append(np.zeros(self.bev_size,dtype=np.uint8))
            gt_labels.append(-1)
            gt_bboxes.append([0,0,0,0])

        gt_masks = np.stack(gt_masks)
        gt_labels = np.array(gt_labels)
        gt_bboxes = np.array(gt_bboxes)

        return gt_masks,gt_labels,gt_bboxes


    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points

        for i in range(len(info['gt_names'])):
            if info['gt_names'][i] in self.NameMapping.keys():
                info['gt_names'][i] = self.NameMapping[info['gt_names'][i]]
        mask = (info['num_points'] >= -1)
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_inds = info['gt_ids']
        gt_labels_3d = []

        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        if not self.with_velocity:
            gt_bboxes_3d = gt_bboxes_3d[:,0:7]
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        ego_future_track, ego_future_mask = self.get_ego_future_xy(index,self.sample_interval,self.predict_frames)
        past_track, past_mask = self.get_past_or_future_xy(index,self.sample_interval,self.past_frames,past_or_future='past',local_xy=True)
        predict_track, predict_mask = self.get_past_or_future_xy(index,self.sample_interval,self.predict_frames,past_or_future='future',local_xy=False)
        mask = (past_mask.sum((1,2))>0).astype(np.int64)
        future_track = predict_track[:,0:self.future_frames,:]*mask[:,None,None]
        future_mask = predict_mask[:,0:self.future_frames,:]*mask[:,None,None]
        full_past_track = np.concatenate([past_track,future_track],axis=1)
        full_past_mask = np.concatenate([past_mask,future_mask],axis=1)
        gt_sdc_bbox, gt_sdc_label =self.generate_sdc_info(index)
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_inds=gt_inds,
            gt_fut_traj=predict_track,
            gt_fut_traj_mask=predict_mask,
            gt_past_traj=full_past_track,
            gt_past_traj_mask=full_past_mask,
            gt_sdc_bbox=gt_sdc_bbox,
            gt_sdc_label=gt_sdc_label,
            gt_sdc_fut_traj=ego_future_track[:,:,0:2],
            gt_sdc_fut_traj_mask=ego_future_mask,
            )
        return anns_results

    def get_ann_boxes_only(self, index):

        info = self.data_infos[index]
        for i in range(len(info['gt_names'])):
            if info['gt_names'][i] in self.NameMapping.keys():
                info['gt_names'][i] = self.NameMapping[info['gt_names'][i]]
        gt_bboxes_3d = info['gt_boxes']
        gt_names_3d = info['gt_names']
        gt_inds = info['gt_ids']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        if not self.with_velocity:
            gt_bboxes_3d = gt_bboxes_3d[:,0:7]
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        boxes_annos = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_inds=gt_inds,
            )
        return boxes_annos

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
        
    def generate_sdc_info(self,idx):

        info = self.data_infos[idx]
        ego_size = info['ego_size']
        ego_vel = info['ego_vel']
        psudo_sdc_bbox = np.array([0.0, 0.0, 0.0, ego_size[0], ego_size[1], ego_size[2], -np.pi, ego_vel[1], ego_vel[0] ])
        if not self.with_velocity:
            psudo_sdc_bbox = psudo_sdc_bbox[0:7]
        gt_bboxes_3d = np.array([psudo_sdc_bbox]).astype(np.float32)
        gt_names_3d = ['car']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
  
        gt_labels_3d = DC(to_tensor(gt_labels_3d))
        gt_bboxes_3d = DC(gt_bboxes_3d, cpu_only=True)

        return gt_bboxes_3d, gt_labels_3d

    def get_past_or_future_xy(self,idx,sample_rate,frames,past_or_future,local_xy=False):

        assert past_or_future in ['past','future']
        if past_or_future == 'past':
            adj_idx_list = range(idx-sample_rate,idx-(frames+1)*sample_rate,-sample_rate)
        else:
            adj_idx_list = range(idx+sample_rate,idx+(frames+1)*sample_rate,sample_rate)

        cur_frame = self.data_infos[idx]
        box_ids = cur_frame['gt_ids']
        adj_track = np.zeros((len(box_ids),frames,2))
        adj_mask = np.zeros((len(box_ids),frames,2))
        world2lidar_ego_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for i in range(len(box_ids)):
            box_id = box_ids[i]
            cur_box2lidar = world2lidar_ego_cur @ cur_frame['npc2world'][i]
            cur_xy = cur_box2lidar[0:2,3]      
            for j in range(len(adj_idx_list)):
                adj_idx = adj_idx_list[j]
                if adj_idx <0 or adj_idx>=len(self.data_infos):
                    break
                adj_frame = self.data_infos[adj_idx]
                if adj_frame['folder'] != cur_frame ['folder']:
                    break
                if len(np.where(adj_frame['gt_ids']==box_id)[0])==0:
                    continue
                assert len(np.where(adj_frame['gt_ids']==box_id)[0]) == 1 , np.where(adj_frame['gt_ids']==box_id)[0]
                adj_idx = np.where(adj_frame['gt_ids']==box_id)[0][0]
                adj_box2lidar = world2lidar_ego_cur @ adj_frame['npc2world'][adj_idx]
                adj_xy = adj_box2lidar[0:2,3]    
                if local_xy:
                    adj_xy -= cur_xy
                adj_track[i,j,:] = adj_xy
                adj_mask[i,j,:] = 1
        return adj_track, adj_mask

    def get_ego_future_xy(self,idx,sample_rate,frames):

        adj_idx_list = range(idx+sample_rate,idx+(frames+1)*sample_rate,sample_rate)
        cur_frame = self.data_infos[idx]
        adj_track = np.zeros((1,frames,3))
        adj_mask = np.zeros((1,frames,2))
        world2lidar_ego_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for j in range(len(adj_idx_list)):
            adj_idx = adj_idx_list[j]
            if adj_idx <0 or adj_idx>=len(self.data_infos):
                break
            adj_frame = self.data_infos[adj_idx]
            if adj_frame['folder'] != cur_frame ['folder']:
                break
            world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
            adj2cur_lidar = world2lidar_ego_cur @ np.linalg.inv(world2lidar_ego_adj)
            xy = adj2cur_lidar[0:2,3]
            yaw = np.arctan2(adj2cur_lidar[1,0],adj2cur_lidar[0,0])
            yaw = -yaw -np.pi
            while yaw > np.pi:
                yaw -= np.pi*2
            while yaw < -np.pi:
                yaw += np.pi*2
            adj_track[0,j,0:2] = xy
            adj_track[0,j,2] = yaw
            adj_mask[0,j,:] = 1

        return adj_track, adj_mask

    def occ_get_transforms(self, indices, data_type=torch.float32):

        l2e_r_mats = []
        l2e_t_vecs = []
        e2g_r_mats = []
        e2g_t_vecs = []

        for index in indices:
            if index == -1:
                l2e_r_mats.append(None)
                l2e_t_vecs.append(None)
                e2g_r_mats.append(None)
                e2g_t_vecs.append(None)
            else:
                info = self.data_infos[index]
                lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
                l2e_r = lidar2ego[0:3,0:3]
                l2e_t = lidar2ego[0:3,3]
                ego2global = np.linalg.inv(info['world2ego'])
                e2g_r = ego2global[0:3,0:3]
                e2g_t = ego2global[0:3,3]
                l2e_r_mats.append(torch.tensor(l2e_r).to(data_type))
                l2e_t_vecs.append(torch.tensor(l2e_t).to(data_type))
                e2g_r_mats.append(torch.tensor(e2g_r).to(data_type))
                e2g_t_vecs.append(torch.tensor(e2g_t).to(data_type))
        res = {
            'occ_l2e_r_mats': l2e_r_mats,
            'occ_l2e_t_vecs': l2e_t_vecs,
            'occ_e2g_r_mats': e2g_r_mats,
            'occ_e2g_t_vecs': e2g_t_vecs,
        }

        return res

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        # NOTE:Curremtly we only support evaluation on detection and planning 

        result_files, tmp_dir = self.format_results(results['bbox_results'], jsonfile_prefix)    
        result_path = result_files
        with open(result_path) as f:
            result_data = json.load(f)
        pred_boxes = EvalBoxes.deserialize(result_data['results'], DetectionBox)
        meta = result_data['meta']

        gt_boxes = self.load_gt()

        metric_data_list = DetectionMetricDataList()
        for class_name in self.eval_cfg['class_names']:
            for dist_th in self.eval_cfg['dist_ths']:
                md = accumulate(gt_boxes, pred_boxes, class_name, center_distance, dist_th)
                metric_data_list.set(class_name, dist_th, md)
                metrics = DetectionMetrics(self.eval_cfg)

        for class_name in self.eval_cfg['class_names']:
            # Compute APs.
            for dist_th in self.eval_cfg['dist_ths']:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.eval_cfg['min_recall'], self.eval_cfg['min_precision'])
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in self.eval_cfg['tp_metrics']:
                metric_data = metric_data_list[(class_name, self.eval_cfg['dist_th_tp'])]
                tp = calc_tp(metric_data, self.eval_cfg['min_recall'], metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = meta.copy()
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        #print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err']))        

        detail = dict()
        metric_prefix = 'bbox_NuScenes'
        for name in self.eval_cfg['class_names']:
            for k, v in metrics_summary['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics_summary['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics_summary['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,self.eval_cfg['err_name_maping'][k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics_summary['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics_summary['mean_ap']

        if 'planning_results_computed' in results.keys():
                planning_results_computed = results['planning_results_computed']
                planning_tab = PrettyTable()
                planning_tab.field_names = [
                    "metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
                for key in planning_results_computed.keys():
                    value = planning_results_computed[key]
                    row_value = []
                    row_value.append(key)
                    for i in range(len(value)):
                        row_value.append('%.4f' % float(value[i]))
                    planning_tab.add_row(row_value)
                print(planning_tab)


        return detail

    def load_gt(self):
        all_annotations = EvalBoxes()
        for i in range(len(self.data_infos)):
            sample_boxes = []
            sample_data = self.data_infos[i]

            gt_boxes = sample_data['gt_boxes']
            
            for j in range(gt_boxes.shape[0]):
                class_name = self.NameMapping[sample_data['gt_names'][j]]
                if not class_name in self.eval_cfg['class_range'].keys():
                    continue
                range_x, range_y = self.eval_cfg['class_range'][class_name]
                if abs(gt_boxes[j,0]) > range_x or abs(gt_boxes[j,1]) > range_y:
                    continue
                sample_boxes.append(DetectionBox(
                                                sample_token=sample_data['folder']+'_'+str(sample_data['frame_idx']),
                                                translation=gt_boxes[j,0:3],
                                                size=gt_boxes[j,3:6],
                                                rotation=list(Quaternion(axis=[0, 0, 1], radians=-gt_boxes[j,6]-np.pi/2)),
                                                velocity=gt_boxes[j,7:9],
                                                num_pts=int(sample_data['num_points'][j]),
                                                detection_name=self.NameMapping[sample_data['gt_names'][j]],
                                                detection_score=-1.0,  
                                                attribute_name=self.NameMapping[sample_data['gt_names'][j]]
                                                ))
            all_annotations.add_boxes(sample_data['folder']+'_'+str(sample_data['frame_idx']), sample_boxes)
        return all_annotations
    
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
        for sample_id, det in enumerate(track_iter_progress(results)):
            #pdb.set_trace()
            annos = []
            box3d = det['boxes_3d']
            scores = det['scores_3d']
            labels = det['labels_3d']
            box_gravity_center = box3d.gravity_center
            box_dims = box3d.dims
            box_yaw = box3d.yaw.numpy()
            box_yaw = -box_yaw - np.pi / 2
            sample_token = self.data_infos[sample_id]['folder'] + '_' + str(self.data_infos[sample_id]['frame_idx'])



            for i in range(len(box3d)):
                #import pdb;pdb.set_trace()
                quat = list(Quaternion(axis=[0, 0, 1], radians=box_yaw[i]))
                velocity = [box3d.tensor[i, 7].item(),box3d.tensor[i, 8].item()]
                name = mapped_class_names[labels[i]]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box_gravity_center[i].tolist(),
                    size=box_dims[i].tolist(),
                    rotation=quat,
                    velocity=velocity,
                    detection_name=name,
                    detection_score=scores[i].item(),
                    attribute_name=name)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        dump(nusc_submissions, res_path)
        return res_path  

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
        assert isinstance(results, list), 'results must be a list'
        # assert len(results) == len(self), (
        #     'The length of results is not equal to the dataset len: {} != {}'.
        #     format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir




@DATASETS.register_module()
class B2D3DDataset(Dataset):
    CLASSES = [
        'car',
        'van',
        'truck',
        'bicycle',
        'traffic_sign',
        'traffic_cone',
        'traffic_light',
        'pedestrian',
        'others',
    ]
    MAP_CLASSES = [
        'Broken',
        'Solid',
        'SolidSolid',
        # 'Center',
        # 'TrafficLight',
        # 'StopSign',
    ]
    ID_COLOR_MAP = [
        (59, 59, 238),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (0, 127, 255),
        (71, 130, 255),
        (127, 127, 0),
    ]

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        map_classes=None,
        name_mapping=None,
        load_interval=1,
        modality=None,
        sample_interval=5,
        past_frames=2,
        future_frames=6,
        test_mode=False,
        vis_score_threshold=0.25,
        data_aug_conf=None,
        sequences_split_num=1,
        with_seq_flag=False,
        keep_consistent_seq_aug=True,
        work_dir=None,
        eval_config=None,
        use_cmd=True,
        use_generated_img=False,
        generated_img_root=None,
        box_type_3d=None,
        multi_adj_frame_id_cfg=None
    ):
        self.load_interval = load_interval
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.box_mode_3d = 0
        self.sample_interval = sample_interval
        self.past_frames = past_frames
        self.future_frames = future_frames

        if classes is not None:
            self.CLASSES = classes
        if map_classes is not None: 
            self.MAP_CLASSES = map_classes
        self.NameMapping = name_mapping
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.vis_score_threshold = vis_score_threshold

        self.data_aug_conf = data_aug_conf
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        if with_seq_flag:
            self._set_sequence_group_flag()
        
        self.work_dir = work_dir
        self.eval_config = eval_config
        self.use_cmd = use_cmd

        self.use_generated_img = use_generated_img
        self.generated_img_root = generated_img_root

        self.eval_cfg = {
            "dist_ths": [0.5, 1.0, 2.0, 4.0],
            "dist_th_tp": 2.0,
            "min_recall": 0.1,
            "min_precision": 0.1,
            "mean_ap_weight": 5,
            "class_names":['car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian'],
            "tp_metrics":['trans_err', 'scale_err', 'orient_err', 'vel_err'],
            "err_name_maping":{'trans_err': 'mATE','scale_err': 'mASE','orient_err': 'mAOE','vel_err': 'mAVE','attr_err': 'mAAE'},
            "class_range":{'car':(50,50),'van':(50,50),'truck':(50,50),'bicycle':(40,40),'traffic_sign':(30,30),'traffic_cone':(30,30),'traffic_light':(30,30),'pedestrian':(40,40)}
        }
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        if not self.test_mode:
            self.flag = np.zeros(len(self), dtype=np.uint8)

    def __len__(self):
        return len(self.data_infos)

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        if self.sequences_split_num == -1:
            self.flag = np.arange(len(self.data_infos))
            return
        
        res = []

        curr_folder = self.data_infos[0]["folder"]
        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and self.data_infos[idx]["folder"] != curr_folder:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
                curr_folder = self.data_infos[idx]["folder"]
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.array(
                    range(len(self.data_infos)), dtype=np.int64
                )
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag]
                                    / self.sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.sequences_split_num
                )
                self.flag = np.array(new_flags, dtype=np.int64)

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
            rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            rotate_3d = 0
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config

    def __getitem__(self, idx):
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = self.get_augmentation()
        data = self.get_data_info(idx)
        data["aug_config"] = aug_config
        data = self.pipeline(data)
        return data

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format="pkl")
        data_infos = data[:: self.load_interval]
        return data_infos
    
    def anno2geom(self, annos):
        map_geoms = {}
        for label, anno_list in annos.items():
            map_geoms[label] = []
            for anno in anno_list:
                geom = LineString(anno)
                map_geoms[label].append(geom)
        return map_geoms
    
    def get_data_info(self, index):
        info = self.data_infos[index]
        # import ipdb; ipdb.set_trace()  #########
        input_dict = dict(
            token=info['token'],
            timestamp=info['timestamp'] / 1e6,
        )
        ## only use 3 classes
        map_annos = {
            0: info["map_annos"][0],
            1: info["map_annos"][1],
            2: info["map_annos"][2],
        }
        map_geoms = self.anno2geom(map_annos)
        input_dict["map_infos"] = map_annos
        input_dict["map_geoms"] = map_geoms

        if self.modality['use_camera']:
            image_paths = []
            depth_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsic = []
            lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
            lidar2global =  self.invert_pose(info['sensors']['LIDAR_TOP']['world2lidar'])
            for sensor_type, cam_info in info['sensors'].items():
                if not 'CAM' in sensor_type:
                    continue
                img_path = osp.join(self.data_root,cam_info['data_path'])
                image_paths.append(img_path)
                depth_path = img_path.replace('rgb_','depth_').replace('.jpg','.png')
                depth_paths.append(depth_path)
                # obtain lidar to image transformation matrix
                cam2ego = cam_info['cam2ego']
                intrinsic = copy.deepcopy(cam_info["intrinsic"])
                cam_intrinsic.append(intrinsic)
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2cam = (self.invert_pose(cam2ego) @ lidar2ego).T
                lidar2img = viewpad @ lidar2cam.T
                lidar2img_rts.append(lidar2img)
                lidar2cam_rts.append(lidar2cam)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    depth_filename=depth_paths,  ###################### here!!!!!!!!!!!
                    lidar2img=lidar2img_rts,
                    lidar2cam=lidar2cam_rts,
                    cam_intrinsic=cam_intrinsic,
                    lidar2global=lidar2global,
                )
            )

        annos = self.get_ann_info(index)
        input_dict.update(annos)
        import ipdb; ipdb.set_trace()  #########
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        mask = (info['num_points'] != 0)
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        for i in range(len(gt_names_3d)):
            if gt_names_3d[i] in self.NameMapping.keys():
                gt_names_3d[i] = self.NameMapping[gt_names_3d[i]]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)        

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )

        if "gt_ids" in info:
            instance_inds = np.array(info["gt_ids"], dtype=np.int)[mask]
            anns_results["instance_inds"] = instance_inds

        gt_agent_fut_trajs, gt_agent_fut_masks = self.get_fut_agent(index, self.sample_interval, self.future_frames)
        anns_results['gt_agent_fut_trajs'] = gt_agent_fut_trajs[mask]
        anns_results['gt_agent_fut_masks'] = gt_agent_fut_masks[mask]

        (
            ego_his_trajs, 
            ego_fut_trajs, 
            ego_fut_masks, 
            command
        ) = self.get_ego_trajs(index,self.sample_interval,self.past_frames,self.future_frames)
        anns_results['gt_ego_fut_trajs'] = ego_fut_trajs
        anns_results['gt_ego_fut_masks'] = ego_fut_masks
        anns_results['gt_ego_fut_cmd'] = command
        anns_results['ego_his_trajs'] = ego_his_trajs

        global2lidar = info['sensors']['LIDAR_TOP']['world2lidar']
        tp_near_global = info["command_near_xy"]
        tp_near_global = np.concatenate([info["command_near_xy"], np.array([0, 1])])
        tp_near_local = global2lidar @ tp_near_global
        anns_results["tp_near"] = tp_near_local[:2]

        tp_far_global = info["command_far_xy"]
        tp_far_global = np.concatenate([info["command_far_xy"], np.array([0, 1])])
        tp_far_local = global2lidar @ tp_far_global
        anns_results["tp_far"] = tp_far_local[:2]

        ego_status = np.zeros(10)
        ego_status[:3] = info["ego_accel"]
        ego_status[3:6] = info["ego_rotation_rate"]
        ego_status[6:9] = info["ego_vel"]
        anns_results["ego_status"] = ego_status.astype(np.float32)
        # def dis(a,b):
        #     a1 = a[:2]
        #     a2 = b[:2]
        #     d = a1-a2
        #     dis = np.sqrt(d[0]*d[0]+d[1]*d[1])
        #     return dis

        # if index != 0:
        #     print("near: ", dis(tp_near_global, self.tp_near_global), "far: ", dis(tp_far_global, self.tp_far_global))

        # self.tp_near_global = tp_near_global
        # self.tp_far_global = tp_far_global
        return anns_results

        ## get future box for planning eval
        fut_ts = int(ego_fut_masks.sum())
        fut_boxes = []
        cur_scene_token = info["folder"]
        cur_T_global = get_T_global(info)
        for i in range(1, fut_ts + 1):
            fut_info = self.data_infos[index + i]
            fut_scene_token = fut_info["scene_token"]
            if cur_scene_token != fut_scene_token:
                break
            if self.use_valid_flag:
                mask = fut_info["valid_flag"]
            else:
                mask = fut_info["num_lidar_pts"] > 0

            fut_gt_bboxes_3d = fut_info["gt_boxes"][mask]
            
            fut_T_global = get_T_global(fut_info)
            T_fut2cur = np.linalg.inv(cur_T_global) @ fut_T_global

            center = fut_gt_bboxes_3d[:, :3] @ T_fut2cur[:3, :3].T + T_fut2cur[:3, 3]
            yaw = np.stack([np.cos(fut_gt_bboxes_3d[:, 6]), np.sin(fut_gt_bboxes_3d[:, 6])], axis=-1)
            yaw = yaw @ T_fut2cur[:2, :2].T
            yaw = np.arctan2(yaw[..., 1], yaw[..., 0])

            fut_gt_bboxes_3d[:, :3] = center
            fut_gt_bboxes_3d[:, 6] = yaw
            fut_boxes.append(fut_gt_bboxes_3d)

        anns_results['fut_boxes'] = fut_boxes
        
        return anns_results

    def get_fut_agent(self, idx, sample_rate, frames):
        adj_idx_list = range(idx,idx+(frames+1)*sample_rate,sample_rate)
        cur_frame = self.data_infos[idx]
        cur_boxes = cur_frame['gt_boxes'].copy()
        box_ids = cur_frame['gt_ids']

        future_track = np.zeros((len(box_ids),frames+1,2))
        future_mask = np.zeros((len(box_ids),frames+1))
        world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for i in range(len(box_ids)):
            box_id = box_ids[i]
            cur_box2lidar = world2lidar_lidar_cur @ cur_frame['npc2world'][i]
            cur_xy = cur_box2lidar[0:2,3]
            for j in range(len(adj_idx_list)):
                adj_idx = adj_idx_list[j]
                if adj_idx < 0 or adj_idx >= len(self.data_infos):
                    break
                adj_frame = self.data_infos[adj_idx]
                if adj_frame['folder'] != cur_frame ['folder']:
                    break
                if len(np.where(adj_frame['gt_ids']==box_id)[0])==0:
                    break
                assert len(np.where(adj_frame['gt_ids']==box_id)[0]) == 1 , np.where(adj_frame['gt_ids']==box_id)[0]
                adj_idx = np.where(adj_frame['gt_ids']==box_id)[0][0]
                adj_box2lidar = world2lidar_lidar_cur @ adj_frame['npc2world'][adj_idx]
                adj_xy = adj_box2lidar[0:2,3]
                if j > 0:
                    last_xy = future_track[i,j-1,:]
                    distance = np.linalg.norm(last_xy - adj_xy)
                    if distance > 10:
                        break
                future_track[i,j,:] = adj_xy
                future_mask[i,j] = 1

        future_track_offset = future_track[:,1:,:] - future_track[:,:-1,:]
        future_mask_offset = future_mask[:,1:]
        future_track_offset[future_mask_offset==0] = 0

        return future_track_offset.astype(np.float32), future_mask_offset.astype(np.float32)

    def get_ego_trajs(self,idx,sample_rate,past_frames,future_frames):
        adj_idx_list = range(idx-past_frames*sample_rate,idx+(future_frames+1)*sample_rate,sample_rate)
        cur_frame = self.data_infos[idx]
        full_adj_track = np.zeros((past_frames+future_frames+1,2))
        full_adj_adj_mask = np.zeros(past_frames+future_frames+1)
        world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for j in range(len(adj_idx_list)):
            adj_idx = adj_idx_list[j]
            if adj_idx <0 or adj_idx>=len(self.data_infos):
                break
            adj_frame = self.data_infos[adj_idx]
            if adj_frame['folder'] != cur_frame ['folder']:
                break
            world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
            adj2cur_lidar = world2lidar_lidar_cur @ np.linalg.inv(world2lidar_ego_adj)
            xy = adj2cur_lidar[0:2,3]
            full_adj_track[j,0:2] = xy
            full_adj_adj_mask[j] = 1
        offset_track = full_adj_track[1:] - full_adj_track[:-1]
        for j in range(past_frames-1,-1,-1):
            if full_adj_adj_mask[j] == 0:
                offset_track[j] = offset_track[j+1]
        for j in range(past_frames,past_frames+future_frames,1):

            if full_adj_adj_mask[j+1] == 0 :
                offset_track[j] = 0
        if self.use_cmd:
            command = self.command2hot(cur_frame['command_near'])
        else:
            command = np.array([0, ])
        offset_track = offset_track.astype(np.float32)
        return offset_track[:past_frames].copy(), offset_track[past_frames:].copy(), full_adj_adj_mask[-future_frames:].copy(), command
    
    def command2hot(self,command,max_dim=6):
        if command < 0:
            command = 4
        command -= 1
        cmd_one_hot = np.zeros(max_dim)
        cmd_one_hot[command] = 1
        return cmd_one_hot

    def invert_pose(self, pose):
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
        return inv_pose

    def _format_bbox(self, results, jsonfile_prefix=None, tracking=False):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            box3d = det['boxes_3d']
            scores = det['scores_3d']
            labels = det['labels_3d']
            box_gravity_center = box3d[:, :3]
            box_dims = box3d[:, 3:6]
            box_yaw = box3d[:, 6]
            sample_token = self.data_infos[sample_id]['token']

            for i in range(len(box3d)):
                quat = list(Quaternion(axis=[0, 0, 1], radians=box_yaw[i]))
                velocity = [box3d[i, 7].item(),box3d[i, 8].item()]
                name = mapped_class_names[labels[i]]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box_gravity_center[i].tolist(),
                    size=box_dims[i].tolist(),
                    rotation=quat,
                    velocity=velocity,
                    detection_name=name,
                    detection_score=scores[i].item(),
                    attribute_name=name)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(
        self, result_path, logger=None, result_name="img_bbox", tracking=False
    ):

        with open(result_path) as f:
            result_data = json.load(f)
        pred_boxes = EvalBoxes.deserialize(result_data['results'], DetectionBox)
        meta = result_data['meta']

        gt_boxes = self.load_gt()

        metric_data_list = DetectionMetricDataList()
        for class_name in self.eval_cfg['class_names']:
            for dist_th in self.eval_cfg['dist_ths']:
                md = accumulate(gt_boxes, pred_boxes, class_name, center_distance, dist_th)
                metric_data_list.set(class_name, dist_th, md)
                metrics = DetectionMetrics(self.eval_cfg)

        for class_name in self.eval_cfg['class_names']:
            # Compute APs.
            for dist_th in self.eval_cfg['dist_ths']:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.eval_cfg['min_recall'], self.eval_cfg['min_precision'])
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in self.eval_cfg['tp_metrics']:
                metric_data = metric_data_list[(class_name, self.eval_cfg['dist_th_tp'])]
                tp = calc_tp(metric_data, self.eval_cfg['min_recall'], metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = meta.copy()
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        #print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err']))        

        detail = dict()
        metric_prefix = 'bbox_NuScenes'
        for name in self.eval_cfg['class_names']:
            for k, v in metrics_summary['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics_summary['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics_summary['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,self.eval_cfg['err_name_maping'][k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics_summary['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics_summary['mean_ap']


        return detail

    def load_gt(self):
        all_annotations = EvalBoxes()
        for i in range(len(self.data_infos)):
            sample_boxes = []
            sample_data = self.data_infos[i]

            gt_boxes = sample_data['gt_boxes']
            
            for j in range(gt_boxes.shape[0]):
                class_name = self.NameMapping[sample_data['gt_names'][j]]
                if not class_name in self.eval_cfg['class_range'].keys():
                    continue
                range_x, range_y = self.eval_cfg['class_range'][class_name]
                if abs(gt_boxes[j,0]) > range_x or abs(gt_boxes[j,1]) > range_y:
                    continue
                sample_boxes.append(DetectionBox(
                    sample_token=sample_data['token'],
                    translation=gt_boxes[j,0:3],
                    size=gt_boxes[j,3:6],
                    rotation=list(Quaternion(axis=[0, 0, 1], radians=gt_boxes[j,6])),
                    velocity=gt_boxes[j,7:9],
                    num_pts=int(sample_data['num_points'][j]),
                    detection_name=self.NameMapping[sample_data['gt_names'][j]],
                    detection_score=-1.0,  
                    attribute_name=self.NameMapping[sample_data['gt_names'][j]]
                ))
            all_annotations.add_boxes(sample_data['token'], sample_boxes)
        return all_annotations

    def format_results(self, results, jsonfile_prefix=None, tracking=False):
        assert isinstance(results, list), "results must be a list"

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            result_files = self._format_bbox(
                results, jsonfile_prefix, tracking=tracking
            )
        else:
            result_files = dict()
            for name in results[0]:
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = jsonfile_prefix
                result_files.update(
                    {
                        name: self._format_bbox(
                            results_, tmp_file_, tracking=tracking
                        )
                    }
                )
        return result_files, tmp_dir

    def format_map_results(self, results, prefix=None):
        submissions = {'results': {},}
        
        for j, pred in enumerate(results):
            '''
            For each case, the result should be formatted as Dict{'vectors': [], 'scores': [], 'labels': []}
            'vectors': List of vector, each vector is a array([[x1, y1], [x2, y2] ...]),
                contain all vectors predicted in this sample.
            'scores: List of score(float), 
                contain scores of all instances in this sample.
            'labels': List of label(int), 
                contain labels of all instances in this sample.
            '''
            if pred is None: # empty prediction
                continue
            pred = pred['img_bbox']

            single_case = {'vectors': [], 'scores': [], 'labels': []}
            token = self.data_infos[j]['token']
            for i in range(len(pred['scores'])):
                score = pred['scores'][i]
                label = pred['labels'][i]
                vector = pred['vectors'][i]

                # A line should have >=2 points
                if len(vector) < 2:
                    continue
                
                single_case['vectors'].append(vector)
                single_case['scores'].append(score)
                single_case['labels'].append(label)
            
            submissions['results'][token] = single_case
        
        out_path = osp.join(prefix, 'submission_vector.json')
        print(f'saving submissions results to {out_path}')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mmcv.dump(submissions, out_path)
        return out_path

    def format_motion_results(self, results, jsonfile_prefix=None, tracking=False, thresh=None):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(
                det['img_bbox'], threshold=None
            )
            sample_token = self.data_infos[sample_id]["token"]
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.det3d_eval_configs,
                self.det3d_eval_version,
                filter_with_cls_range=False,
            )
            for i, box in enumerate(boxes):
                if thresh is not None and box.score < thresh:
                    continue
                name = mapped_class_names[box.label]
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = B2D3DDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = B2D3DDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )
                if not tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                else:
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=str(box.token),
                        )
                    )
                nusc_anno.update(
                    dict(
                        trajs=det['img_bbox']['trajs_3d'][i].numpy(),
                    )
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        return nusc_submissions 

    def _evaluate_single_motion(self,
                         results,
                         result_path,
                         logger=None,
                         metric='bbox',
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
        from nuscenes import NuScenes
        from .evaluation.motion.motion_eval_uniad import NuScenesEval as NuScenesEvalMotion

        output_dir = result_path
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEvalMotion(
            nusc,
            config=copy.deepcopy(self.det3d_eval_configs),
            result_path=results,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
            seconds=6)
        metrics = nusc_eval.main(render_curves=False)
        
        MOTION_METRICS = ['EPA', 'min_ade_err', 'min_fde_err', 'miss_rate_err']
        class_names = ['car', 'pedestrian']

        table = prettytable.PrettyTable()
        table.field_names = ["class names"] + MOTION_METRICS
        for class_name in class_names:
            row_data = [class_name]
            for m in MOTION_METRICS:
                row_data.append('%.4f' % metrics[f'{class_name}_{m}'])
            table.add_row(row_data)
        print_log('\n'+str(table), logger=logger)
        return metrics

    def evaluate(
        self,
        results,
        eval_mode,
        metric=None,
        logger=None,
        jsonfile_prefix=None,
        result_names=["img_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        res_path = "results.pkl"
        res_path = osp.join(self.work_dir, res_path)
        print('All Results write to', res_path)
        mmcv.dump(results, res_path)

        results_dict = dict()
        if eval_mode['with_det']:
            self.tracking = eval_mode["with_tracking"]
            self.tracking_threshold = eval_mode["tracking_threshold"]
            for metric in ["detection", "tracking"]:
                tracking = metric == "tracking"
                if tracking and not self.tracking:
                    continue
                result_files, tmp_dir = self.format_results(
                    results, jsonfile_prefix=self.work_dir, tracking=tracking
                )

                if isinstance(result_files, dict):
                    for name in result_names:
                        ret_dict = self._evaluate_single(
                            result_files[name], tracking=tracking
                        )
                    results_dict.update(ret_dict)
                elif isinstance(result_files, str):
                    ret_dict = self._evaluate_single(
                        result_files, tracking=tracking
                    )
                    results_dict.update(ret_dict)

                if tmp_dir is not None:
                    tmp_dir.cleanup()

        if eval_mode['with_map']:
            from .evaluation.map.vector_eval import VectorEvaluate
            self.map_evaluator = VectorEvaluate(self.eval_config)
            result_path = self.format_map_results(results, prefix=self.work_dir)
            map_results_dict = self.map_evaluator.evaluate(result_path, logger=logger)
            results_dict.update(map_results_dict)

        if eval_mode['with_motion']:
            thresh = eval_mode["motion_threshhold"]
            result_files = self.format_motion_results(results, jsonfile_prefix=self.work_dir, thresh=thresh)
            motion_results_dict = self._evaluate_single_motion(result_files, self.work_dir, logger=logger)
            results_dict.update(motion_results_dict)
        
        if eval_mode['with_planning']:
            from .evaluation.planning.planning_eval import planning_eval
            planning_results_dict = planning_eval(results, self.eval_config, logger=logger)
            results_dict.update(planning_results_dict)

        if show or out_dir:
            self.show(results, save_dir=out_dir, show=show, pipeline=pipeline)
        
        # print main metrics for recording
        metric_str = '\n'
        if "img_bbox_NuScenes/NDS" in results_dict:
            metric_str += f'mAP: {results_dict.get("img_bbox_NuScenes/mAP"):.4f}\n'
            metric_str += f'mATE: {results_dict.get("img_bbox_NuScenes/mATE"):.4f}\n'
            metric_str += f'mASE: {results_dict.get("img_bbox_NuScenes/mASE"):.4f}\n'
            metric_str += f'mAOE: {results_dict.get("img_bbox_NuScenes/mAOE"):.4f}\n' 
            metric_str += f'mAVE: {results_dict.get("img_bbox_NuScenes/mAVE"):.4f}\n' 
            metric_str += f'mAAE: {results_dict.get("img_bbox_NuScenes/mAAE"):.4f}\n' 
            metric_str += f'NDS: {results_dict.get("img_bbox_NuScenes/NDS"):.4f}\n\n'
        
        if "img_bbox_NuScenes/amota" in results_dict:
            metric_str += f'AMOTA: {results_dict["img_bbox_NuScenes/amota"]:.4f}\n' 
            metric_str += f'AMOTP: {results_dict["img_bbox_NuScenes/amotp"]:.4f}\n' 
            metric_str += f'RECALL: {results_dict["img_bbox_NuScenes/recall"]:.4f}\n' 
            metric_str += f'MOTAR: {results_dict["img_bbox_NuScenes/motar"]:.4f}\n' 
            metric_str += f'MOTA: {results_dict["img_bbox_NuScenes/mota"]:.4f}\n' 
            metric_str += f'MOTP: {results_dict["img_bbox_NuScenes/motp"]:.4f}\n' 
            metric_str += f'IDS: {results_dict["img_bbox_NuScenes/ids"]}\n\n' 

        if "mAP_normal" in results_dict:
            # metric_str += f'ped_crossing= {results_dict["ped_crossing"]:.4f}\n' 
            # metric_str += f'divider= {results_dict["divider"]:.4f}\n' 
            # metric_str += f'boundary= {results_dict["boundary"]:.4f}\n' 
            metric_str += f'mAP_normal= {results_dict["mAP_normal"]:.4f}\n\n' 

        if "car_EPA" in results_dict:
            metric_str += f'Car / Ped\n' 
            metric_str += f'epa= {results_dict["car_EPA"]:.4f} / {results_dict["pedestrian_EPA"]:.4f}\n'
            metric_str += f'ade= {results_dict["car_min_ade_err"]:.4f} / {results_dict["pedestrian_min_ade_err"]:.4f}\n'
            metric_str += f'fde= {results_dict["car_min_fde_err"]:.4f} / {results_dict["pedestrian_min_fde_err"]:.4f}\n'
            metric_str += f'mr= {results_dict["car_miss_rate_err"]:.4f} / {results_dict["pedestrian_miss_rate_err"]:.4f}\n\n' 

        if "L2" in results_dict:
            metric_str += f'obj_box_col: {(results_dict["obj_box_col"]*100):.3f}%\n'
            metric_str += f'L2: {results_dict["L2"]:.4f}\n\n'
        
        print_log(metric_str, logger=logger)
        return results_dict

    def show(self, results, save_dir=None, show=False, pipeline=None):
        save_dir = "./" if save_dir is None else save_dir
        save_dir = os.path.join(save_dir, "visual")
        print_log(os.path.abspath(save_dir))
        pipeline = Compose(pipeline)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        videoWriter = None

        for i, result in enumerate(results):
            if "img_bbox" in result.keys():
                result = result["img_bbox"]
            data_info = pipeline(self.get_data_info(i))
            imgs = []

            raw_imgs = data_info["img"]
            lidar2img = data_info["img_metas"].data["lidar2img"]
            pred_bboxes_3d = result["boxes_3d"][
                result["scores_3d"] > self.vis_score_threshold
            ]
            if "instance_ids" in result and self.tracking:
                color = []
                for id in result["instance_ids"].cpu().numpy().tolist():
                    color.append(
                        self.ID_COLOR_MAP[int(id % len(self.ID_COLOR_MAP))]
                    )
            elif "labels_3d" in result:
                color = []
                for id in result["labels_3d"].cpu().numpy().tolist():
                    color.append(self.ID_COLOR_MAP[id])
            else:
                color = (255, 0, 0)

            # ===== draw boxes_3d to images =====
            for j, img_origin in enumerate(raw_imgs):
                img = img_origin.copy()
                if len(pred_bboxes_3d) != 0:
                    img = draw_lidar_bbox3d_on_img(
                        pred_bboxes_3d,
                        img,
                        lidar2img[j],
                        img_metas=None,
                        color=color,
                        thickness=3,
                    )
                imgs.append(img)

            # ===== draw boxes_3d to BEV =====
            bev = draw_lidar_bbox3d_on_bev(
                pred_bboxes_3d,
                bev_size=img.shape[0] * 2,
                color=color,
            )

            # ===== put text and concat =====
            for j, name in enumerate(
                [
                    "front",
                    "front right",
                    "front left",
                    "rear",
                    "rear left",
                    "rear right",
                ]
            ):
                imgs[j] = cv2.rectangle(
                    imgs[j],
                    (0, 0),
                    (440, 80),
                    color=(255, 255, 255),
                    thickness=-1,
                )
                w, h = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                text_x = int(220 - w / 2)
                text_y = int(40 + h / 2)

                imgs[j] = cv2.putText(
                    imgs[j],
                    name,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            image = np.concatenate(
                [
                    np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
                    np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
                ],
                axis=0,
            )
            image = np.concatenate([image, bev], axis=1)

            # ===== save video =====
            if videoWriter is None:
                videoWriter = cv2.VideoWriter(
                    os.path.join(save_dir, "video.avi"),
                    fourcc,
                    7,
                    image.shape[:2][::-1],
                )
            cv2.imwrite(os.path.join(save_dir, f"{i}.jpg"), image)
            videoWriter.write(image)
        videoWriter.release()


def get_T_global(info):
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = pyquaternion.Quaternion(
        info["lidar2ego_rotation"]
    ).rotation_matrix
    lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
    ego2global = np.eye(4)
    ego2global[:3, :3] = pyquaternion.Quaternion(
        info["ego2global_rotation"]
    ).rotation_matrix
    ego2global[:3, 3] = np.array(info["ego2global_translation"])
    return ego2global @ lidar2ego



def output_to_nusc_box(detection):
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
    trajs = detection['trajs_3d'].numpy()


    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = CustomNuscenesBox(
            center=box_gravity_center[i],
            size=box_dims[i],
            orientation=quat,
            fut_trajs=trajs[i],
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
        eval_version (str): Evaluation version.
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
        cls_range_x_map = eval_configs.class_range_x
        cls_range_y_map = eval_configs.class_range_y
        x_distance, y_distance = box.center[0], box.center[1]
        det_range_x = cls_range_x_map[classes[box.label]]
        det_range_y = cls_range_y_map[classes[box.label]]
        if abs(x_distance) > det_range_x or abs(y_distance) > det_range_y:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list