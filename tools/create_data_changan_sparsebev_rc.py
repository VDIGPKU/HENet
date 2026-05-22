import json
import functools
import os
import os.path as osp
from collections import defaultdict
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import mmcv
from pyquaternion import Quaternion
import open3d as o3d
from IPython import embed
from copy import deepcopy
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pandas as pd
from math import cos, sin
import pickle

def get_directories(path):
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def get_files_in_directory(directory):
    return [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]


class ChangAn(object):
    # TODO: consistency with nusc
    camera_names = ['cam_front', 'cam_front_left', 'cam_front_right', 'cam_rear', 'cam_rear_left', 'cam_rear_right',
                #    'cam_front_30fov',
                   ]
    radar_names = ["radar_front", "radar_front_corner", "radar_rear_corner"]
    # 'construction_vehicle', ,, 'barrier'
    label_name_map = {
        'cone': 'traffic_cone', 
        'bicycle': 'bicycle', 
        'pedestrian_else': 'pedestrian', 
        'vehicle_else': 'car', 
        'vehicle': 'car', 
        'trolley': 'trailer', 
        'rider': 'motorcycle', 
        'bus': 'bus', 
        'truck': 'truck', 
        'pedestrian': 'pedestrian',
        'tricycle': 'bicycle', 
        'cluster_region': 'barrier', 
        'warningcolumn': 'traffic_cone', 
        'waterhorse': 'barrier', 
        'obstacle_else': 'barrier', 
    }
    # radar_names = ["radar_front"]

    def __init__(self, dataset_root, sweeps_num=5, num_radar_sweeps=10):
        self.dataset_root = dataset_root
        self.sweeps_num = sweeps_num
        self.num_radar_sweeps = num_radar_sweeps
        self.label_set = set()
        # self.data_root = osp.join(self.dataset_root, 'data')
        # self._collect_basic_infos()
        self.data_info_train, self.data_info_val = self.get_info(split_interval=5)
        self.data_info = self.data_info_train
        # print(self.label_set)
    
    def save_data(self, path):
        metadata = dict(
            version=0.1
        )

        with open(osp.join(path,'train_infos_v1.pkl'), 'wb') as fid:
            pickle.dump(dict(infos=self.data_info_train, metadata=metadata), fid)
        
        with open(osp.join(path,'val_infos_v1.pkl'), 'wb') as fid:
            pickle.dump(dict(infos=self.data_info_val, metadata=metadata), fid)
    
    @staticmethod
    def rotate_z(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])
    
    # @staticmethod
    def pastradar2currego(self, past_radars, curr_radar):
        def get_rt(rotation, translation):
            r = Quaternion(rotation).rotation_matrix
            t = np.array(translation)

            rt = np.eye(4)
            rt[:3, :3] = r
            rt[:3, 3] = t

            return rt
        # print(1, past_radars[0]['sensor2lidar_rotation'])
        for past_radar in past_radars:
            # sensor == ego == lidar
            past_ego2global_rt = get_rt(past_radar['ego2global_rotation'], past_radar['ego2global_translation'])
            curr_ego2global_rt = get_rt(curr_radar['ego2global_rotation'], curr_radar['ego2global_translation'])
            sensor2lidar = inv(curr_ego2global_rt) @ past_ego2global_rt

            past_radar['sensor2lidar_translation'] = sensor2lidar[:3, -1]
            past_radar['sensor2lidar_rotation'] = sensor2lidar[:3, :3]
            # print(sensor2lidar)

        # print(2, past_radars[0]['sensor2lidar_rotation'])
        return past_radars
    
    def get_info(self, split_interval=5):
        all_dirs = get_directories(self.dataset_root)
        train_dirs = []
        val_dirs = []
        for i, dirs in enumerate(all_dirs):
            if (i+1)%5 == 0:
                val_dirs.append(dirs)
            else:
                train_dirs.append(dirs)
        print('trainset {}, valset {}'.format(len(train_dirs), len(val_dirs)))
        train_info, train_bad_dirs = self._get_info(train_dirs)
        val_info, val_bad_dirs = self._get_info(val_dirs)

        print('dirs without odometry.txt:')
        # print(train_bad_dirs)
        print('train {}'.format(len(train_bad_dirs)))
        for item in train_bad_dirs:
            print(item)
        
        print('val {}'.format(len(val_bad_dirs)))
        for item in val_bad_dirs:
            print(item)

        return train_info, val_info
    
    def _get_info(self, all_dirs):
        data_info = []
        bad_dirs = []
        bad_files = []
        # for dirs in all_dirs:
        token_id = 0
        for id in mmcv.track_iter_progress(range(len(all_dirs))):
            dirs = all_dirs[id]
            # print(dirs)
            prev_sweep = ''
            prev_radar = dict()
            odometry = dict()

            if not os.path.isfile(osp.join(dirs, 'info_file/GT_raw/odometry.txt')):
                # print(f'\nbad dirs: {dirs}')
                bad_dirs.append(dirs)
                continue

            with open(osp.join(dirs, 'info_file/GT_raw/odometry.txt'), 'r') as file:
                for line in file:
                    line = line.split(' ')
                    odometry[line[0]] = line
            file_names = get_files_in_directory(osp.join(dirs, f'merge_json'))
            file_names = [f.split('.')[0] for f in file_names]
            # sort(file_names)
            file_names.sort()
            # print(file_names[0], file_names[1])

            calib = mmcv.load(osp.join(dirs, f'info_file/calib.yaml'))
            # sync = mmcv.load(osp.join(dirs, f'info_file/sync.json'))["cam_front"]
            sync = mmcv.load(osp.join(dirs, f'info_file/sync.json'))["lidar_pandar128"]

            for file_name in file_names:
                odometry_info = odometry[file_name]
                ego_x, ego_y, ego_z = odometry_info[1:4]
                ego_x = float(ego_x)
                ego_y = float(ego_y)
                ego_z = float(ego_z)
                qx, qy, qz, qw = odometry_info[4:8]
                qx = float(qx)
                qy = float(qy)
                qz = float(qz)
                qw = float(qw)

                sweep_info = dict()
                # NOTE: lidar = ego
                sweep_info['lidar2ego_translation'] = [0,0,0]
                sweep_info['lidar2ego_rotation'] = [1,0,0,0]
                sweep_info['ego2global_translation'] = [ego_x, ego_y, ego_z]
                sweep_info['ego2global_rotation'] = [qw, qx, qy, qz]

                ann_info = mmcv.load(osp.join(dirs, f'merge_json/{file_name}.json'))
                # if len(ann_info['annotations']) == 0:
                #     print('\n no box file:', osp.join(dirs, f'merge_json/{file_name}.json'))
                #     continue
                sweep_info['timestamp'] = ann_info['timestamp']
                lidar_path = ann_info['name']

                img_infos = ann_info['images']
                sweep_info['lidar_path'] = osp.join(dirs, f'bev_pcd/{lidar_path}')

                sweep_info['token'] = str(token_id)
                token_id += 1

                lidar_timestamp = ann_info['timestamp']
                radar_infos = dict()

                flag = False
                for radar in self.radar_names:
                    radar_info = dict()
                    radar_timestamp = sync[str(lidar_timestamp)][radar]
                    data_path = osp.join(dirs, f'radar_object_json/{radar}/{radar_timestamp}.json')
                    radar_info['data_path'] = data_path
                    if not os.path.isfile(radar_info['data_path']):
                        # print('\n',lidar_timestamp, radar_info['data_path'])
                        assert float(radar_timestamp) == 0
                    # if not os.path.isfile(data_path):
                    #     # print(f'\nbad dirs: {dirs}')
                    #     bad_files.append(data_path)
                    #     flag = True
                    #     break
                    radar_info['timestamp'] = float(radar_timestamp)
                    radar_info['sensor2ego_translation'] = [0,0,0]
                    radar_info['sensor2ego_rotation'] = [1,0,0,0]
                    radar_info['ego2global_translation'] = sweep_info['ego2global_translation']
                    radar_info['ego2global_rotation'] = sweep_info['ego2global_rotation']

                    radar_info['sensor2lidar_translation'] = np.array([0,0,0])
                    radar_info['sensor2lidar_rotation'] = np.eye(3)

                    if radar not in prev_radar:
                        radar_infos[radar] = [radar_info]*self.num_radar_sweeps
                    else:
                        sweep_radar = self.pastradar2currego(deepcopy(prev_radar[radar]), radar_info)
                        
                        radar_infos[radar] = [radar_info] + sweep_radar[:-1] # t, t-1, t-2, ...
                    
                    prev_radar[radar] = deepcopy(radar_infos[radar])
                    # radar_infos[radar] = [radar_info]
                # if flag:
                #     continue
                sweep_info['radars'] = radar_infos


                gt_boxes = []
                gt_names = []
                gt_velocity = []
                valid_flag = []
                attributes = []

                if len(ann_info['annotations']) == 0:
                    gt_boxes.append([0,0,0,0,0,0,0])
                    gt_names.append('a')
                    gt_velocity.append([0,0])
                    valid_flag.append(False)
                else:
                    for ann in ann_info['annotations']:
                        gt_names.append(self.label_name_map[ann['label']])
                        center = ann['center']
                        size = ann['size']
                        yaw = ann['yaw']
                        
                        box = [center['x'], center['y'], center['z'],
                                size['x'], size['y'], size['z'], 
                                yaw]
                        gt_boxes.append(box)

                        velocity = [ann['velocity2ground']['x'], ann['velocity2ground']['y']] # TODO: velocity2ground_in_global or velocity2ground or velocity2ego
                        gt_velocity.append(velocity)

                        valid_flag.append(ann['attributes']['ignore'] == "No")
                        attributes.append(ann['attributes'])

                sweep_info['gt_boxes'] = np.asarray(gt_boxes)
                sweep_info['gt_names'] = np.asarray(gt_names)
                sweep_info['gt_velocity'] = np.asarray(gt_velocity)
                sweep_info['valid_flag'] = valid_flag
                sweep_info['attributes'] = attributes

                # calib = mmcv.load(osp.join(dirs, f'info_file/calib.yaml'))
                sweep_info['cams'] = dict()
                img_timestamps = sync[str(lidar_timestamp)]
                for cam in self.camera_names: 
                    # 'sensor2ego_translation', 'sensor2ego_rotation', # 4
                    # 'ego2global_translation', 'ego2global_rotation', # 4
                    # 'sensor2lidar_rotation', 'sensor2lidar_translation', 
                    # 'sensor2global_rotation', 'sensor2global_translation'

                    cam_info = dict()
                    img_timestamp = img_timestamps[cam]
                    cam_info['data_path'] = osp.join(dirs, f'undistort_images/{cam}/{img_timestamp}.jpg')
                    assert os.path.isfile(cam_info['data_path']), print(cam_info['data_path'])
                    cam_info['timestamp'] = float(img_timestamp)
                    # for img_info in img_infos:
                    #     if img_info['sensorAngle'] == cam:
                    #         img_path = img_info['name']
                    #         timestamp = 
                    #         cam_info['data_path'] = osp.join(dirs, f'undistort_images/{img_path}')
                    #         # if not os.path.isfile(data_path):
                    #         #     # print(f'\nbad dirs: {dirs}')
                    #         #     bad_files.append(data_path)
                    #         #     flag = True
                    #         #     break
                    #         cam_info['timestamp'] = img_info['timestamp']

                    cam_info['cam_intrinsic'] = np.array(calib['Intrinsic'][cam]['K'])

                    cam_info['ego2global_translation'] = sweep_info['ego2global_translation']
                    cam_info['ego2global_rotation'] = sweep_info['ego2global_rotation']

                    ego2sensor = np.array(calib['Car2Camera'][cam])

                    cam_info['lidar2cam'] = ego2sensor

                    sensor2ego = np.linalg.inv(ego2sensor)
                    sensor2ego_rotation = Quaternion(matrix=sensor2ego).q.tolist()
                    sensor2ego_translation = sensor2ego[:3, -1].tolist()
                    cam_info['sensor2ego_translation'] = sensor2ego_translation
                    cam_info['sensor2ego_rotation'] = sensor2ego_rotation

                    cam_info['sensor2lidar_rotation'] = sensor2ego[:3, :3]
                    cam_info['sensor2lidar_translation'] = sensor2ego[:3, -1]

                    ego2global_r = Quaternion(cam_info['ego2global_rotation']).rotation_matrix
                    ego2global_t = cam_info['ego2global_translation']

                    ego2global_rt = np.eye(4)
                    ego2global_rt[:3, :3] = ego2global_r
                    ego2global_rt[:3, 3] = ego2global_t

                    # ego2global_rt = inv(ego2global_rt)

                    sensor2global = ego2global_rt @ sensor2ego

                    # sensor2ego_r = sensor2ego[:3, :3]
                    # sensor2ego_t = sensor2ego[:3, -1]

                    # sensor2global_rotation = sensor2ego_r.T @ ego2global_r.T
                    # sensor2global_translation = sensor2ego_t @ ego2global_r.T + ego2global_t

                    # print(1, sensor2global)
                    # print(2, sensor2global_rotation, sensor2global_translation)


                    cam_info['sensor2global_rotation'] = sensor2global[:3, :3].T # NOTE
                    cam_info['sensor2global_translation'] = sensor2global[:3, -1]

                    sweep_info['cams'][cam] = cam_info
                
                # sweep_info['radars'] = 
                if prev_sweep == '':
                    sweep_info['sweeps'] = []
                    prev_sweep = []
                    for i in range(self.sweeps_num):
                        prev_sweep.append(sweep_info['cams'])
                else:
                    sweep_info['sweeps'] = prev_sweep
                    prev_sweep = prev_sweep[1:self.sweeps_num]
                    prev_sweep.append(sweep_info['cams'])

                data_info.append(sweep_info)
        return data_info, bad_dirs

    def project_lidar_to_image(self, points, img_list, frame_info):
        points_img_dict = dict()
        # embed()
        for cam_no, cam_name in enumerate(self.__class__.camera_names):
            calib_info = frame_info['cams'][cam_name]
            # cam_2_velo = calib_info['cam_to_velo']
            lidar2cam = calib_info['lidar2cam']
            # print(lidar2cam)
            # lidar2cam[:3, :3] = lidar2cam[:3, :3].T
            cam_intri = np.hstack([calib_info['cam_intrinsic'], np.zeros((3, 1), dtype=np.float32)])
            point_xyz = points[:, :3]
            points_homo = np.hstack(
                [point_xyz, np.ones(point_xyz.shape[0], dtype=np.float32).reshape((-1, 1))])
            points_lidar = np.dot(points_homo, lidar2cam.T)
            mask = points_lidar[:, 2] > 0
            points_lidar = points_lidar[mask]
            points_img = np.dot(points_lidar, cam_intri.T)
            # print(cam_intri.T)
            points_img = points_img / points_img[:, [2]]
            img_buf = deepcopy(img_list[cam_no])
            for point in points_img:
                try:
                    cv2.circle(img_buf, (int(point[0]), int(point[1])), 2, color=(0, 0, 255), thickness=-1)
                except:
                    print(int(point[0]), int(point[1]))
            points_img_dict[cam_name] = np.concatenate([img_buf, img_list[cam_no]], axis=1) 
        return points_img_dict

    def project_boxes_to_image(self, img_list, frame_info):

        img_dict = dict()
        for cam_no, cam_name in enumerate(self.__class__.camera_names):
            img_buf = img_list[cam_no]

            calib_info = frame_info['cams'][cam_name]
            # cam_2_velo = calib_info['cam_to_velo']
            lidar2cam = calib_info['lidar2cam']
            # print(lidar2cam)
            cam_intri = np.hstack([calib_info['cam_intrinsic'], np.zeros((3, 1), dtype=np.float32)])

            cam_annos_3d = np.array(frame_info['gt_boxes'])

            corners_norm = np.stack(np.unravel_index(np.arange(8), [2, 2, 2]), axis=1).astype(
                np.float32)[[0, 1, 3, 2, 0, 4, 5, 7, 6, 4, 5, 1, 3, 7, 6, 2], :] - 0.5
            corners = np.multiply(cam_annos_3d[:, 3: 6].reshape(-1, 1, 3), corners_norm)
            rot_matrix = np.stack(list([np.transpose(self.rotate_z(box[-1])) for box in cam_annos_3d]), axis=0)
            corners = np.einsum('nij,njk->nik', corners, rot_matrix) + cam_annos_3d[:, :3].reshape((-1, 1, 3))

            for i, corner in enumerate(corners):
                points_homo = np.hstack([corner, np.ones(corner.shape[0], dtype=np.float32).reshape((-1, 1))])
                points_lidar = np.dot(points_homo, lidar2cam.T)
                mask = points_lidar[:, 2] > 0
                points_lidar = points_lidar[mask]
                points_img = np.dot(points_lidar, cam_intri.T)
                points_img = points_img / points_img[:, [2]]
                if points_img.shape[0] != 16:
                    continue
                for j in range(15):
                    cv2.line(img_buf, (int(points_img[j][0]), int(points_img[j][1])), (int(points_img[j+1][0]), int(points_img[j+1][1])), (0, 255, 0), 2, cv2.LINE_AA)

            # cam_annos_2d = frame_info['annos']['boxes_2d'][cam_name]

            # for box2d in cam_annos_2d:
            #     box2d = list(map(int, box2d))
            #     if box2d[0] < 0:
            #         continue
            #     cv2.rectangle(img_buf, tuple(box2d[:2]), tuple(box2d[2:]), (255, 0, 0), 2)

            img_dict[cam_name] = img_buf
        return img_dict

def load_point_cloud(data_path):
        pcd = o3d.io.read_point_cloud(data_path)
        points = np.asarray(pcd.points)
        return points

def load_image(cam_path):
    img_buf = cv2.cvtColor(cv2.imread(cam_path), cv2.COLOR_BGR2RGB)
    return img_buf

def load_points_nus(pts_filename):

    file_client = mmcv.FileClient(backend='disk')
    try:
        pts_bytes = file_client.get(pts_filename)
        points = np.frombuffer(pts_bytes, dtype=np.float32)
    except ConnectionError:
        mmcv.check_file_exist(pts_filename)
        if pts_filename.endswith('.npy'):
            points = np.load(pts_filename)
        else:
            points = np.fromfile(pts_filename, dtype=np.float32)

    return points

def project_lidar_to_image_nus(points, img_list, frame_info):
        points_img_dict = dict()
        # embed()
        keys = frame_info['cams'].keys()
        for cam_no, cam_name in enumerate(keys):
            calib_info = frame_info['cams'][cam_name]
            # cam_2_velo = calib_info['cam_to_velo']
            # lidar2cam = calib_info['lidar2cam']
            lidar2cam_r = np.linalg.inv(calib_info['sensor2lidar_rotation'])
            lidar2cam_t = calib_info['sensor2lidar_translation'] @ lidar2cam_r.T

            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t

            lidar2cam_rt = lidar2cam_rt.T
            print(lidar2cam_rt)

            cam_intri = np.hstack([calib_info['cam_intrinsic'], np.zeros((3, 1), dtype=np.float32)])
            point_xyz = points[:, :3]
            points_homo = np.hstack(
                [point_xyz, np.ones(point_xyz.shape[0], dtype=np.float32).reshape((-1, 1))])
            points_lidar = np.dot(points_homo, lidar2cam_rt.T)
            mask = points_lidar[:, 2] > 0
            points_lidar = points_lidar[mask]
            points_img = np.dot(points_lidar, cam_intri.T)
            # print(cam_intri.T)
            points_img = points_img / points_img[:, [2]]
            img_buf = img_list[cam_no]
            for point in points_img:
                try:
                    cv2.circle(img_buf, (int(point[0]), int(point[1])), 2, color=(0, 0, 255), thickness=-1)
                except:
                    print(int(point[0]), int(point[1]))
            points_img_dict[cam_name] = img_buf
        return points_img_dict


def project_boxes_to_image_nus(img_list, frame_info):
    img_dict = dict()
    keys = frame_info['cams'].keys()
    for cam_no, cam_name in enumerate(keys):
        img_buf = img_list[cam_no]

        calib_info = frame_info['cams'][cam_name]
        lidar2cam_r = np.linalg.inv(calib_info['sensor2lidar_rotation'])
        lidar2cam_t = calib_info['sensor2lidar_translation'] @ lidar2cam_r.T

        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t

        lidar2cam_rt = lidar2cam_rt.T
        # print(lidar2cam)
        cam_intri = np.hstack([calib_info['cam_intrinsic'], np.zeros((3, 1), dtype=np.float32)])

        cam_annos_3d = np.array(frame_info['gt_boxes'])

        corners_norm = np.stack(np.unravel_index(np.arange(8), [2, 2, 2]), axis=1).astype(
            np.float32)[[0, 1, 3, 2, 0, 4, 5, 7, 6, 4, 5, 1, 3, 7, 6, 2], :] - 0.5
        corners = np.multiply(cam_annos_3d[:, 3: 6].reshape(-1, 1, 3), corners_norm)
        rot_matrix = np.stack(list([np.transpose(ChangAn.rotate_z(box[-1])) for box in cam_annos_3d]), axis=0)
        corners = np.einsum('nij,njk->nik', corners, rot_matrix) + cam_annos_3d[:, :3].reshape((-1, 1, 3))

        for i, corner in enumerate(corners):
            points_homo = np.hstack([corner, np.ones(corner.shape[0], dtype=np.float32).reshape((-1, 1))])
            points_lidar = np.dot(points_homo, lidar2cam_rt.T)
            mask = points_lidar[:, 2] > 0
            points_lidar = points_lidar[mask]
            points_img = np.dot(points_lidar, cam_intri.T)
            points_img = points_img / points_img[:, [2]]
            if points_img.shape[0] != 16:
                continue
            for j in range(15):
                cv2.line(img_buf, (int(points_img[j][0]), int(points_img[j][1])), (int(points_img[j+1][0]), int(points_img[j+1][1])), (0, 255, 0), 2, cv2.LINE_AA)

        # cam_annos_2d = frame_info['annos']['boxes_2d'][cam_name]

        # for box2d in cam_annos_2d:
        #     box2d = list(map(int, box2d))
        #     if box2d[0] < 0:
        #         continue
        #     cv2.rectangle(img_buf, tuple(box2d[:2]), tuple(box2d[2:]), (255, 0, 0), 2)

        img_dict[cam_name] = img_buf
    return img_dict

def compose_lidar2img(ego2global_translation_curr,
                      ego2global_rotation_curr,
                      lidar2ego_translation_curr,
                      lidar2ego_rotation_curr,
                      sensor2global_translation_past,
                      sensor2global_rotation_past):
    # print(ego2global_translation_curr,
    #                   ego2global_rotation_curr,
    #                   lidar2ego_translation_curr,
    #                   lidar2ego_rotation_curr,
    #                   sensor2global_translation_past,
    #                   sensor2global_rotation_past)
    ego2global_rotation_curr = Quaternion(ego2global_rotation_curr).rotation_matrix
    lidar2ego_rotation_curr = Quaternion(lidar2ego_rotation_curr).rotation_matrix
    ego2global_translation_curr = np.asarray(ego2global_translation_curr)
    lidar2ego_translation_curr = np.asarray(lidar2ego_translation_curr)

    # print(ego2global_translation_curr,
    #                   ego2global_rotation_curr,
    #                   lidar2ego_translation_curr,
    #                   lidar2ego_rotation_curr,
    #                   sensor2global_translation_past,
    #                   sensor2global_rotation_past)

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

    # viewpad = np.eye(4)
    # viewpad[:cam_intrinsic_past.shape[0], :cam_intrinsic_past.shape[1]] = cam_intrinsic_past
    # lidar2img = (viewpad @ lidar2cam_rt.T).astype(np.float32)

    return lidar2cam_rt.T

def print_pcgt_on_bev(pc, gt, path, radar=None):
    points = pc

    if gt.shape[0] > 0:
        x = gt[:, 0].reshape(-1)
        y = gt[:, 1].reshape(-1)
        z = gt[:, 2].reshape(-1)
        dx = gt[:, 3].reshape(-1)
        dy = gt[:, 4].reshape(-1)
        dz = gt[:, 5].reshape(-1)
        yaw = gt[:, 6].reshape(-1)
        ll = x.shape[0]

    fig = plt.figure(figsize=(32, 32))

    xxx = points[:, 0]
    yyy = points[:, 1]
    zzz = points[:, 2]
    plt.scatter(xxx, yyy, s=0.5)

    if gt.shape[0] > 0:
        for i in range(ll):
            xcos = dx[i] * cos(yaw[i]) / 2
            xsin = dx[i] * sin(yaw[i]) / 2
            ycos = dy[i] * cos(yaw[i]) / 2
            ysin = dy[i] * sin(yaw[i]) / 2
            x1 = x[i] + xcos - ysin
            y1 = y[i] + xsin + ycos
            x2 = x[i] + xcos + ysin
            y2 = y[i] + xsin - ycos
            x3 = x[i] - xcos + ysin
            y3 = y[i] - xsin - ycos
            x4 = x[i] - xcos - ysin
            y4 = y[i] - xsin + ycos
            plt.plot([x1,x2,x3,x4,x1], [y1,y2,y3,y4,y1], lw=1)

    if radar is not None:
        plt.scatter(radar[:, 0], radar[:, 1], s=10, c='r')

    # fig, axes = plt.subplots(nrows=3, figsize=(32, 32))

    # axes[0].hist(points[:, 0], bins=30, color='green', edgecolor='k')
    # axes[1].hist(points[:, 1], bins=30, color='green', edgecolor='k')
    # axes[2].hist(points[:, 2], bins=30, color='green', edgecolor='k')

    plt.savefig(path)
    plt.close()

from nuscenes.utils.data_classes import RadarPointCloud



def load_radar_nus(radars_dict):
    def _load_points_radar(pts_filename):

        radar_obj = RadarPointCloud.from_file(pts_filename)
        points = radar_obj.points

        return points.transpose().astype(np.float32)

    points_sweep_list = []
    # embed()
    for key, sweeps in radars_dict.items():
        print(len(sweeps))
        idxes = list(range(len(sweeps)))
        ts = sweeps[0]['timestamp'] * 1e-6
        for idx in idxes:
            sweep = sweeps[idx]

            points_sweep = _load_points_radar(sweep['data_path'])
            points_sweep = np.copy(points_sweep).reshape(-1, 18)

            points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                'sensor2lidar_rotation'].T
            points_sweep[:, :3] += sweep['sensor2lidar_translation']
            # print()
            points_sweep_ = points_sweep[:, :3]
            points_sweep_list.append(points_sweep_)
    
    points = np.concatenate(points_sweep_list, axis=0)
    # print(points.shape)
        
    return points

def load_radar_nus(radars_dict):
    def _load_points_radar(pts_filename):

        radar_obj = RadarPointCloud.from_file(pts_filename)
        points = radar_obj.points

        return points.transpose().astype(np.float32)

    points_sweep_list = []
    # embed()
    for key, sweeps in radars_dict.items():
        # print(len(sweeps))
        idxes = list(range(len(sweeps)))
        ts = sweeps[0]['timestamp'] * 1e-6
        for idx in idxes:
            sweep = sweeps[idx]

            points_sweep = _load_points_radar(sweep['data_path'])
            points_sweep = np.copy(points_sweep).reshape(-1, 18)

            points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                'sensor2lidar_rotation'].T
            points_sweep[:, :3] += sweep['sensor2lidar_translation']
            # print()
            points_sweep_ = points_sweep[:, :3]
            points_sweep_list.append(points_sweep_)
    
    points = np.concatenate(points_sweep_list, axis=0)
    # print(points.shape)
        
    return points


def load_radar_changan(radars_dict):
    def _load_points_radar(pts_filename):

        radar_obj = mmcv.load(pts_filename)
        radar_type = pts_filename.split('/')[-2]
        radar_obj = radar_obj[radar_type]
        keys = list(radar_obj.keys())
        assert len(keys) == 1
        radar_obj = radar_obj[keys[0]]
        points = []
        for radar in radar_obj:
            p = [radar['center_x'], radar['center_y'], radar['center_z'],
                 radar['class'], radar['confidence']/100, radar['obstacle_prob']/100, 
                 radar['motionstatus'],
                 radar['size_x'], radar['size_y'], radar['size_z'], radar['yaw'],
                 radar['velocity_lateral'], radar['velocity_longitudinal']
                ]
            p = np.array(p)
            points.append(p)
        points = np.stack(points)

        return points.astype(np.float32)

    points_sweep_list = []
    # embed()
    for key, sweeps in radars_dict.items():
        # print(len(sweeps))
        idxes = list(range(len(sweeps)))
        # print(sweeps[0]['timestamp'])
        ts = sweeps[0]['timestamp'] * 1e-6
        for idx in idxes:
            sweep = sweeps[idx]
            if not os.path.isfile(sweep['data_path']):
                continue
            points_sweep = _load_points_radar(sweep['data_path'])
            points_sweep = np.copy(points_sweep)
            # print(points_sweep.shape)

            points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                'sensor2lidar_rotation'].T
            points_sweep[:, :3] += sweep['sensor2lidar_translation']
            # print()
            points_sweep_ = points_sweep[:, :3]
            points_sweep_list.append(points_sweep_)
    
    points = np.concatenate(points_sweep_list, axis=0)
    # print(points.shape)
        
    return points

if __name__ == '__main__':
    dataset = ChangAn('./data/changan')
    dataset.save_data('./data/changan')

#### vis code ####
# if __name__ == '__main__':
#     dataset = ChangAn('./changan_data')

    ### vis for curr point/box to img
    # a = dataset.data_info[0]
    # points = load_point_cloud(a['lidar_path'])
    # img_list = []
    # for i in a['cams'].keys():
    #     img = load_image(a['cams'][i]['data_path'])
    #     img_list.append(img)
    
    # img_buf_dict = dataset.project_lidar_to_image(points, img_list, a)
    # img_buf_dict = dataset.project_boxes_to_image(img_list, a)

    # ### vis for cur point/box to past img
    # curr = dataset.data_info[10]
    # points = load_point_cloud(curr['lidar_path'])
    # past = dataset.data_info[0]
    # img_list = []
    # for i in past['cams'].keys():
    #     img = load_image(past['cams'][i]['data_path'])
    #     img_list.append(img)
    
    # # cams_info = past['cams']
    # for cam in dataset.camera_names:
    #     # curr_info = curr['cams'][cam]
    #     past_info = past['cams'][cam]
    #     currlidar2pastimg = compose_lidar2img(curr['ego2global_translation'],
    #                     curr['ego2global_rotation'],
    #                     curr['lidar2ego_translation'],
    #                     curr['lidar2ego_rotation'],
    #                     past_info['sensor2global_translation'],
    #                     past_info['sensor2global_rotation'])
    #     past['cams'][cam]['lidar2cam'] = currlidar2pastimg
    
    # img_buf_dict = dataset.project_lidar_to_image(points, img_list, past)
    # for cam_name, img_buf in img_buf_dict.items():
    #     # print(img_buf.shape)
    #     cv2.imwrite('./vis1/lidar_project_{}.jpg'.format(cam_name), cv2.cvtColor(img_buf, cv2.COLOR_BGR2RGB))

    # past['gt_boxes'] = curr['gt_boxes']
    # img_buf_dict = dataset.project_boxes_to_image(img_list, past)
    # for cam_name, img_buf in img_buf_dict.items():
    #     # print(img_buf.shape)
    #     cv2.imwrite('./vis1/box_project_{}.jpg'.format(cam_name), cv2.cvtColor(img_buf, cv2.COLOR_BGR2RGB))


    #### vis nus
    # a=mmcv.load('/home/linzhiwei/project/bevperception/data/nuscenes/nuscenes_R_10frame_infos_val_occ.pkl')['infos'][0]
    # # points = load_points_nus(a['lidar_path'])
    # # points = points.reshape(-1, 5)
    # img_list = []
    # for i in a['cams'].keys():
    #     img = load_image(a['cams'][i]['data_path'])
    #     img_list.append(img)

    # img_buf_dict = project_boxes_to_image_nus(img_list, a)

    # i = 0
    # for cam_name, img_buf in img_buf_dict.items():
    #     print(img_buf.shape)
    #     cv2.imwrite('./vis1/nus/box_project_{}.jpg'.format(cam_name), cv2.cvtColor(img_buf, cv2.COLOR_BGR2RGB))
    

    # ##### vis nus box on radar/lidar
    # a=mmcv.load('/home/linzhiwei/project/bevperception/data/nuscenes/nuscenes_R_10frame_infos_val_occ.pkl')['infos'][10]
    # points = load_points_nus(a['lidar_path'])
    # points = points.reshape(-1, 5)
    # radar_points = load_radar_nus(a['radars'])

    # print_pcgt_on_bev(points, a['gt_boxes'], "./vis1/nus/lidar_radar_box.jpg", radar_points)

    #### vis nus box on radar/lidar
    # a = dataset.data_info[0]
    # points = load_point_cloud(a['lidar_path'])
    # radar_points = load_radar_changan(a['radars'])

    # print_pcgt_on_bev(points, a['gt_boxes'], "./vis1/lidar_radar_box.jpg", radar_points)
    # for i, a in enumerate(dataset.data_info):
    #     print(i)
    #     points = load_point_cloud(a['lidar_path'])
    #     radar_points = load_radar_changan(a['radars'])

    #     print_pcgt_on_bev(points, a['gt_boxes'], f"./vis1/lidar_radar_box_{i}.jpg", radar_points)

    