_base_ = [
    '../centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py',
]

# pay attention to this when base point cloud range is changed!
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

bda_aug_conf = dict(
    rot_lim=(-45, 45),
    scale_lim=(0.9, 1.1),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    trans_xyz=[0.5, 0.5, 0.5] # set [0, 0, 0] for cam and fusion model!!!
    )

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

model = dict(
    type='DynamicCenterPoint',
    pts_voxel_layer=dict(
        max_num_points=-1, max_voxels=(-1, -1),
    ),
    pts_voxel_encoder=dict(
        _delete_=True,
        type='DynamicPillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=(0.2, 0.2, 8),
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)),
)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        # sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True
        ),
  dict(
        type='LoadAnnotationsBEVDepthLidarPre',
        img_info_prototype='mmcv',
        ),
    dict(
        type='LoadAnnotationsBEVDepthLidarPost',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True
        ),
    dict(
        type='LoadAnnotationsBEVDepthLidarPre',
        ),
    dict(
        type='LoadAnnotationsBEVDepthLidarPost',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet'
)

test_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=data_root + 'nuscenes_R_infos_val.pkl',
    load_interval=1,
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    test_mode=True,
    box_type_3d='LiDAR',
    img_info_prototype='bevdet',
    include_location=False
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_R_infos_train.pkl',
            load_interval=1,
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            img_info_prototype='bevdet',
            include_location=False
            )),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)
data['train']['dataset'].update(share_data_config)
# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
evaluation = dict(interval=5)
resume_from='work_dirs/dynamic_centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus/epoch_5.pth'
