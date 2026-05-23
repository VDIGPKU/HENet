_base_ = ['../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-54.0, -54.0, -4.0, 54.0, 54.0, 4.0]
voxel_size = [0.075, 0.075, 0.2]
# radar_voxel_size = [0.8, 0.8, 8]
# radar_voxel_size = [0.2, 0.2, 8]
# x y z vx_comp vy_comp rcs 
# radar_use_dims = [0, 1, 2, 8, 9, 5, 18]
#radar_use_dims = [0, 1, 2, 5]

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}

# grid_config = {
#     'x': [-54.0, 54.0, 0.075*8],
#     'y': [-54.0, 54.0, 0.075*8],
#     'z': [-5, 3, 8],
#     'depth': [1.0, 60.0, 0.5],
# }

# voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 80

multi_adj_frame_id_cfg = (1, 1+1, 1)

out_size_factor = 8

model = dict(
    type='BEVMAEPP_LRC_BEVDepth4D',
    # freeze_img=True,
    # freeze_lidar=True,
    # se=True,
    imc=256,
    lic=512,
    module_fusion=['L', 'C'],
    # lidar_ckpt='work_dirs/checkpoint/transfusion-20ep-3layer.pth',
    # cam_ckpt='work_dirs/checkpoint/det-256x704-r50-BEV128-9kf-depth.pth',
    interpolate_feat=True,

    ## start lidar config
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=5,
    ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='SyncBN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='SyncBN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    # end lidar config

    ## start camera config
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        # pretrained='work_dirs/checkpoint/mask_rcnn_r50_fpn_coco-2x_1x_nuim_20201008_195238-b1742a60.pth',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=False, #改
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=512,
        out_channels=numC_Trans,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        # stride=[2, 2, 1], # modify!!!!!!
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        # scale_factor=2, # modify!!!!!!
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pre_process=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_layer=[2,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    # end cam config

    # #radar start
    # radar_voxel_layer=dict(
    #     max_num_points=10, 
    #     voxel_size=radar_voxel_size, 
    #     max_voxels=(90000, 120000),
    #     point_cloud_range=point_cloud_range),

    # radar_voxel_encoder=dict(
    #     type='RadarFeatureNet',
    #     in_channels=6+1,
    #     feat_channels=[32, 64],
    #     with_distance=False,
    #     point_cloud_range=point_cloud_range,
    #     voxel_size=radar_voxel_size,
    #     norm_cfg=dict(
    #         type='BN1d',
    #         eps=1.0e-3,
    #         momentum=0.01)
    # ),
    # radar_middle_encoder=dict(
    #     type='PointPillarsScatter',
    #     in_channels=64,
    #     # output_shape=[128, 128],
    #     output_shape=[512, 512],
    # ),

    # radar_bev_backbone=dict(
    #     type='SECOND',
    #     in_channels=64,
    #     out_channels=[64, 128, 256],
    #     layer_nums=[3, 5, 5],
    #     layer_strides=[2, 2, 2],
    #     norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
    #     conv_cfg=dict(type='Conv2d', bias=False)),
    # radar_bev_neck=dict(
    #     type='SECONDFPN',
    #     in_channels=[64, 128, 256],
    #     out_channels=[128, 128, 128],
    #     upsample_strides=[0.5, 1, 2],
    #     norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
    #     upsample_cfg=dict(type='deconv', bias=False),
    #     use_conv_for_no_stride=True),
    # rac=sum([128, 128, 128]),
    # radar_reduc_conv=dict(),
    # #radar end

    pts_bbox_head=dict(
        type='SGKDProjMaskNormCosHead',
        img_channels=256,
        pts_channels=512,
        out_channels=1024,
        # loss_distill=dict(type='MSELoss', reduction='none', loss_weight=2e-5),
    ),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25),
                flip_wh=True,
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
        ))
)

# Data
dataset_type = 'NuScenesDataset_R'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-45, 45),
    scale_lim=(0.9, 1.1),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    trans_xyz=[0,0,0]
    )


train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args
    ),
    # load cam
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    # dict( #load radar
    #     type='LoadRadarPointsMultiSweeps',
    #     load_dim=18,
    #     sweeps_num=4,
    #     use_dim=radar_use_dims,
    #     max_num=1200, ),
    # dict(
    #     type='LoadAnnotationsBEVDepth',
    #     bda_aug_conf=bda_aug_conf,
    #     classes=class_names),

    dict(
        type='LoadAnnotationsBEVDepthLidarPre',
        img_info_prototype='mmcv',
        ),
    dict(
        type='LoadAnnotationsBEVDepthLidarPost',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),

    # dict(type='GlobalRotScaleTrans_radar'),

    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        keys_name='ori_points',
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),

    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['points','img_inputs', 'gt_bboxes_3d', 'gt_labels_3d',
                                'gt_depth',])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args
    ),
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    # dict(
    #     type='LoadRadarPointsMultiSweeps',
    #     load_dim=18,
    #     sweeps_num=4,
    #     use_dim=radar_use_dims,
    #     max_num=1200, ),
    # dict(
    #     type='LoadAnnotationsBEVDepth',
    #     bda_aug_conf=bda_aug_conf,
    #     classes=class_names,
    #     is_train=False),
    dict(
        type='LoadAnnotationsBEVDepthLidarPre',
        ),
    dict(
        type='LoadAnnotationsBEVDepthLidarPost',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),

    # dict(type='GlobalRotScaleTrans_radar'),

    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # dict(
            #     type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=data_root + '/nuscenes_R_infos_val.pkl',
    load_interval=1,
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    test_mode=True,
    img_info_prototype='bevdet',
    box_type_3d='LiDAR'
        )

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + '/nuscenes_R_infos_train.pkl',
            load_interval=1,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR',
            img_info_prototype='bevdet',
            )),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)
data['train']['dataset'].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-5, weight_decay=0.05, betas=(0.9, 0.999),) #小10倍
optimizer_config = dict(
    cumulative_iters=2,
    grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(5, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)

runner = dict(type='EpochBasedRunner', max_epochs=20)
evaluation = dict(interval=20)

custom_hooks = [
    # dict(
    #     type='MEGVIIEMAHook',
    #     init_updates=10560,
    #     priority='NORMAL',
    # ),
    dict(
        type='SequentialControlHook', 
        temporal_start_epoch=-1,
    ),
    dict(
        type='SyncbnControlHook',
        syncbn_start_epoch=0,
    ),
]
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# fp16 = dict(loss_scale='dynamic')
# load_from='work_dirs/checkpoint/mask_rcnn_r50_fpn_coco-2x_1x_nuim_20201008_195238-b1742a60.pth'

checkpoint_config = dict(interval=1)
validate=True

not_print_model=True
find_unused_parameters=True
