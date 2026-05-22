'''
mAP: 0.3085
mATE: 0.7149
mASE: 0.2800
mAOE: 0.5977
mAVE: 0.7787
mAAE: 0.2300
NDS: 0.3941
Eval time: 142.1s

Per-class results:
Object Class	AP	ATE	ASE	AOE	AVE	AAE
car	0.523	0.510	0.157	0.104	0.837	0.212
truck	0.229	0.712	0.216	0.133	0.719	0.219
bus	0.276	0.880	0.217	0.120	1.722	0.413
trailer	0.155	0.993	0.231	0.367	0.427	0.179
construction_vehicle	0.069	0.932	0.516	1.303	0.117	0.371
pedestrian	0.332	0.769	0.308	1.311	0.891	0.306
motorcycle	0.276	0.719	0.263	0.750	1.216	0.131
bicycle	0.208	0.682	0.268	1.179	0.301	0.009
traffic_cone	0.518	0.477	0.328	nan	nan	nan
barrier	0.500	0.474	0.297	0.113	nan	nan
'''

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
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
    # 'input_size': (256, 704),
    'input_size': (448, 800),
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
    'depth': [1.0, 60.0, 1.0],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 64
out_size_factor = 8

model = dict(
    type='BEVDet',
    # with_bevencoder=True,
    img_backbone=dict(
        type='CBSwinTransformer',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0,1, 2, 3),
        use_checkpoint=False),
    img_neck=dict(
        # type='CustomFPN',
        # # in_channels=[1024, 2048],
        # in_channels=[384, 768],
        # out_channels=256,
        # num_outs=1,
        # start_level=0,
        # out_ids=[0]
        type='FPNC',
        # final_dim=data_config['input_size'],
        final_dim=(900, 1600),
        downsample=8, 
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        outC=256,
        use_adp=True,
        num_outs=5
        ),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        # input_size=data_config['input_size'],
        input_size=(900, 1600),
        in_channels=256,
        # out_channels=numC_Trans,
        out_channels=256,
        downsample=8),
    # img_bev_encoder_backbone=dict(
    #     type='CustomResNet',
    #     numC_input=numC_Trans,
    #     num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    # img_bev_encoder_neck=dict(
    #     type='FPN_LSS',
    #     in_channels=numC_Trans * 8 + numC_Trans * 2,
    #     out_channels=256),
    img_bev_encoder_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    img_bev_encoder_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[128, 128],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=256, # modify
        hidden_channel=128,
        num_classes=len(class_names),
        num_decoder_layers=3,
        num_heads=8,
        learnable_query_pos=False,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),

        # todo loss weight * 6 or not
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25*3),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        ),
    # model training and testing settings
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
            # grid_size=[1440, 1440, 40],  # [x_len, y_len, 1]
            grid_size=[1024, 1024, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            # todo code weights [0.2,0.2] -> [1.0, 1.0]
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1024, 1024, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
        ))
)

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

# bda_aug_conf = dict(
#     rot_lim=(-45, 45),
#     scale_lim=(0.9, 1.1),
#     flip_dx_ratio=0.5,
#     flip_dy_ratio=0.5)
bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.,
    flip_dy_ratio=0.)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
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
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet',
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'nuscenes_R_infos_val.pkl')

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        # type='CBGSDataset',
        # dataset=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_R_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'
        # )
        ),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)
# data['train']['dataset'].update(share_data_config)
data['train'].update(share_data_config)

# Optimizer
# optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
# optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=200,
#     warmup_ratio=0.001,
#     step=[20,])
optimizer = dict(type='AdamW', lr=0.00001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
optimizer_config = dict(
    cumulative_iters=4,
    grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)

runner = dict(type='EpochBasedRunner', max_epochs=20)

custom_hooks = [
    # dict(
    #     type='MEGVIIEMAHook',
    #     init_updates=10560,
    #     priority='NORMAL',
    # ),
    dict(
        type='SyncbnControlHook',
        syncbn_start_epoch=0,
    ),
]

# fp16 = dict(loss_scale='dynamic')
load_from = 'work_dirs/mask_rcnn_dbswin-t_fpn_3x_nuim_cocopre.pth'
not_print_model=True
# find_unused_parameters=True

evaluation = dict(interval=5)
