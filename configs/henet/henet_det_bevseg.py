'''

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
map_classes = ['vehicle', 'drivable_area', 'divider']

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (640, 1152),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.097, 0.178),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

data_config_longterm = {
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
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 1.0],
}

grid_config_longterm = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 80

multi_adj_frame_id_cfg = (1, 1 + 1, 1)

multi_adj_frame_id_cfg_longterm = (1, 8 + 1, 1)

# vovnet backbone
# 对应的out channels = [256, 512, 768, 1024]
# need a neck
# 同时要修改Vovnet代码，把forward的输出从dict改成list
build_vovnet_backbone = {
    'type': 'VovNet',
    'norm': 'BN',  # TODO: 明明config是FrozenBN，为什么pth对应的是BN呢
    'name': 'V-99-eSE',
    'input_ch': 3,
    'out_features': ['stage2', 'stage3', 'stage4', 'stage5'],
    'checkpoint': None,
    'with_cp': False,
}

# vovnet with fpn
build_vovnet_fpn_backbone = {
    'type': 'VovNetFPN',
    'bottom_up_config': build_vovnet_backbone,
    'in_features': ['stage2', 'stage3', 'stage4', 'stage5'],
    'out_channels': 256,
    'norm': 'FrozenBN',
    'top_block': {'type': 'LastLevelMaxPool'},
    'fuse_type': 'sum',
    '_size_divisibility_mul_2': False,
    'checkpoint': None,
}

# vovnet with fpn with p6
build_fcos_vovnet_fpn_backbone_p6 = {
    'type': 'VovNetFPN',
    'bottom_up_config': build_vovnet_backbone,
    'in_features': ['stage2', 'stage3', 'stage4', 'stage5'],
    'out_channels': 256,
    'out_layers': ['p2', 'p4'],
    'norm': 'BN',  # TODO: 明明config是FrozenBN，为什么pth对应的是BN呢
    'top_block': {'type': 'LastLevelP6',
                  'in_channels_top': 256,
                  'out_channels': 256,
                  'in_features': 'p5',
                  },
    'fuse_type': 'sum',
    '_size_divisibility_mul_2': True,
    # 'checkpoint': '/home/wangxinhao/dd3d/depth_pretrained_v99-3jlw0p36-20210423_010520-model_final-remapped.pth',
    'checkpoint': 'work_dirs/depth_pretrained_v99.pth',
    'with_cp': False,
}

# choose type
img_backbone_type = {
    'vovnet_backbone': build_vovnet_backbone,
    'vovnet_fpn_backbone': build_vovnet_fpn_backbone,
    'fcos_vovnet_fpn_backbone_p6': build_fcos_vovnet_fpn_backbone_p6,
}

model = dict(
    type='BEVStereo4D_mix_encoder',
    diff_bev=dict(
        st_scope=[grid_config['x'], grid_config['y']],
        lt_scope=[grid_config_longterm['x'], grid_config_longterm['y']],),
    longterm_model=dict(
        type='BEVDepth4D',
        align_after_view_transfromation=False,
        num_adj=len(range(*multi_adj_frame_id_cfg_longterm)),
        img_backbone=dict(
            pretrained='torchvision://resnet50',
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(2, 3),
            frozen_stages=-1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            # with_cp=True,
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
            grid_config=grid_config_longterm,
            input_size=data_config_longterm['input_size'],
            in_channels=512,
            out_channels=numC_Trans,
            depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
            downsample=16),
        train_cfg=dict(
            pts=dict(
                point_cloud_range=point_cloud_range,
                grid_size=[1024, 1024, 40],
                voxel_size=voxel_size,
                out_size_factor=8,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
        test_cfg=dict(
            pts=dict(
                pc_range=point_cloud_range[:2],
                post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_per_img=500,
                max_pool_nms=False,
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                pre_max_size=1000,
                post_max_size=500,

                # Scale-NMS
                nms_type=['rotate'],
                nms_thr=[0.2],
                nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                                     1.1, 1.0, 1.0, 1.5, 3.5]]
            )
        )
    ),
    imc=numC_Trans * (len(range(*multi_adj_frame_id_cfg)) + 1),
    longterm_imc=numC_Trans * (len(range(*multi_adj_frame_id_cfg_longterm)) + 1),
    reduc_conv=dict(se=True),

    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    img_backbone=img_backbone_type['fcos_vovnet_fpn_backbone_p6'],
    img_view_transformer=dict(
        type='LSSViewTransformerBEVStereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        sid=True,
        depthnet_cfg=dict(use_dcn=False,
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg)) +
                                 len(range(*multi_adj_frame_id_cfg_longterm)) + 2),
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    img_bev_encoder_backbone_forseg=dict(
        type='CustomResNet',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg)) +
                                 len(range(*multi_adj_frame_id_cfg_longterm)) + 2),
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck_forseg=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pre_process=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_layer=[2, ],
        num_channels=[numC_Trans, ],
        stride=[1, ],
        backbone_output_ids=[0, ]),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=10, class_names=['car', 'truck',
                                            'construction_vehicle',
                                            'bus', 'trailer',
                                            'barrier',
                                            'motorcycle', 'bicycle',
                                            'pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=3.),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=1.),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=500,

            # Scale-NMS
            nms_type=['rotate'],
            nms_thr=[0.2],
            nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                                 1.1, 1.0, 1.0, 1.5, 3.5]]
        )
    ),
    pts_seg_head=dict(
        type='BEVSegHead',
        conv_config=[[256, 512, 3], [512, 256, 3], [256, 256, 3],
                     [256, 256, 1], [256, 256, 1], [256, 128, 1], [128, 64, 1]],
        grid_transform=dict(
            input_scope=[[-51.2, 51.2, 0.8], [-51.2, 51.2, 0.8]],
            output_scope=[[-50, 50, 0.5], [-50, 50, 0.5]],),
        classes=map_classes,
        loss='focal',
        loss_weight={
            'vehicle': 300, 'drivable_area': 70, 'divider': 200,
        },
    )
)

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True,),
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config_longterm,
        sequential=True,
        suffix='_lt'),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config_longterm, suffix='_lt'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='LoadBEVSegmentation',
         dataset_root=data_root,
         xbound=[-50.0, 50.0, 0.5],
         ybound=[-50.0, 50.0, 0.5],
         classes=map_classes,
         bbox_classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'img_inputs_lt', 'gt_bboxes_3d', 'gt_labels_3d',
                                'gt_depth', 'gt_depth_lt', 'gt_masks_bev'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(type='PrepareImageInputs', data_config=data_config_longterm, sequential=True, suffix='_lt'),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadBEVSegmentation',
         dataset_root=data_root,
         xbound=[-50.0, 50.0, 0.5],
         ybound=[-50.0, 50.0, 0.5],
         classes=map_classes,
         bbox_classes=class_names),
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
            dict(type='Collect3D', keys=['points', 'img_inputs', 'img_inputs_lt', 'gt_masks_bev'])
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
    map_classes=['vehicle', 'drivable_area', 'divider'],
    modality=input_modality,
    stereo=True,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
    multi_adj_frame_id_cfg_longterm=multi_adj_frame_id_cfg_longterm,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'nuscenes_C_infos_val.pkl')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        # type='CBGSDataset',
        # dataset=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_C_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['train', 'val', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=4e-5, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
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

runner = dict(type='EpochBasedRunner', max_epochs=60)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SequentialControlHook',
        temporal_start_epoch=0,
    ),
    dict(
        type='SyncbnControlHook',
        syncbn_start_epoch=0,
    ),
]

find_unused_parameters = True

# load_from = 'work_dirs/det_640x1152_v299_bev256_2kfstereo_multihead.pth'
# load_mix_from = 'work_dirs/det_256x704_r50_bev128_9kf.pth'

evaluation = dict(interval=999)
checkpoint_config = dict(interval=5)
# work_dir = 'work_dirs/detsegVAD_bevmix640_final'

# fp16 = dict(loss_scale='dynamic')
