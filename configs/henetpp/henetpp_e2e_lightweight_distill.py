_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']

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
    # 'input_size': (256, 704),h
    'input_size': (384, 704),
    'src_size': (900, 1600),

    # 'resize': (-0.06, 0.11),
    # 'rot': (-5.4, 5.4),
    'resize': (0, 0),
    'rot': (0, 0),
    # 'flip': True,
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,

    # debug
    # 'resize': (0, 0),
    # 'rot': (0, 0),
    # 'flip': False,
    # 'crop_h': (0.0, 0.0),
    # 'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

point_cloud_range_det = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size_det = [0.2, 0.2, 8]
point_cloud_range = [-40, -40, -5, 40, 40, 3]
radar_voxel_size = [0.2, 0.2, 8]
radar_use_dims = [0, 1, 2, 8, 9, 5, 18]
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 32
numRadar_Trans = 64
multi_adj_frame_id_cfg = (1, 1 + 8, 1)
embed_dims = 256
num_layers = 6
num_query = 900
num_frames = 8
num_levels = 4
num_points = 4
with_cp = False

model = dict(
    type='HenetppRC_planner',
    ret_2d_feat=False,
    freeze_img=False,
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    num_fut_steps=6,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=False,
        style='pytorch'
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]
    ),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVStereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=False,
        loss_depth_weight=0.05,
        depthnet_cfg=dict(use_dcn=False,
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.),
        downsample=16
    ),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg)) + 1),
        num_layer=[1, 2, 4],
        with_cp=False,
        num_channels=[numC_Trans, numC_Trans * 2, numC_Trans * 4],
        stride=[1, 2, 2],
        backbone_output_ids=[0, 1, 2]
    ),
    img_bev_encoder_neck=dict(type='LSSFPN3D',
                              in_channels=numC_Trans * 7,
                              out_channels=numC_Trans),
    pre_process=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans,
        with_cp=False,
        num_layer=[1, ],
        num_channels=[numC_Trans, ],
        stride=[1, ],
        backbone_output_ids=[0, ]
    ),
    radar_voxel_layer=dict(
        max_num_points=10,
        voxel_size=radar_voxel_size,
        max_voxels=(90000, 120000),
        point_cloud_range=point_cloud_range
    ),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=5,
    ),
    radar_voxel_encoder=dict(
        type='RadarFeatureNetAdapterNoMaskV2',
        # return_rcs=True,
        in_channels=6 + 1,
        feat_channels=[32, 64],
        with_distance=False,
        point_cloud_range=point_cloud_range,
        voxel_size=radar_voxel_size,
        norm_cfg=dict(
            type='BN1d',
            eps=1.0e-3,
            momentum=0.01
        ),
        with_pos_embed=True,
        permute_injection_extraction=True
    ),
    radar_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[400, 400]
    ),

    radar_bev_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    radar_bev_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True
    ),

    imc=numC_Trans,
    rac=sum([128, 128, 128]),
    radar_reduc_conv=True,

    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0
    ),
    use_mask=True,

    data_aug=dict(
        img_color_aug=True,
        img_norm_cfg=dict(
            mean=[123.675, 116.280, 103.530],
            std=[58.395, 57.120, 57.375],
            to_rgb=True
        ),
        img_pad_cfg=dict(size_divisor=32)
    ),
    stop_prev_grad=0,
    neck_det=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=embed_dims,
        num_outs=num_levels
    ),
    pts_bbox_head=dict(
        type='SparseBEVHead',
        num_classes=10,
        in_channels=embed_dims,
        num_query=num_query,
        query_denoising=True,
        query_denoising_groups=10,
        code_size=10,
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        sync_cls_avg_factor=True,
        return_bbox_feat=True,
        transformer=dict(
            type='SparseBEVTransformer',
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=10,
            code_size=10,
            pc_range=point_cloud_range_det
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range_det,
            max_num=300,
            voxel_size=voxel_size_det,
            score_threshold=0.05,
            num_classes=10
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=embed_dims // 2,
            normalize=True,
            offset=-0.5
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)
    ),

    motion_encoder=dict(
        type='CustomTransformerDecoder', 
        num_layers=1,  
        return_intermediate=False,  
        transformerlayers=dict(
            type='BaseTransformerLayer', 
            attn_cfgs=[
                dict(
                    type='MultiheadAttention', 
                    embed_dims=256, 
                    num_heads=8,  
                    dropout=0.1  
                ),
            ],
            feedforward_channels=512,  
            ffn_dropout=0.1, 
            operation_order=('cross_attn', 'norm', 'ffn', 'norm') 
        )
    ),
    motion_decoder=dict(
        type = 'CustomTransformerDecoder',
        # other args
        return_intermediate = False,
        num_layers=1,
        transformerlayers = dict(
            type='BaseTransformerLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',  
                    embed_dims=256,  
                    num_heads=8, 
                    dropout=0.1 
                ),
            ],
            feedforward_channels=512, 
            ffn_dropout=0.1, 
            operation_order=('cross_attn', 'norm', 'ffn', 'norm') 
        )
    ),

    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1], 
        voxel_size=voxel_size_det, 
        point_cloud_range=point_cloud_range_det,  
        out_size_factor=4,  
        assigner=dict(
            type='HungarianAssigner3D_2', 
            cls_cost=dict(type='FocalLossCost', weight=2.0), 
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),  
            iou_cost=dict(type='IoUCost', weight=0.0)  
        )
    ))
)

dataset_type = 'NuScenesDatasetOccpancyPlanner' 
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')
occ_gt_data_root = 'data/nuscenes'


ida_aug_conf_mf = {
    'resize_lim': (0.38, 0.55),  
    'final_dim': (384, 704), 
    'bot_pct_lim': (0.0, 0.0),  
    'rot_lim': (0.0, 0.0),  
    'H': 900, 'W': 1600,  
    'rand_flip': True,  
}

bda_aug_conf = dict(
    rot_lim=(-0., 0.), 
    scale_lim=(1., 1.),  
    flip_dx_ratio=0, 
    flip_dy_ratio=0 
)

train_pipeline = [
    dict(
        type='PrepareImageInputs',  
        is_train=True, 
        data_config=data_config,  
        sequential=True 
    ),
    dict(
        type='LoadRadarPointsMultiSweeps',  
        load_dim=18, 
        sweeps_num=8, 
        use_dim=radar_use_dims, 
        max_num=1200, 
    ),
    dict(type='LoadOccGTFromFile', data_root=occ_gt_data_root), 
    dict(
        type='LoadAnnotationsBEVDepth', 
        bda_aug_conf=bda_aug_conf, 
        classes=class_names,
        is_train=True 
    ),
    dict(type='GlobalRotScaleTrans_radar'),  
    dict(
        type='LoadPointsFromFile',  
        coord_type='LIDAR', 
        load_dim=5, 
        use_dim=5, 
        file_client_args=file_client_args  
    ),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config), 
    
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),  
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1), 
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False), 
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range_det), 
    dict(type='ObjectNameFilter', classes=class_names),  
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf_mf, training=True), 
    dict(type='GlobalRotScaleTransImage', rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05]), 
    dict(type='DefaultFormatBundle3D', class_names=class_names), 
    dict(
        type='Collect3D', 
        # keys=[]
        keys=['img_inputs', 'gt_depth', 'voxel_semantics', 'mask_lidar', 'mask_camera', 'radar',
              'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 'gt_fut_trajs_abs'], 
        meta_keys=(
            'filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp') 
    )
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True), 
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18, 
        sweeps_num=8, 
        use_dim=radar_use_dims, 
        max_num=1200,  
    ),
    dict(
        type='LoadAnnotationsBEVDepth',  
        bda_aug_conf=bda_aug_conf,  
        classes=class_names, 
        is_train=False 
    ),
    dict(type='GlobalRotScaleTrans_radar'), 
    dict(
        type='LoadPointsFromFile', 
        coord_type='LIDAR',  
        load_dim=5,  
        use_dim=5,  
        file_client_args=file_client_args  
    ),

    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range_det), 
    dict(type='ObjectNameFilter', classes=class_names), 
    
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'), 
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1, test_mode=True), 
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf_mf, training=False), 
    dict(
        type='MultiScaleFlipAug3D', 
        img_scale=(1333, 800),  
        pts_scale_ratio=1, 
        flip=False, 
        transforms=[
            dict(
                type='DefaultFormatBundle3D', 
                class_names=class_names,
                with_label=False
            ),
            dict(type='Collect3D', 
                 keys=['points', 'img_inputs', 'gt_labels_3d', 'voxel_semantics', 'radar', 
                 'gt_bboxes_3d', 'img', 'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 
                 'fut_valid_flag', 'gt_ego_fut_cmd', 'gt_ego_lcf_feat', 'gt_fut_trajs', 'gt_attr_labels', 'gt_boxes'], 
                 meta_keys=(
                     'filename', 'box_type_3d', 'ori_shape', 'img_shape', 'pad_shape',
                     'lidar2img', 'img_timestamp'
                    ) 
                 )
        ]
    )
]

input_modality = dict(
    use_lidar=False, 
    use_camera=True, 
    use_radar=True,
    use_map=False, 
    use_external=False  
)

share_data_config = dict(
    type=dataset_type,  
    classes=class_names,  
    modality=input_modality,  
    stereo=True,  
    filter_empty_gt=True,
    img_info_prototype='mmcv+bevdet4d', 
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg, 
    use_rays=False  
)

test_data_config = dict(
    load_traj=True, 
    load_others=True,
    pipeline=test_pipeline, 
    use_valid_flag = True,
    ann_file=data_root + 'nuscenes_R_10frame_infos_val_occ_updated.pkl', 
    det_info_file=data_root + 'nuscenes_infos_val_sweep_updated_2.pkl', 
)

train_data_config = dict(
    data_root=data_root, 
    ann_file=data_root + 'nuscenes_R_10frame_infos_train_occ_updated.pkl', 
    load_traj=True, 
    pipeline=train_pipeline, 
    classes=class_names, 
    test_mode=False, 
    use_valid_flag=True, 
    load_adj_occ_labels=False,  
    box_type_3d='LiDAR',
    det_info_file=data_root + 'nuscenes_infos_train_sweep_updated_3.pkl',
)

data = dict(
    samples_per_gpu=1,
    # workers_per_gpu=0,
    workers_per_gpu=8,
    train=train_data_config,
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

optimizer = dict(
    type='AdamW',
    lr=1.0e-5, 
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1), 
    }),
    weight_decay=0.01 
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2)) 

lr_config = dict(
    policy='CosineAnnealing', 
    warmup='linear', 
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

# total_epochs = 24
total_epochs = 60

# custom_hooks = [
#     dict(
#         type='MEGVIIEMAHook',
#         init_updates=10560,
#         priority='NORMAL',
#     ),
# ]


# evaluation = dict(interval=3, pipeline=test_pipeline, metric='bbox', map_metric='chamfer')
evaluation = dict(interval=100)


# log_config = dict(
#     interval=50,
# )

# load_from = 'work_dirs/rc_occ_unfreeze_12ema.pth'
# load_from = '/data/bevperception/work_dirs/rc_with_planner_e2e_no_bug/epoch_1.pth'
# load_det_from = 'work_dirs/sparsebev_r50_fp32.pth'

resume_from = '/data/bevperception/work_dirs/rc_with_planner_e2e_no_bug/epoch_1.pth'

