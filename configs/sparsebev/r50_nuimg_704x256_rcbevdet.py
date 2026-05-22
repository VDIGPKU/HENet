
'''
12 ep

mAP: 0.5194                                                                                                                                                                                                            
                                                                                                                                                                                                                       
mATE: 0.4881                                                                                                                                                                                                           
mASE: 0.2679                                                                                                                                                                                                           
mAOE: 0.4079                                                                                               
mAVE: 0.2210                                                                                               
mAAE: 0.1772                                                                                               
NDS: 0.6035

'''

_base_ = ['../_base_/datasets/nus-3d.py','../_base_/default_runtime.py']


# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# radar_voxel_size = [0.8, 0.8, 8]
radar_voxel_size = [0.2, 0.2, 8]
# x y z vx_comp vy_comp rcs 
radar_use_dims = [0, 1, 2, 8, 9, 5, 18]
#radar_use_dims = [0, 1, 2, 5]



# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
#point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8] #different from rcbevdet
out_size_factor = 8 #new
voxel_size_r = [0.1, 0.1, 0.2] 
pillar_size = [voxel_size_r[0]*out_size_factor, voxel_size_r[1]*out_size_factor, point_cloud_range[5]-point_cloud_range[2]] #new
numC_Trans = 80

multi_adj_frame_id_cfg = (1, 8+1, 1)


# arch config
embed_dims = 256
num_layers = 6
num_query = 900
num_frames = 8
num_levels = 4
num_points = 4

img_backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    with_cp=False)
img_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=embed_dims,
    num_outs=num_levels)
img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True)

model = dict(
    type='SparseBEV_rc',
    data_aug=dict(
        img_color_aug=True,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    stop_prev_grad=0,
    img_backbone=img_backbone,
    img_neck=img_neck,
    freeze_img=True,
    #radar start
    radar_voxel_layer=dict(
        max_num_points=10, 
        voxel_size=radar_voxel_size, 
        max_voxels=(90000, 120000),
        point_cloud_range=point_cloud_range),
    pts_pillar_layer=dict(
        max_num_points=20,
        voxel_size=pillar_size,
        max_voxels=(30000, 60000),
        point_cloud_range=point_cloud_range),
    # radar_voxel_encoder=dict(
    #     type='HardSimpleVFE',
    #     num_features=7,
    # ),    
    radar_voxel_encoder=dict(
        type='RadarFeatureNetV2',
        in_channels=6+1,
        feat_channels=[32, 64],
        with_distance=False,
        point_cloud_range=point_cloud_range,
        voxel_size=radar_voxel_size,
        norm_cfg=dict(
            type='BN1d',
            eps=1.0e-3,
            momentum=0.01)
    ),
    # radar_voxel_encoder=dict(
    #     type='RadarFeatureNetAdapterNoMaskV2',
    #     return_rcs=False,
    #     in_channels=6+1,
    #     feat_channels=[32, 64],
    #     with_distance=False,
    #     point_cloud_range=point_cloud_range,
    #     voxel_size=radar_voxel_size,
    #     norm_cfg=dict(
    #         type='BN1d',
    #         eps=1.0e-3,
    #         momentum=0.01),
    #     with_pos_embed=True,
    # ),
    radar_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        # output_shape=[128, 128],
        output_shape=[512, 512],
    ),
    # radar_middle_encoder=dict(
    #     type='PointPillarsScatterRCSr2',
    #     in_channels=64,
    #     # output_shape=[128, 128],
    #     output_shape=[512, 512],
    # ),

    radar_bev_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    radar_bev_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    rac=sum([128, 128, 128]),
    #DeformAttn=dict(),
    #radar_reduc_conv=dict(),
    #radar end

    pts_bbox_head=dict(
        type='SparseBEVHead_rc',
        num_classes=10,
        in_channels=embed_dims, #radar
        num_query=num_query,
        query_denoising=True,
        query_denoising_groups=10,
        code_size=10,
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        sync_cls_avg_factor=True,
        transformer=dict(
            type='SparseBEVTransformer_rc',
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=10,
            code_size=10,
            pc_range=point_cloud_range),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            score_threshold=0.05,
            num_classes=10),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=embed_dims // 2,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
        )
    ))
)

dataset_type = 'CustomNuScenesDataset_sparsebev_rc' #new dataset
dataset_root = 'data/nuscenes/'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=True
)


ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': False, #False change
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1),

    dict( #load radar
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=4,
        use_dim=radar_use_dims,
        max_num=1200, 
        rote90=False,
        ),
    

    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage', rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05]), #[-0.3925, 0.3925] [0.95, 1.05]
    dict(type='GlobalRotScaleTrans_radar', is_rad=True),

    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'radar'], meta_keys=(
        'filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp')) #radar
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1, test_mode=True),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),

    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=8, #new
        use_dim=radar_use_dims,
        max_num=1200, 
        rote90=False,),


    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img', 'radar'], meta_keys=(
                'filename', 'box_type_3d', 'ori_shape', 'img_shape', 'pad_shape',
                'lidar2img', 'img_timestamp'))
        ])
]

data = dict(
    #samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_RC_infos_train_sweep.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_RC_infos_val_sweep.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_RC_infos_val_sweep.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.01
)

optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale=512.0,
    grad_clip=dict(max_norm=35, norm_type=2)
)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)
total_epochs = 12
batch_size = 8 #sparse
# load pretrained weights
# load_from =  'pretrain/r50_nuimg_704x256.pth'  #'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'#
load_from = 'work_dirs/pretrain_r50_nuimg_704x256.pth'
revise_keys = None #[('backbone', 'img_backbone')]

# resume the last training
resume_from = None #'/data1/liuzhe/test/bevperception/outputs/SparseBEV_rc/RC/latest.pth'

# checkpointing
checkpoint_config = dict(interval=1, max_keep_ckpts=6)

# logging
# log_config = dict(
#     interval=1,
#     hooks=[
#         dict(type='MyTextLoggerHook', interval=1, reset_flag=True),
#         dict(type='MyTensorboardLoggerHook', interval=500, reset_flag=True)
#     ]
# )

# evaluation
#evaluation = dict(interval=4)
eval_config = dict(interval=4)

# other flags
debug = False