'''
--w-channel-wise --w-symmetric --a-symmetric --w-quantizer=LSQPlusQuantizer --a-quantizer=LSQPlusQuantizer
please use train_sparse_quant.sh, not train_quant.sh
'''

_base_ = ['../r50_nuimg_704x256_rcbevdet.py']

batch_size = 4 #sparse

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.01
)

load_from = 'work_dirs/SparseBEV_rc_r50_nuimg_704x256_rcbevdet_epoch_12.pth'

model = dict(freeze_img=False)

batch_size = 8 #sparse
optimizer = dict(lr=1e-5)

eval_config = dict(interval=2)
total_epochs = 12
checkpoint_config = dict(interval=1, max_keep_ckpts=6)
resume_from = 'work_dirs/SparseBEV_rc/r50_nuimg_704x256_rcbevdet_lsqp88/epoch_6.pth'