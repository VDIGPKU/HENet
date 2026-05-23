import torch

a = torch.load('work_dirs/transfusion_nusc_voxel_L_bs4_3layer_pretrain_syncbn_40ep_after30_300fintune/epoch_40.pth')
ckpt = a['state_dict']

key = list(ckpt.keys())

for k in key:
    ckpt['student.'+k] = ckpt[k]
    del ckpt[k]

a['state_dict'] = ckpt

torch.save(a, 'work_dirs/transfusion_nusc_voxel_L_bs4_3layer_pretrain_syncbn_40ep_after30_300fintune/semi_epoch_40.pth')
