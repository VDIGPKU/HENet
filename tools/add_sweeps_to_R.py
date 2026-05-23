import mmcv


for split in ['train', 'val']:
    a = mmcv.load(f'data/nuscenes/nuscenes_infos_{split}.pkl')
    info = a['infos']

    b = mmcv.load(f'data/nuscenes/nuscenes_R_infos_{split}_origin.pkl')
    info_R = b['infos']

    info_R_update = []

    for i in range(len(info)):
        tmp = info_R[i]
        tmp['sweeps'] = info[i]['sweeps']
        info_R_update.append(tmp)

    b['infos'] = info_R_update

    mmcv.dump(b, f'data/nuscenes/nuscenes_R_infos_{split}.pkl')
