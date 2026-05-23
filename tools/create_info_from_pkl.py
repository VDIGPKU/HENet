import mmcv
import numpy as np
from sklearn.cluster import KMeans

def format_decimal(num):
    a, b = str(num).split('.')
    return float(a + '.' + b[:2])

def get_jnb_threshold(score_list):
    kclf = KMeans(n_clusters=2)
    data_kmeans = np.array(score_list)
    data_kmeans = data_kmeans.reshape(len(data_kmeans), -1)
    kclf.fit(data_kmeans)
    threshod = kclf.cluster_centers_.reshape(-1)
    res = np.sort(threshod)[::-1]
    res = [format_decimal(r) for r in res]
    return res  

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data = mmcv.load('/data2/linzhiwei/project/bevfusion-git/data/nuscenes/nuscenes_infos_val.pkl')

data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))


pred = mmcv.load('/data2/linzhiwei/project/bevfusion-git/work_dirs/bevf_tf_2x8_cul2_10e_nusc_aug_3layer_300proposal_new_cam/20230424_193531_test/out.pkl')

assert len(pred)==len(data_infos)

prog_bar = mmcv.ProgressBar(len(pred))

score_thr_sum = 0
low_score_thr_sum = 0

for i in range(len(pred)):
    p = pred[i]['pts_bbox']
    info = data_infos[i]

    box = p['boxes_3d']
    score = p['scores_3d']
    label = p['labels_3d']
    # print(box[label.sort()[1]])
    # exit()
    # if len(score)>2:
    #     score_thr = get_jnb_threshold(score)
    #     score_thr = score_thr[0]
    #     low_score_thr = score_thr[0]
    #     # print(score_thr)
    # else:
    #     score_thr = 0.0
    #     low_score_thr = 0.0

    assert len(score)>2
    score_thr = get_jnb_threshold(score)
    score_thr, low_score_thr = score_thr
    # low_score_thr = score_thr[1]

    # print(score_thr)
    keep = score > score_thr
    low_keep = (score > low_score_thr) & (score < score_thr)

    score_thr_sum += score_thr
    low_score_thr_sum += low_score_thr


    label_name = []
    new_box = []

    num_lidar_pts = []
    valid_flag = []
    p_weight = []

    high_box = box[keep]
    high_label = label[keep]
    for j in range(len(high_box)):

        b = high_box[j].tensor[0].numpy()
        b[2] += 0.5 * b[5]
        new_box.append(b)
        label_name.append(class_names[high_label[j]])

        num_lidar_pts.append(100)
        valid_flag.append(True)
        p_weight.append(1.0)
    
    low_box = box[low_keep]
    low_label = label[low_keep]
    low_score = score[low_keep]
    for j in range(len(low_box)):

        b = low_box[j].tensor[0].numpy()
        b[2] += 0.5 * b[5]
        new_box.append(b)
        label_name.append(class_names[low_label[j]])

        num_lidar_pts.append(100)
        valid_flag.append(True)
        p_weight.append(low_score[j])
    
    new_box = np.array(new_box)
    
    data_infos[i]['gt_boxes'] = new_box[:,:7]
    data_infos[i]['gt_names'] = np.array(label_name)
    data_infos[i]['gt_velocity'] = new_box[:,7:]

    data_infos[i]['num_lidar_pts'] = np.array(num_lidar_pts)
    data_infos[i]['valid_flag'] = np.array(valid_flag)

    data_infos[i]['p_weight'] = np.array(p_weight)

    if (i+1)%50 == 0:
        print(score_thr_sum/(i+1), low_score_thr_sum/(i+1))

    prog_bar.update()

data['infos'] = data_infos

mmcv.dump(data, '/data2/linzhiwei/project/bevfusion-git/data/nuscenes/nuscenes_infos_val_semi_dynamic_v2.pkl')

# class_names = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
# ]

# data = mmcv.load('/data2/linzhiwei/project/bevfusion-git/data/nuscenes/nuscenes_infos_val.pkl')

# data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))


# pred = mmcv.load('/data2/linzhiwei/project/bevfusion-git/work_dirs/bevf_tf_2x8_cul2_10e_nusc_aug_3layer_300proposal_new_cam/20230424_193531_test/out.pkl')

# assert len(pred)==len(data_infos)

# prog_bar = mmcv.ProgressBar(len(pred))

# score_thr_sum = 0

# for i in range(len(pred)):
#     p = pred[i]['pts_bbox']
#     info = data_infos[i]

#     box = p['boxes_3d']
#     score = p['scores_3d']
#     label = p['labels_3d']

#     label_name = []
#     new_box = []

#     num_lidar_pts = []
#     valid_flag = []

#     for j in range(len(box)):

#         b = box[j].tensor[0].numpy()
#         b[2] += 0.5 * b[5]
#         new_box.append(b)
#         label_name.append(class_names[label[j]])

#         num_lidar_pts.append(100)
#         valid_flag.append(True)
    
#     new_box = np.array(new_box)
    
#     data_infos[i]['gt_boxes'] = new_box[:,:7]
#     data_infos[i]['gt_names'] = np.array(label_name)
#     data_infos[i]['gt_velocity'] = new_box[:,7:]

#     data_infos[i]['num_lidar_pts'] = np.array(num_lidar_pts)
#     data_infos[i]['valid_flag'] = np.array(valid_flag)

#     if (i+1)%50 == 0:
#         print(score_thr_sum/(i+1))

#     prog_bar.update()

# data['infos'] = data_infos

# mmcv.dump(data, '/data2/linzhiwei/project/bevfusion-git/data/nuscenes/nuscenes_infos_val_semi_all.pkl')
