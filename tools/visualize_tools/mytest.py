#%%
import pickle


#%%
# 读取pkl文件
with open('/home/yanghansong/bevperception/data_batch-0.pkl', 'rb') as file:
    data = pickle.load(file)


#%%
# 取出一个batch中的一条数据和对应的gt_bboxes_3d
points = data['points'].data[0][0]
gt_bboxes_3d = data['gt_bboxes_3d'].data[0][0]

#%%
# 保存为pkl文件，可用于本地加载
# data_dict = {
#     'points':points,
#     'YAW_AXIS':gt_bboxes_3d.YAW_AXIS,
#     'bev':gt_bboxes_3d.bev,
#     'bottom_center':gt_bboxes_3d.bottom_center,
#     'bottom_height':gt_bboxes_3d.bottom_height,
#     'box_dim':gt_bboxes_3d.box_dim,
#     'center':gt_bboxes_3d.center,
#     'corners':gt_bboxes_3d.corners,
#     'dims':gt_bboxes_3d.dims,
#     'device':gt_bboxes_3d.device,
#     'gravity_center':gt_bboxes_3d.gravity_center,
#     'nearest_bev':gt_bboxes_3d.nearest_bev,
#     'tensor':gt_bboxes_3d.tensor,
#     'top_height':gt_bboxes_3d.top_height,
#     'volume':gt_bboxes_3d.volume,
#     'with_yaw':gt_bboxes_3d.with_yaw,
#     'yaw':gt_bboxes_3d.yaw
# }

# import pickle
# with open('/home/yanghansong/bevperception/data_batch_dict.pkl', 'wb') as file:
#     pickle.dump(data_dict, file)

#%%
