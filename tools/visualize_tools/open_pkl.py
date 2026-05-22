import pickle

#%%
# 读取pkl文件
with open('/home/yanghansong/bevperception/data/nuscenes/nuscenes_R_with_occ_path_infos_train.pkl', 'rb') as file:
    data1 = pickle.load(file)



print(data1)
# print(data)

