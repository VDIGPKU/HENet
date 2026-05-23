import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import ipdb
from math import cos, sin


def randerobj(self, path, points):
    s = 0.03  # size of points
    bias = [
        [0, 0, s],
        [s, 0, s],
        [0, s, s],
        [s, s, s],
        [0, s, 0],
        [s, s, 0],
        [0, 0, 0],
        [s, 0, 0],
    ]
    link = [
        [1, 2, 3],
        [3, 2, 4],
        [3, 4, 5],
        [5, 4, 6],
        [5, 6, 7],
        [7, 6, 8],
        [7, 8, 1],
        [1, 8, 2],
        [2, 8, 4],
        [4, 8, 6],
        [7, 1, 5],
        [5, 1, 3],
    ]
    with open(path, 'w') as f:
        for i in range(points.shape[0]):
            x = points[i][0]
            y = points[i][1]
            z = points[i][2]
            for index in range(8):
                f.write("v " +
                        str(x.item() + bias[index][0]) + " " +
                        str(y.item() + bias[index][1]) + " " +
                        str(z.item() + bias[index][2]) + "\n")
            for index in range(12):
                f.write("f " +
                        str(link[index][0] + i * 8) + " " +
                        str(link[index][1] + i * 8) + " " +
                        str(link[index][2] + i * 8) + "\n")


def vis(points, gt_bboxes_3d, gt_labels_3d, img):
    aim_class = 2  # 2 = construction_vehicle

    randstr = str(torch.rand(1))

    points = points[0][:, :3]
    gt_bboxes = gt_bboxes_3d[0]
    gt_labels = gt_labels_3d[0]

    right_bboxes_ind = (gt_labels == aim_class).nonzero()
    if right_bboxes_ind.shape[0] == 0:
        return

    right_bboxes = gt_bboxes[right_bboxes_ind.view(-1)]

    import cv2
    img_id = 0
    for eachimg in img[0]:
        path = "/home/xiazhongyu/vis/construction_vehicle-frameid" + randstr + \
               "-imgid" + str(img_id) + ".png"
        img_to_draw = eachimg.permute(1, 2, 0).cpu().numpy()

        cv2.imwrite(path, img_to_draw)
        img_id += 1


def print_pcgt_on_bev(pc, gt, name, app):
    points = pc

    if gt.shape[0] > 0:
        x = gt[:, 0].reshape(-1)
        y = gt[:, 1].reshape(-1)
        z = gt[:, 2].reshape(-1)
        dx = gt[:, 3].reshape(-1)
        dy = gt[:, 4].reshape(-1)
        dz = gt[:, 5].reshape(-1)
        yaw = gt[:, 6].reshape(-1)
        ll = x.shape[0]

    fig = plt.figure(figsize=(32, 32))
    plt.plot([100, 100, -100, -100, 100], [100, -100, -100, 100, 100], lw=0.5)
    plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)

    xxx = points[:, 0]
    yyy = points[:, 1]
    zzz = points[:, 2]
    plt.scatter(xxx, yyy, s=0.5)

    if gt.shape[0] > 0:
        for i in range(ll):
            xcos = dx[i] * cos(yaw[i]) / 2
            xsin = dx[i] * sin(yaw[i]) / 2
            ycos = dy[i] * cos(yaw[i]) / 2
            ysin = dy[i] * sin(yaw[i]) / 2
            x1 = x[i] + xcos - ysin
            y1 = y[i] + xsin + ycos
            x2 = x[i] + xcos + ysin
            y2 = y[i] + xsin - ycos
            x3 = x[i] - xcos + ysin
            y3 = y[i] - xsin - ycos
            x4 = x[i] - xcos - ysin
            y4 = y[i] - xsin + ycos
            plt.plot([x1,x2,x3,x4,x1], [y1,y2,y3,y4,y1], lw=0.5)

    plt.savefig("/home/xiazhongyu/BEVFusion/utils/vis/" + name[18:-8] + app + ".png")


def print_pc_on_bev(pc, name, app):
    points = pc
    # print(points[0:5])

    fig = plt.figure(figsize=(16, 16))
    plt.plot([100, 100, -100, -100, 100], [100, -100, -100, 100, 100], lw=0.5)
    plt.plot([50,50,-50,-50,50], [50,-50,-50,50,50], lw=0.5)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    plt.scatter(x, y, s=2)
    plt.savefig("/home/xiazhongyu/BEVFusion/utils/vis/" + name[18:-8] + app + ".png")


def print_gt_on_bev(gt, lid, app):

    if gt.shape[0] == 0:
        return

    names = []
    for lidar_info in lid:
        name = lidar_info['LIDAR_TOP']['filename']
        name = name[18:-8]
        names.append(name)

    print(gt)
    x = gt[:, 0].reshape(-1)
    y = gt[:, 1].reshape(-1)
    z = gt[:, 2].reshape(-1)
    dx = gt[:, 3].reshape(-1)
    dy = gt[:, 4].reshape(-1)
    dz = gt[:, 5].reshape(-1)
    yaw = gt[:, 6].reshape(-1)
    ll = x.shape[0]

    fig = plt.figure(figsize=(16, 16))

    plt.plot([50,50,-50,-50,50], [50,-50,-50,50,50], lw=0.5)

    for i in range(ll):
        x1 = x[i] + dy[i] * cos(yaw[i]) / 2 - dx[i] * sin(yaw[i]) / 2
        y1 = y[i] + dy[i] * sin(yaw[i]) / 2 + dx[i] * cos(yaw[i]) / 2
        x2 = x[i] + dy[i] * cos(yaw[i]) / 2 + dx[i] * sin(yaw[i]) / 2
        y2 = y[i] + dy[i] * sin(yaw[i]) / 2 - dx[i] * cos(yaw[i]) / 2
        x3 = x[i] - dy[i] * cos(yaw[i]) / 2 + dx[i] * sin(yaw[i]) / 2
        y3 = y[i] - dy[i] * sin(yaw[i]) / 2 - dx[i] * cos(yaw[i]) / 2
        x4 = x[i] - dy[i] * cos(yaw[i]) / 2 - dx[i] * sin(yaw[i]) / 2
        y4 = y[i] - dy[i] * sin(yaw[i]) / 2 + dx[i] * cos(yaw[i]) / 2
        plt.plot([x1,x2,x3,x4,x1], [y1,y2,y3,y4,y1], lw=0.5)

    for i in names:
        plt.savefig("/home/xiazhongyu/BEVFusion/utils/vis/" + i + app + "_gt.png")


def print_bev(pts_feats, name):
    fig = plt.figure(figsize=(16, 16))
    # plt.plot([100, 100, -100, -100, 100], [100, -100, -100, 100, 100], lw=0.5)
    # plt.plot([50,50,-50,-50,50], [50,-50,-50,50,50], lw=0.5)
    print('\n******** BEGIN PRINT **********\n')
    for idx in range(pts_feats.shape[0]):
        max_feat = 0
        pts_feat = pts_feats[idx].cpu().detach().numpy() # 放到cpu上 去掉梯度 转换成numpy
        print('pts_feat.shape =', pts_feat.shape)
        feat_2d = np.zeros(pts_feat.shape[1:])
        print('feat_2d.shape =', feat_2d.shape)
        for h in range(feat_2d.shape[0]):
            for w in range(feat_2d.shape[1]):
                for c in range(pts_feat.shape[0]):
                    feat_2d[h][w] += abs(pts_feat[c][h][w])
        # for h in range(feat_2d.shape[0]):
        #     for w in range(feat_2d.shape[1]):
        #         if feat_2d[h][w] != 0:
        #             feat_2d[h][w] = 255
        # ipdb.set_trace()
        plt.imshow(feat_2d, cmap=plt.cm.gray, vmin=0, vmax=255)
        plt.savefig("/home/wangxinhao/BEVFusion/utils/vis_BACK/" + name + '-' + str(idx) + ".png")
        # plt.savefig("/root/BEVFusion/utils/vis_FRONT_LEFT/" + name + str(idx) + ".png")
    print('\n******** END PRINT **********\n')


def my_print_gt_on_bev(gt, img_metas, name):
    print('\n******** BEGIN PRINT GT**********\n')
    gt = gt.tensor
    if gt.shape[0] == 0:
        return

    print(gt)
    x = gt[:, 0].reshape(-1)
    y = gt[:, 1].reshape(-1)
    z = gt[:, 2].reshape(-1)
    dx = gt[:, 3].reshape(-1)
    dy = gt[:, 4].reshape(-1)
    dz = gt[:, 5].reshape(-1)
    yaw = gt[:, 6].reshape(-1)
    ll = x.shape[0]

    fig = plt.figure(figsize=(16, 16))

    plt.plot([50,50,-50,-50,50], [50,-50,-50,50,50], lw=0.5)

    for i in range(ll):
        x1 = x[i] + dy[i] * cos(yaw[i]) / 2 - dx[i] * sin(yaw[i]) / 2
        y1 = y[i] + dy[i] * sin(yaw[i]) / 2 + dx[i] * cos(yaw[i]) / 2
        x2 = x[i] + dy[i] * cos(yaw[i]) / 2 + dx[i] * sin(yaw[i]) / 2
        y2 = y[i] + dy[i] * sin(yaw[i]) / 2 - dx[i] * cos(yaw[i]) / 2
        x3 = x[i] - dy[i] * cos(yaw[i]) / 2 + dx[i] * sin(yaw[i]) / 2
        y3 = y[i] - dy[i] * sin(yaw[i]) / 2 - dx[i] * cos(yaw[i]) / 2
        x4 = x[i] - dy[i] * cos(yaw[i]) / 2 - dx[i] * sin(yaw[i]) / 2
        y4 = y[i] - dy[i] * sin(yaw[i]) / 2 + dx[i] * cos(yaw[i]) / 2
        plt.plot([x1,x2,x3,x4,x1], [y1,y2,y3,y4,y1], lw=0.5)

    plt.savefig("/home/wangxinhao/BEVFusion/utils/vis_gt_fusion/" + name + '-' + img_metas['sample_idx'] + ".png")
    print('\n******** END PRINT GT**********\n')


def print_gt_and_bev(pts_feats, gt, img_metas, name):
    print('\n******** BEGIN PRINT GT **********\n')
    assert(pts_feats.shape[0] == len(gt))
    for idx in range(len(gt)):
        # if img_metas[idx]['sample_idx'] != 'ca9a282c9e77460f8360f564131a8af5':
            # continue
        corner = gt[idx].corners

        fig = plt.figure(figsize=(16, 16))

        plt.plot([50,50,-50,-50,50], [50,-50,-50,50,50], lw=0.5)
        """
        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

        .. code-block:: none

                                            up z
                            front x           ^
                                    /            |
                                /             |
                    (x1, y0, z1) + -----------  + (x1, y1, z1)
                                /|            / |
                                / |           /  |
                (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / oriign    | /
            left y<-------- + ----------- + (x0, y1, z0)
                (x0, y0, z0)
        """
        for i in range(corner.shape[0]):
            x1 = corner[i][0][0]
            y1 = corner[i][0][1]
            x2 = corner[i][2][0]
            y2 = corner[i][2][1]
            x3 = corner[i][6][0]
            y3 = corner[i][6][1]
            x4 = corner[i][4][0]
            y4 = corner[i][4][1]
            plt.plot([x1,x2,x3,x4,x1], [y1,y2,y3,y4,y1], lw=0.5)

        pts_feat = pts_feats[idx].cpu().detach().numpy() # 放到cpu上 去掉梯度 转换成numpy
        print('pts_feat.shape =', pts_feat.shape)
        feat_2d = np.zeros(pts_feat.shape[1:])
        print('feat_2d.shape =', feat_2d.shape)
        for h in range(feat_2d.shape[0]):
            for w in range(feat_2d.shape[1]):
                for c in range(pts_feat.shape[0]):
                    feat_2d[h][w] += abs(pts_feat[c][h][w])

        # plt.imshow(feat_2d, cmap=plt.cm.gray, vmin=0, vmax=255, extent=[-50, 50, -50, 50], origin = 'lower') # extent=（left,right,bottom, top） # 注意图片坐标原点！
        plt.savefig("/home/wangxinhao/BEVFusion/utils/vis_pred/" + name + '-' + img_metas[idx]['sample_idx'] + ".png")
    print('\n******** END PRINT GT **********\n')
