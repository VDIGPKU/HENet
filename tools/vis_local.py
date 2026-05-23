import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import plotly.graph_objects as go


color_map = ['orange', 'pink', 'yellow', 'blue', 'cyan',
             'darkorange', 'red', 'lightyellow', 'brown', 'purple',
             'darkred', 'violet', 'indigo', 'lightgreen', 'snow',
             'darkcyan', 'green',]


def draw_xiazhongyu(vis_dict, path):
    color = {
        0: 'darkblue',
        1: 'darkolivegreen',
        2: 'sienna',
        3: 'darkorange',
        4: 'darkslateblue',
        5: 'darkslategray',
        6: 'darkcyan',
        7: 'darkgreen',
        8: 'darkviolet',
        9: 'darkgoldenrod',
    }

    img_inputs = vis_dict['img_inputs']
    imgs_ori = vis_dict['imgs_ori']
    gt_bbox_corners = vis_dict['gt_bbox_corners']
    gt_bbox_labels = vis_dict['gt_bbox_labels']
    gt_seg = vis_dict['gt_seg']
    pred_bbox_corners = vis_dict['pred_bbox_corners']
    pred_bbox_labels = vis_dict['pred_bbox_labels']
    pred_seg = vis_dict['pred_seg']

    imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, _ = img_inputs[0]

    fig = plt.figure(figsize=(24, 21))
    gs = fig.add_gridspec(18 + 24, 16 * 3)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    f_ax1 = fig.add_subplot(gs[0:9, 0:16])
    f_ax2 = fig.add_subplot(gs[0:9, 16:16 * 2])
    f_ax3 = fig.add_subplot(gs[0:9, 16 * 2:16 * 3])
    f_ax4 = fig.add_subplot(gs[9:18, 0:16])
    f_ax5 = fig.add_subplot(gs[9:18, 16:16 * 2])
    f_ax6 = fig.add_subplot(gs[9:18, 16 * 2:16 * 3])
    f_ax7 = fig.add_subplot(gs[18:18 + 24, 0:24])
    f_ax8 = fig.add_subplot(gs[18:18 + 24, 24:24 * 2])
    for ax in [f_ax1, f_ax2, f_ax3, f_ax4, f_ax5, f_ax6, f_ax7, f_ax8]:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    ### vis bbox on img ###
    bbox_corners = pred_bbox_corners
    bbox_labels = pred_bbox_labels
    for img, ax, sensor2ego, intrin in zip(imgs_ori[0],
                                           [f_ax1, f_ax2, f_ax3, f_ax6, f_ax5, f_ax4],
                                           sensor2egos[0],
                                           intrins[0]):

        # obtain ego to image transformation matrix
        ego2cam_rt = torch.inverse(sensor2ego).T
        intrinsic = intrin
        viewpad = torch.eye(4).to(ego2cam_rt.device)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        ego2cam_rt = (viewpad @ ego2cam_rt.T)

        num_bbox = bbox_corners.shape[0]
        pred_bbox_corners_4d = torch.cat([bbox_corners, torch.ones([num_bbox, 8, 1])], dim=-1).to(ego2cam_rt.device)
        cors_2d = pred_bbox_corners_4d.view(-1, 4) @ ego2cam_rt.T
        cors_2d = cors_2d.cpu()
        cors_2d[:, 0] /= torch.abs(cors_2d[:, 2])
        cors_2d[:, 1] /= torch.abs(cors_2d[:, 2])
        cors_2d[:, 0] *= 1152 / 1600
        cors_2d[:, 1] *= 640 / 900
        cors_2d = cors_2d.view(-1, 8, 4)

        for i in range(len(cors_2d)):
            cor = cors_2d[i]
            c = color[int(bbox_labels[i])]
            pair = [[0, 1], [0, 3], [1, 2], [2, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4, 7], [6, 7], [5, 6], [5, 4]]
            for p in pair:
                if cor[p[0], 2] > 0 and cor[p[1], 2] > 0:
                    x1 = cor[p[0], 0]
                    y1 = cor[p[0], 1]
                    x2 = cor[p[1], 0]
                    y2 = cor[p[1], 1]
                    ax.plot([x1, x2], [y1, y2], c=c)
                else:
                    x1 = cor[p[0], 0]
                    y1 = cor[p[0], 1]
                    z1 = torch.abs(cor[p[0], 2])
                    x2 = cor[p[1], 0]
                    y2 = cor[p[1], 1]
                    z2 = torch.abs(cor[p[1], 2])
                    x0 = x1 + (x2 - x1) * (z1 / (z1 + z2))
                    y0 = y1 + (y2 - y1) * (z1 / (z1 + z2))
                    if cor[p[0], 2] > 0:
                        ax.plot([x1, x0], [y1, y0], c=c)
                    elif cor[p[1], 2] > 0:
                        ax.plot([x2, x0], [y2, y0], c=c)

        ax.imshow(img[0].cpu())

    ### vis gt ###
    for i in range(gt_bbox_corners.shape[0]):
        x1 = gt_bbox_corners[i][0][0]
        y1 = gt_bbox_corners[i][0][1]
        x2 = gt_bbox_corners[i][2][0]
        y2 = gt_bbox_corners[i][2][1]
        x3 = gt_bbox_corners[i][6][0]
        y3 = gt_bbox_corners[i][6][1]
        x4 = gt_bbox_corners[i][4][0]
        y4 = gt_bbox_corners[i][4][1]
        f_ax7.set_ylim(ymin=-50, ymax=50)
        f_ax7.set_xlim(xmin=-50, xmax=50)
        # f_ax7.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1],
        #            c=color[int(gt_bbox_labels[i])],
        #            lw=2)  # facing right
        f_ax7.plot([-y1, -y2, -y3, -y4, -y1], [x1, x2, x3, x4, x1],
                   c=color[int(gt_bbox_labels[i])],
                   lw=2)  # facing up
    f_ax7.patch.set_facecolor('silver')
    f_ax7.patch.set_alpha(0.5)
    for xx in range(200):
        for yy in range(200):
            # xc = -50 + xx * 0.5  # facing right
            # yc = -50 + yy * 0.5  # facing right
            xc = -(-50 + yy * 0.5)  # facing up
            yc = -50 + xx * 0.5  # facing up
            # 0 vehicle, 1 可行驶区域, 2 车道线
            if gt_seg[1, xx, yy] == 1:
                f_ax7.add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.5, facecolor="green"))
            if gt_seg[2, xx, yy] == 1:
                f_ax7.add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.5, facecolor="red"))
            if gt_seg[0, xx, yy] == 1:
                f_ax7.add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.5, facecolor="blue"))

    ### vis pred ###
    for i in range(pred_bbox_corners.shape[0]):
        x1 = pred_bbox_corners[i][0][0]
        y1 = pred_bbox_corners[i][0][1]
        x2 = pred_bbox_corners[i][2][0]
        y2 = pred_bbox_corners[i][2][1]
        x3 = pred_bbox_corners[i][6][0]
        y3 = pred_bbox_corners[i][6][1]
        x4 = pred_bbox_corners[i][4][0]
        y4 = pred_bbox_corners[i][4][1]
        f_ax8.set_ylim(ymin=-50, ymax=50)
        f_ax8.set_xlim(xmin=-50, xmax=50)
        # f_ax8.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1],
        #            c=color[int(pred_bbox_labels[i])],
        #            lw=2)  # facing right
        f_ax8.plot([-y1, -y2, -y3, -y4, -y1], [x1, x2, x3, x4, x1],
                   c=color[int(pred_bbox_labels[i])],
                   lw=2)  # facing up
    f_ax8.patch.set_facecolor('silver')
    f_ax8.patch.set_alpha(0.5)
    for xx in range(200):
        for yy in range(200):
            # xc = -50 + xx * 0.5  # facing right
            # yc = -50 + yy * 0.5  # facing right
            xc = -(-50 + yy * 0.5)  # facing up
            yc = -50 + xx * 0.5  # facing up
            # 0 vehicle, 1 可行驶区域, 2 车道线
            if pred_seg[1, xx, yy] > 0.45:
                f_ax8.add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.5, facecolor="green"))
            if pred_seg[2, xx, yy] > 0.40:
                f_ax8.add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.5, facecolor="red"))
            if pred_seg[0, xx, yy] > 0.45:
                f_ax8.add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.5, facecolor="blue"))

    plt.savefig(path)
    print("vis saved as " + path)


def draw_bbox(gt_boxes, gt_labels, path):
    color = {
        0: 'darkblue',
        1: 'darkolivegreen',
        2: 'sienna',
        3: 'darkorange',
        4: 'darkslateblue',
        5: 'darkslategray',
        6: 'darkcyan',
        7: 'darkgreen',
        8: 'darkviolet',
        9: 'darkgoldenrod',
    }
    gt_bbox_corners = gt_boxes
    gt_bbox_labels = gt_labels
    fig = plt.figure(figsize=(24, 21))
    for i in range(gt_bbox_corners.shape[0]):
        if gt_bbox_labels[i] == -1:
            continue
        x1 = gt_bbox_corners[i][0][0]
        y1 = gt_bbox_corners[i][0][1]
        x2 = gt_bbox_corners[i][2][0]
        y2 = gt_bbox_corners[i][2][1]
        x3 = gt_bbox_corners[i][6][0]
        y3 = gt_bbox_corners[i][6][1]
        x4 = gt_bbox_corners[i][4][0]
        y4 = gt_bbox_corners[i][4][1]
        plt.ylim(ymin=-50, ymax=50)
        plt.xlim(xmin=-50, xmax=50)
        # f_ax7.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1],
        #            c=color[int(gt_bbox_labels[i])],
        #            lw=2)  # facing right
        plt.plot([-y1, -y2, -y3, -y4, -y1], [x1, x2, x3, x4, x1],
                   c=color[int(gt_bbox_labels[i])],
                   lw=2)  # facing up
    plt.savefig(path)


def draw_xiazhongyu_occ(vis_dict, path):

    imgs_ori = vis_dict['imgs_ori']
    occ_gt = vis_dict['occ_gt']
    occ_pred = vis_dict['occ_pred']

    fig = plt.figure(figsize=(24, 21))
    gs = fig.add_gridspec(18 + 24, 16 * 3)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    f_ax1 = fig.add_subplot(gs[0:9, 0:16])
    f_ax2 = fig.add_subplot(gs[0:9, 16:16 * 2])
    f_ax3 = fig.add_subplot(gs[0:9, 16 * 2:16 * 3])
    f_ax4 = fig.add_subplot(gs[9:18, 0:16])
    f_ax5 = fig.add_subplot(gs[9:18, 16:16 * 2])
    f_ax6 = fig.add_subplot(gs[9:18, 16 * 2:16 * 3])
    f_ax7 = fig.add_subplot(gs[18:18 + 24, 0:24], projection='3d')
    f_ax8 = fig.add_subplot(gs[18:18 + 24, 24:24 * 2], projection='3d')
    for ax in [f_ax1, f_ax2, f_ax3, f_ax4, f_ax5, f_ax6]:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    ### vis img ###
    for img, ax in zip(imgs_ori[0], [f_ax1, f_ax2, f_ax3, f_ax6, f_ax5, f_ax4]):
        ax.imshow(img[0].cpu())

    ### vis occ ###
    # color_map = ['orange', 'pink', 'yellow', 'blue', 'cyan',
    #              'darkorange', 'red', 'lightyellow', 'brown', 'purple',
    #              'darkred', 'violet', 'indigo', 'lightgreen', 'snow',
    #              'darkcyan', 'green',]
    for occ, ax in zip([occ_gt, occ_pred], [f_ax7, f_ax8]):
        ax.set_box_aspect([200, 200, 16])
        ax.set_axis_off()
        ax.w_zaxis.line.set_visible(False)
        ax.set_zticks([])
        ax.view_init(elev=35, azim=180)

        voxels = (occ < 17)
        colors = np.empty(voxels.shape, dtype=object)
        for i in range(17):
            colors[occ == i] = color_map[i]
        ax.voxels(voxels, facecolors=colors, edgecolor='silver', linewidth=0.05)

        #ax.voxels(voxels)

    plt.savefig(path)
    print("vis saved as " + path)


def draw_occ(occ, name):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([200, 200, 16])
    ax.set_axis_off()
    ax.w_zaxis.line.set_visible(False)
    ax.set_zticks([])
    ax.view_init(elev=35, azim=180)

    voxels = (occ < 17)
    colors = np.empty(voxels.shape, dtype=object)
    for i in range(17):
        if np.isin(i, occ):
            colors[occ == i] = color_map[i]
    ax.voxels(voxels, facecolors=colors, edgecolor='silver', linewidth=0.05)
    plt.savefig(name)
    print('save at', name)


def draw_occ_orinpy(occ, name):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([200, 200, 16])
    ax.set_axis_off()
    ax.w_zaxis.line.set_visible(False)
    ax.set_zticks([])
    ax.view_init(elev=35, azim=180)

    voxels = (occ > 0)
    colors = np.empty(voxels.shape, dtype=object)
    for i in range(17):
        if np.isin(i, occ):
            colors[occ == i] = color_map[i]
    ax.voxels(voxels, facecolors=colors, edgecolor='silver', linewidth=0.05)
    plt.savefig(name)
    print('save at', name)


def draw_occ_html(occ_grid):
    ### vis occ ###
    x, y, z = np.indices(occ_grid.shape)

    mask = occ_grid < 17
    xv, yv, zv, val = x[mask], y[mask], z[mask], occ_grid[mask]

    fig = go.Figure(data=go.Scatter3d(
        x=xv.flatten(),
        y=yv.flatten(),
        z=zv.flatten(),
        mode='markers',
        marker=dict(
            size=1,  # 球体大小
            color=[color_map[v] for v in val.flatten()],
            opacity=0.5,
            symbol='circle'
        )
    ))

    # 保存交互式 HTML
    fig.update_layout(scene_aspectmode="data")
    fig.write_html("/home/chenwenhao/voxel_3d.html")

