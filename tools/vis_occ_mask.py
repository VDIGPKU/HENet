import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


color_map = ['orange', 'pink', 'yellow', 'blue', 'cyan',
             'darkorange', 'red', 'lightyellow', 'brown', 'purple',
             'darkred', 'violet', 'indigo', 'lightgreen', 'snow',
             'darkcyan', 'green',]

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


def draw_occ_mask(occ, name='mask.jpg'):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([200, 200, 16])
    ax.set_axis_off()
    ax.w_zaxis.line.set_visible(False)
    ax.set_zticks([])
    ax.view_init(elev=35, azim=180)

    voxels = (occ > 0)
    colors = np.empty(voxels.shape, dtype=object)
    if np.isin(1, occ):
        colors[occ == 1] = 'darkred'
    if np.isin(2, occ):
        colors[occ == 2] = 'darkorange'
    if np.isin(3, occ):
        colors[occ == 3] = 'orange'
    if np.isin(4, occ):
        colors[occ == 4] = 'lightgreen'
    colors[occ >= 5] = 'darkgreen'
    ax.voxels(voxels, facecolors=colors, edgecolor='silver', linewidth=0.05)
    plt.savefig(name)
    print('save at', name)




if __name__ == '__main__':
    occ = np.load('CD701_000088_2024-08-21_16-24-26_30222158-30241658/gt_occ/30222158.npy')
    draw_occ_orinpy(occ, 'occ.jpg')
    occ = np.load('CD701_000088_2024-08-21_16-24-26_30222158-30241658/gt_occ_mask/30222158.npy')
    draw_occ_mask(occ, 'occmask.jpg')
