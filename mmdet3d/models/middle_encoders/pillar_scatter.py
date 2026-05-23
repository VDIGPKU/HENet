
import torch
from mmcv.runner import auto_fp16
from torch import nn

from ..builder import MIDDLE_ENCODERS

from mmdet3d.core import draw_heatmap_gaussian, draw_heatmap_gaussian_feat


@MIDDLE_ENCODERS.register_module()
class PointPillarsScatter(nn.Module):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, in_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.in_channels = in_channels
        self.fp16_enabled = False

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size=None):
        """Foraward function to scatter features."""
        # TODO: rewrite the function in a batch manner
        # no need to deal with different batch cases
        if batch_size is not None:
            return self.forward_batch(voxel_features, coors, batch_size)
        else:
            return self.forward_single(voxel_features, coors)

    def forward_single(self, voxel_features, coors):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates of each voxel.
                The first column indicates the sample ID.
        """
        # Create the canvas for this sample
        canvas = torch.zeros(
            self.in_channels,
            self.nx * self.ny,
            dtype=voxel_features.dtype,
            device=voxel_features.device)

        indices = coors[:, 2] * self.nx + coors[:, 3]
        indices = indices.long()
        voxels = voxel_features.t()
        # Now scatter the blob back to the canvas.
        canvas[:, indices] = voxels
        # Undo the column stacking to final 4-dim tensor
        canvas = canvas.view(1, self.in_channels, self.ny, self.nx)
        return canvas

    def forward_trt(self, voxel_features, coors):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates of each voxel.
                The first column indicates the sample ID.
        """
        # Create the canvas for this sample
        canvas = torch.zeros(
            self.in_channels,
            self.nx * self.ny,
            dtype=voxel_features.dtype,
            device=voxel_features.device).T
        indices = coors[:, 2] * self.nx + coors[:, 3]
        indices = indices.long()
        voxels = voxel_features
        # Now scatter the blob back to the canvas.
        canvas[indices, :] = voxels
        canvas = canvas.T
        # Undo the column stacking to final 4-dim tensor
        canvas = canvas.view(1, self.in_channels, self.ny, self.nx)
        return canvas

    def forward_batch(self, voxel_features, coors, batch_size):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
        """
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.in_channels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny,
                                         self.nx)

        return batch_canvas


@MIDDLE_ENCODERS.register_module()
class PointPillarsScatterRCS(PointPillarsScatter):

    def __init__(self, in_channels, output_shape):
        super(PointPillarsScatterRCS, self).__init__(in_channels, output_shape)
        # self.compress = nn.Conv2d(in_channels*2, in_channels, 1)
        # self.rcs_att = nn.Conv2d(2+1, in_channels, 1)
        # self.rcs_att = nn.Conv2d(2+1, in_channels, 1)

    def forward(self, voxel_features, coors, batch_size=None):
        point_features, rcs = voxel_features
        features = super().forward(point_features, coors, batch_size)
        # rcs_features = super().forward(rcs[:, :2], coors, batch_size, 2)

        heatmap = point_features.new_zeros((batch_size, self.ny,self.nx))
        heatmap_feat = torch.zeros_like(features)

        r = rcs[:, 0]**2 + rcs[:, 1]**2
        true_rcs = rcs[:, -2] * r
        true_rcs = torch.nn.functional.relu(true_rcs)

        radius = true_rcs + 1

        for i in range(coors.shape[0]):
            batch, _, y, x = coors[i]
            draw_heatmap_gaussian(heatmap[batch], [x, y], int(radius[i].data.item()))
            heatmap_feat[batch] = draw_heatmap_gaussian_feat(heatmap_feat[batch], [x, y], int(radius[i].data.item()), point_features[i])
            # left, right, top, bottom = draw_heatmap_gaussian_feat(heatmap_feat[batch], [x, y], int(radius[i].data.item()))
            # heatmap_feat[batch][:, y - top:y + bottom, x - left:x + right] += point_features[i].view(-1, 1, 1).expand_as(heatmap_feat[batch][:, y - top:y + bottom, x - left:x + right])
            # heatmap_feat[batch][:, y - left:y + right, x - top:x + bottom] += point_features[i].view(-1, 1, 1).expand_as(heatmap_feat[batch][:, y - left:y + right, x - top:x + bottom])
            # print(int(radius[i].data.item()), x, y, left, right, top, bottom, )
        return heatmap_feat*heatmap.unsqueeze(dim=1)

        # rcs_att = self.rcs_att(heatmap.unsqueeze(dim=1))

        # features_att = self.compress(torch.cat([features, rcs_att*heatmap_feat], dim=1))


        # heatmap = heatmap.unsqueeze(dim=1)
        # features_att = heatmap_feat*heatmap
        # features = torch.nn.functional.interpolate(features, size=(128,128))
        # heatmap_feat = torch.nn.functional.interpolate(heatmap_feat, size=(128,128))
        # heatmap = torch.nn.functional.interpolate(heatmap, size=(128,128))
        # # print(features_att.shape)
        # features_att = torch.nn.functional.interpolate(features_att, size=(128,128))
        # self.show(heatmap_feat, 'heatmap_feat')
        # self.show(features, 'feat')
        # self.show(heatmap, 'heatmap')
        # self.show(features_att, 'att')
        # exit(0)

        # return features_att

    # # @auto_fp16(apply_to=('voxel_features', ))
    # def forward(self, voxel_features, coors, batch_size=None):
    #     point_features, rcs = voxel_features
    #     features = super().forward(point_features, coors, batch_size)
    #     rcs_features = super().forward(rcs[:, :2], coors, batch_size, 2)

    #     heatmap = point_features.new_zeros((batch_size, self.ny,self.nx))
    #     # radius = torch.tanh(rcs[:, -2]) * 10
    #     r = rcs[:, 0]**2 + rcs[:, 1]**2
    #     true_rcs = rcs[:, -2] * r
    #     true_rcs = torch.nn.functional.relu(true_rcs)
    #     # radius = torch.sigmoid(rcs[:, -2]) * 40 + 1
    #     radius = torch.sigmoid(true_rcs) * 40 + 1
    #     # print(radius.max(), radius.min(), radius.shape)
    #     # print(coors.shape)
    #     for i in range(coors.shape[0]):
    #         batch, _, y, x = coors[i]
    #         draw_heatmap_gaussian(heatmap[batch], [y, x], int(radius[i].data.item()))
    #     # features = features * heatmap.unsqueeze(dim=1)
    #     rcs_att = self.rcs_att(torch.cat([rcs_features, heatmap.unsqueeze(dim=1)], dim=1))
    #     # features_att = features + heatmap.unsqueeze(dim=1)
    #     features_att = self.compress(torch.cat([features, rcs_att], dim=1))

    #     # heatmap = heatmap.unsqueeze(dim=1)
    #     # features = torch.nn.functional.interpolate(features, size=(128,128))
    #     # heatmap = torch.nn.functional.interpolate(heatmap, size=(128,128))
    #     # features_att = torch.nn.functional.interpolate(features_att, size=(128,128))
    #     # self.show(features, 'feat')
    #     # self.show(heatmap, 'heatmap')
    #     # self.show(features_att, 'att')
    #     # exit(0)

    #     return features_att

    def show(self, pts_feats, path):
        import matplotlib.pyplot as plt
        import numpy as np
        fig = plt.figure(figsize=(8, 8))
        print('\n******** BEGIN PRINT **********\n')
        for idx in range(pts_feats.shape[0]):
            pts_feat = pts_feats[idx].cpu().detach().numpy() # 放到cpu上 去掉梯度 转换成numpy
            feat_2d = np.zeros(pts_feat.shape[1:])
            for h in range(feat_2d.shape[0]):
                for w in range(feat_2d.shape[1]):
                    for c in range(pts_feat.shape[0]):
                        feat_2d[h][w] += abs(pts_feat[c][h][w])
            feat_2d = feat_2d/feat_2d.max()
            # plt.imshow(feat_2d, cmap=plt.cm.gray, vmin=0, vmax=255)
            plt.imshow(feat_2d)
            plt.savefig(f"/data1/linzhiwei/project/bevperception/vis/{idx}_{path}.png")
        print('\n******** END PRINT **********\n')


@MIDDLE_ENCODERS.register_module()
class PointPillarsScatterRCSr2(PointPillarsScatter):

    def __init__(self, in_channels, output_shape):
        super(PointPillarsScatterRCSr2, self).__init__(in_channels, output_shape)
        self.compress = nn.Conv2d(in_channels*2, in_channels, 3, padding=1)
        self.rcs_att = nn.Conv2d(2, in_channels, 1)
        # self.rcs_att = nn.Conv2d(2+1, in_channels, 1)

    def forward(self, voxel_features, coors, batch_size=None):
        point_features, rcs = voxel_features
        features = super().forward(point_features, coors, batch_size)
        # rcs_features = super().forward(rcs[:, :2], coors, batch_size, 2)

        heatmap = point_features.new_zeros((batch_size, self.ny,self.nx))
        heatmap_feat = point_features.new_zeros((batch_size, 1, self.ny,self.nx))

        r = rcs[:, 0]**2 + rcs[:, 1]**2
        true_rcs = rcs[:, -2] * r
        true_rcs = torch.nn.functional.relu(true_rcs)

        radius = true_rcs + 1

        for i in range(coors.shape[0]):
            batch, _, y, x = coors[i]
            # draw_heatmap_gaussian(heatmap[batch], [y, x], int(radius[i].data.item()))
            draw_heatmap_gaussian(heatmap[batch], [x, y], int(radius[i].data.item()))
            heatmap_feat[batch] = draw_heatmap_gaussian_feat(heatmap_feat[batch], [x, y], int(radius[i].data.item()), rcs[i, -2])
            # draw_heatmap_gaussian_feat(heatmap_feat[batch], [y, x], int(radius[i].data.item()), point_features[i])

        # rcs_att = self.rcs_att(heatmap.unsqueeze(dim=1))
        rcs_att = self.rcs_att(torch.cat([heatmap.unsqueeze(dim=1), heatmap_feat],dim=1))

        features_att = self.compress(torch.cat([features, rcs_att], dim=1))
        return features_att

        # heatmap = heatmap.unsqueeze(dim=1)
        # features = torch.nn.functional.interpolate(features, size=(128,128))
        # heatmap = torch.nn.functional.interpolate(heatmap, size=(128,128))
        # features_att = torch.nn.functional.interpolate(features_att, size=(128,128))
        # self.show(features, 'feat')
        # self.show(heatmap, 'heatmap')
        # self.show(features_att, 'att')
        # exit(0)

        # return features_att

    # # @auto_fp16(apply_to=('voxel_features', ))
    # def forward(self, voxel_features, coors, batch_size=None):
    #     point_features, rcs = voxel_features
    #     features = super().forward(point_features, coors, batch_size)
    #     rcs_features = super().forward(rcs[:, :2], coors, batch_size, 2)

    #     heatmap = point_features.new_zeros((batch_size, self.ny,self.nx))
    #     # radius = torch.tanh(rcs[:, -2]) * 10
    #     r = rcs[:, 0]**2 + rcs[:, 1]**2
    #     true_rcs = rcs[:, -2] * r
    #     true_rcs = torch.nn.functional.relu(true_rcs)
    #     # radius = torch.sigmoid(rcs[:, -2]) * 40 + 1
    #     radius = torch.sigmoid(true_rcs) * 40 + 1
    #     # print(radius.max(), radius.min(), radius.shape)
    #     # print(coors.shape)
    #     for i in range(coors.shape[0]):
    #         batch, _, y, x = coors[i]
    #         draw_heatmap_gaussian(heatmap[batch], [y, x], int(radius[i].data.item()))
    #     # features = features * heatmap.unsqueeze(dim=1)
    #     rcs_att = self.rcs_att(torch.cat([rcs_features, heatmap.unsqueeze(dim=1)], dim=1))
    #     # features_att = features + heatmap.unsqueeze(dim=1)
    #     features_att = self.compress(torch.cat([features, rcs_att], dim=1))

    #     # heatmap = heatmap.unsqueeze(dim=1)
    #     # features = torch.nn.functional.interpolate(features, size=(128,128))
    #     # heatmap = torch.nn.functional.interpolate(heatmap, size=(128,128))
    #     # features_att = torch.nn.functional.interpolate(features_att, size=(128,128))
    #     # self.show(features, 'feat')
    #     # self.show(heatmap, 'heatmap')
    #     # self.show(features_att, 'att')
    #     # exit(0)

    #     return features_att

    def show(self, pts_feats, path):
        import matplotlib.pyplot as plt
        import numpy as np
        fig = plt.figure(figsize=(8, 8))
        print('\n******** BEGIN PRINT **********\n')
        for idx in range(pts_feats.shape[0]):
            pts_feat = pts_feats[idx].cpu().detach().numpy() # 放到cpu上 去掉梯度 转换成numpy
            feat_2d = np.zeros(pts_feat.shape[1:])
            for h in range(feat_2d.shape[0]):
                for w in range(feat_2d.shape[1]):
                    for c in range(pts_feat.shape[0]):
                        feat_2d[h][w] += abs(pts_feat[c][h][w])
            feat_2d = feat_2d/feat_2d.max()
            # plt.imshow(feat_2d, cmap=plt.cm.gray, vmin=0, vmax=255)
            plt.imshow(feat_2d)
            plt.savefig(f"/data1/linzhiwei/project/bevperception/vis/{idx}_{path}.png")
        print('\n******** END PRINT **********\n')