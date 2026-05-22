from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

import mmcv

from mmdet3d.models.builder import HEADS

__all__ = ["BEVSegmentationHead", "BEVSegmentationHeadv2", "BEVSegmentationMultiHead"]


def sigmoid_xent_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


class BEVGridTransform(nn.Module):
    def __init__(
        self,
        *,
        input_scope: List[Tuple[float, float, float]],
        output_scope: List[Tuple[float, float, float]],
        prescale_factor: float = 1,
    ) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prescale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.prescale_factor,
                mode="bilinear",
                align_corners=False,
            )

        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
            self.input_scope, self.output_scope
        ):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))

        u, v = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)

        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return x


@HEADS.register_module()
class BEVSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        grid_transform: Dict[str, Any],
        classes: List[str],
        loss: str,
        loss_weight=1,
        type=0,
        mid_channels=0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.loss = loss
        self.loss_weight = loss_weight

        self.transform = BEVGridTransform(**grid_transform)
        if type == 0:
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, len(classes), 1),
            )
        elif type == 1: # 更换Norm
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.InstanceNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.InstanceNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, len(classes), kernel_size=1, padding=0),
            )
        elif type == 2: # 更换channels
            if mid_channels == 0:
                mid_channels = in_channels
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.InstanceNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(True),
                nn.Conv2d(mid_channels, len(classes), kernel_size=1, padding=0),
            )
        elif type == 3: # 5层
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, len(classes), 1),
            )
        elif type == 4:
            if mid_channels == 0:
                mid_channels = in_channels
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(True),
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(True),
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(True),
                nn.Conv2d(mid_channels, len(classes), 1),
            )
        else:
            assert False

    def forward(
            self,
            x: torch.Tensor,
            target: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:

        if isinstance(x, (list, tuple)):
            x = x[0]

        if isinstance(target, (list, tuple)):
            target = torch.stack(target, dim=0)

        x = self.transform(x)
        x = self.classifier(x)

        # DEBUG for visibility
        # target = target[0]
        # if not self.training:
        #     for i in range(x.shape[0]):
        #         self.vis(x[i], target[i], 1)

        if self.training:
            losses = {}
            for index, name in enumerate(self.classes):
                if self.loss == "xent":
                    loss = sigmoid_xent_loss(x[:, index], target[:, index]) * self.loss_weight
                elif self.loss == "focal":
                    loss = sigmoid_focal_loss(x[:, index], target[:, index]) * self.loss_weight
                else:
                    raise ValueError(f"unsupported loss: {self.loss}")
                losses[f"{name}_{self.loss}_loss"] = loss
            return losses
        else:
            return torch.sigmoid(x)

    def vis(self, x, t, type):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import random
        from datetime import datetime

        randstr = datetime.now().strftime('%H:%M:%S')

        print('\n******** BEGIN PRINT GT**********\n')
        print("BEV:", t.shape)
        fig = plt.figure(figsize=(16, 16))
        plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
        plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
        for xx in range(200):
            for yy in range(200):
                xc = -50 + xx * 0.5
                yc = -50 + yy * 0.5
                # 0 vehicle, 1 可行驶区域, 2 车道线
                if type == 0:
                    if t[0, xx, yy] == 1:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="blue"))
                elif type == 1:
                    if t[0, xx, yy] == 1:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="green"))
                elif type == 2:
                    if t[0, xx, yy] == 1:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="red"))
                else:
                    raise ValueError
        if type == 0:
            plt.savefig("/home/wangxinhao/vis/seg_v/gt/gt" + randstr + ".png")
        elif type == 1:
            plt.savefig("/home/wangxinhao/vis/seg_a/gt/gt" + randstr + ".png")
        elif type == 2:
            plt.savefig("/home/wangxinhao/vis/seg_d/gt/gt" + randstr + ".png")
        else:
            raise TypeError
        print('\n******** END PRINT GT**********\n')

        print('\n******** BEGIN PRINT PRED**********\n')
        print("BEV:", x.shape)
        fig = plt.figure(figsize=(16, 16))
        plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
        plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
        for xx in range(200):
            for yy in range(200):
                xc = -50 + xx * 0.5
                yc = -50 + yy * 0.5
                # 0 vehicle, 1 可行驶区域, 2 车道线
                if type == 0:
                    if x[0, xx, yy] > 0.45:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="blue"))
                elif type == 1:
                    if x[0, xx, yy] > 0.45:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="green"))
                elif type == 2:
                    if x[0, xx, yy] > 0.40:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="red"))
                else:
                    raise TypeError
        if type == 0:
            plt.savefig("/home/wangxinhao/vis/seg_v/pred/pred" + randstr + ".png")
        elif type == 1:
            plt.savefig("/home/wangxinhao/vis/seg_a/pred/pred" + randstr + ".png")
        elif type == 2:
            plt.savefig("/home/wangxinhao/vis/seg_d/pred/pred" + randstr + ".png")
        else:
            raise TypeError

        print('\n******** END PRINT PRED**********\n')


@HEADS.register_module()
class BEVSegmentationHeadv2(nn.Module):
    def __init__(
            self,
            in_channels: int,
            classes: List[str],
            loss: str,
            loss_weight=1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.loss = loss
        self.loss_weight = loss_weight

        assert type(loss_weight) == int or type(loss_weight) == float \
               or type(loss_weight) == mmcv.utils.config.ConfigDict

        # self.transform = BEVGridTransform(**grid_transform)
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, len(classes), 1),
        )

    def forward(
            self,
            x: torch.Tensor,
            target: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:

        if isinstance(x, (list, tuple)):
            x = x[0]

        if isinstance(target, (list, tuple)):
            target = torch.stack(target, dim=0)

        # x = self.transform(x)
        # print("before", x.shape)
        x = self.classifier(x)
        # print("after", x.shape)

        # DEBUG for visibility
        # target = target[0]
        # if not self.training:
        #     for i in range(x.shape[0]):
        #         self.vis(x[i], target[i], 1)

        if self.training:
            losses = {}
            for index, name in enumerate(self.classes):
                if self.loss == "xent":
                    if type(self.loss_weight) == mmcv.utils.config.ConfigDict:
                        loss = sigmoid_xent_loss(x[:, index], target[:, index]) * self.loss_weight[name]
                    else:
                        loss = sigmoid_xent_loss(x[:, index], target[:, index]) * self.loss_weight
                elif self.loss == "focal":
                    if type(self.loss_weight) == mmcv.utils.config.ConfigDict:
                        loss = sigmoid_focal_loss(x[:, index], target[:, index]) * self.loss_weight[name]
                    else:
                        loss = sigmoid_focal_loss(x[:, index], target[:, index]) * self.loss_weight
                else:
                    raise ValueError(f"unsupported loss: {self.loss}")
                losses[f"{name}_{self.loss}_loss"] = loss
            return losses
        else:
            return torch.sigmoid(x)

    def vis(self, x, t, type):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import random
        from datetime import datetime

        randstr = datetime.now().strftime('%H:%M:%S')

        print('\n******** BEGIN PRINT GT**********\n')
        print("BEV:", t.shape)
        fig = plt.figure(figsize=(16, 16))
        plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
        plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
        for xx in range(200):
            for yy in range(200):
                xc = -50 + xx * 0.5
                yc = -50 + yy * 0.5
                # 0 vehicle, 1 可行驶区域, 2 车道线
                if type == 0:
                    if t[0, xx, yy] == 1:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="blue"))
                elif type == 1:
                    if t[0, xx, yy] == 1:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="green"))
                elif type == 2:
                    if t[0, xx, yy] == 1:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="red"))
                else:
                    raise ValueError
        if type == 0:
            plt.savefig("/home/wangxinhao/vis/seg_v/gt/gt" + randstr + ".png")
        elif type == 1:
            plt.savefig("/home/wangxinhao/vis/seg_a/gt/gt" + randstr + ".png")
        elif type == 2:
            plt.savefig("/home/wangxinhao/vis/seg_d/gt/gt" + randstr + ".png")
        else:
            raise TypeError
        print('\n******** END PRINT GT**********\n')

        print('\n******** BEGIN PRINT PRED**********\n')
        print("BEV:", x.shape)
        fig = plt.figure(figsize=(16, 16))
        plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
        plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
        for xx in range(200):
            for yy in range(200):
                xc = -50 + xx * 0.5
                yc = -50 + yy * 0.5
                # 0 vehicle, 1 可行驶区域, 2 车道线
                if type == 0:
                    if x[0, xx, yy] > 0.45:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="blue"))
                elif type == 1:
                    if x[0, xx, yy] > 0.45:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="green"))
                elif type == 2:
                    if x[0, xx, yy] > 0.40:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="red"))
                else:
                    raise TypeError
        if type == 0:
            plt.savefig("/home/wangxinhao/vis/seg_v/pred/pred" + randstr + ".png")
        elif type == 1:
            plt.savefig("/home/wangxinhao/vis/seg_a/pred/pred" + randstr + ".png")
        elif type == 2:
            plt.savefig("/home/wangxinhao/vis/seg_d/pred/pred" + randstr + ".png")
        else:
            raise TypeError

        print('\n******** END PRINT PRED**********\n')


@HEADS.register_module()
class BEVSegmentationMultiHead(nn.Module):
    def __init__(
            self,
            in_channels: int,
            classes: List[str],
            loss: str,
            loss_weight=1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.loss = loss
        self.loss_weight = loss_weight

        assert type(loss_weight) == int or type(loss_weight) == float \
               or type(loss_weight) == mmcv.utils.config.ConfigDict

        # self.transform = BEVGridTransform(**grid_transform)
        self.classifiers = nn.ModuleList()
        for _ in classes:
            self.classifiers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    nn.Conv2d(256, 256, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    nn.Conv2d(256, 128, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.Conv2d(128, 64, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.Conv2d(64, 1, 1),
                )
            )

    def forward(
            self,
            x: torch.Tensor,
            target: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:

        if isinstance(x, (list, tuple)):
            x = x[0]

        if isinstance(target, (list, tuple)):
            target = torch.stack(target, dim=0)

        # x = self.transform(x)
        # print("before", x.shape)
        out_list = []
        for classifier in self.classifiers:
            out = classifier(x)
            out_list.append(out)
        x = torch.cat(out_list, dim=1)
        # print("after", x.shape)

        # DEBUG for visibility
        # target = target[0]
        # if not self.training:
        #     for i in range(x.shape[0]):
        #         self.vis(x[i], target[i], 1)

        if self.training:
            losses = {}
            for index, name in enumerate(self.classes):
                if self.loss == "xent":
                    if type(self.loss_weight) == mmcv.utils.config.ConfigDict:
                        loss = sigmoid_xent_loss(x[:, index], target[:, index]) * self.loss_weight[name]
                    else:
                        loss = sigmoid_xent_loss(x[:, index], target[:, index]) * self.loss_weight
                elif self.loss == "focal":
                    if type(self.loss_weight) == mmcv.utils.config.ConfigDict:
                        loss = sigmoid_focal_loss(x[:, index], target[:, index]) * self.loss_weight[name]
                    else:
                        loss = sigmoid_focal_loss(x[:, index], target[:, index]) * self.loss_weight
                else:
                    raise ValueError(f"unsupported loss: {self.loss}")
                losses[f"{name}_{self.loss}_loss"] = loss
            return losses
        else:
            return torch.sigmoid(x)

    def vis(self, x, t, type):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import random
        from datetime import datetime

        randstr = datetime.now().strftime('%H:%M:%S')

        print('\n******** BEGIN PRINT GT**********\n')
        print("BEV:", t.shape)
        fig = plt.figure(figsize=(16, 16))
        plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
        plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
        for xx in range(200):
            for yy in range(200):
                xc = -50 + xx * 0.5
                yc = -50 + yy * 0.5
                # 0 vehicle, 1 可行驶区域, 2 车道线
                if type == 0:
                    if t[0, xx, yy] == 1:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="blue"))
                elif type == 1:
                    if t[0, xx, yy] == 1:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="green"))
                elif type == 2:
                    if t[0, xx, yy] == 1:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="red"))
                else:
                    raise ValueError
        if type == 0:
            plt.savefig("/home/wangxinhao/vis/seg_v/gt/gt" + randstr + ".png")
        elif type == 1:
            plt.savefig("/home/wangxinhao/vis/seg_a/gt/gt" + randstr + ".png")
        elif type == 2:
            plt.savefig("/home/wangxinhao/vis/seg_d/gt/gt" + randstr + ".png")
        else:
            raise TypeError
        print('\n******** END PRINT GT**********\n')

        print('\n******** BEGIN PRINT PRED**********\n')
        print("BEV:", x.shape)
        fig = plt.figure(figsize=(16, 16))
        plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
        plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
        for xx in range(200):
            for yy in range(200):
                xc = -50 + xx * 0.5
                yc = -50 + yy * 0.5
                # 0 vehicle, 1 可行驶区域, 2 车道线
                if type == 0:
                    if x[0, xx, yy] > 0.45:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="blue"))
                elif type == 1:
                    if x[0, xx, yy] > 0.45:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="green"))
                elif type == 2:
                    if x[0, xx, yy] > 0.40:
                        plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="red"))
                else:
                    raise TypeError
        if type == 0:
            plt.savefig("/home/wangxinhao/vis/seg_v/pred/pred" + randstr + ".png")
        elif type == 1:
            plt.savefig("/home/wangxinhao/vis/seg_a/pred/pred" + randstr + ".png")
        elif type == 2:
            plt.savefig("/home/wangxinhao/vis/seg_d/pred/pred" + randstr + ".png")
        else:
            raise TypeError

        print('\n******** END PRINT PRED**********\n')
