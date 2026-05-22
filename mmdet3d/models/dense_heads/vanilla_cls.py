from torch import nn
from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class BEVClsHead(nn.Module):
    def __init__(self, inc, classes):
        super().__init__()
        self.classes = classes
        self.avgpool = nn.AvgPool2d(64)
        self.fc = nn.Linear(inc, classes)
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x):

        if isinstance(x, (list, tuple)):
            x = x[0]

        B = x.shape[0]
        x = self.avgpool(x)
        x = x.view(B, -1)
        x = self.fc(x)

        return x

    def loss(self, x, target):
        loss = self.lossfn(x, target)
        return loss
