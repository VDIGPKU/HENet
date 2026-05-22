import torch.nn as nn
import torchvision
import torch
from mmdet.models import BACKBONES


@BACKBONES.register_module()
class ResNet_tri(nn.Module):
    def __init__(self, dim_type, dim_min, dim_max):
        super().__init__()
        self.encoder = torchvision.models.resnet50()
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
        )
        self.dim_type = dim_type
        self.dim_min = dim_min
        self.dim_max = dim_max

    def forward(self, X):

        with torch.no_grad():
            feats = self.encoder(X)
            feats = self.projector(feats)
            dim_min = self.dim_min
            dim_max = self.dim_max

            if self.dim_type == 'random':
                feats = feats[:, dim_min:dim_max]
            elif self.dim_type == 'tri':
                softplus = torch.nn.Softplus()
                S = softplus(self.s)
                S = S.to("cuda")
                S_2, index = torch.sort(S, descending=True)
                feats = feats[:, index[dim_min:dim_max]]
            elif self.dim_type == 'ncl':
                feats_copy = torch.nn.functional.relu(feats)
                feats_sum = torch.sum(feats_copy, dim=0)
                S, index = torch.sort(feats_sum, descending=True)
                for i in range(512):
                    print(i, S[i])
                feats = feats[:, index[dim_min:dim_max]]

        return feats


# def main():
#     train_loader, val_loader = prepare_data(
#         'imagenet',
#         train_data_path="/data1/mjli/imagenet/IMAGENET/train",
#         val_data_path="/data1/mjli/imagenet/IMAGENET/val",
#         data_format="image_folder",
#         batch_size=256,
#         num_workers=4,
#     )
#
#     model = IResNet("tri", 0, 20)
#     model = model.to("cuda")
#
#     ckpt_path = "trained_models/simclr/15zjg2cv/simclr-imagenet-tri-15zjg2cv-ep=94.ckpt"
#     # ckpt_path = "trained_models/simclr/2tvkrj9d/simclr-imagenet-base-2tvkrj9d-ep=98.ckpt"
#     # ckpt_path = "trained_models/simclr/1m867sst/simclr-imagenet-ncl-1m867sst-ep=98.ckpt"
#
#     state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
#     scale_param = 0
#     state2 = state.copy()
#     for k in list(state.keys()):
#         if "encoder" in k:
#             state[k.replace("encoder", "backbone")] = state[k]
#             logging.warn(
#                 "You are using an older checkpoint. Use a new one as some issues might arrise."
#             )
#         if "backbone" in k:
#             state[k.replace("backbone.", "")] = state[k]
#         if "scale" in k:
#             scale_param = state[k]
#         if "projector" in k:
#             state2[k.replace("projector.", "")] = state2[k]
#
#         del state2[k]
#         del state[k]
#     model.encoder.load_state_dict(state, strict=False)
#     model.projector.load_state_dict(state2)
#     model.s = scale_param
#
#     for data in val_loader:
#         X, _ = data
#         X = X.to("cuda")
#         z = model(X)
#         print(z)
#         exit()
#
#
# if __name__ == "__main__":
#     main()


