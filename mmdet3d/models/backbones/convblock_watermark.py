
import random

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from tools.watermark_cache import GlobalConfig


class ConvBlock(nn.Module):
    def __init__(self, i, o, ks=3, s=1, pd=1, bn='bn', relu=True, passport_kwargs=None):
        super().__init__()

        self.conv = nn.Conv2d(i, o, ks, s, pd, bias=False)
        self.device = self.conv.weight.device
        self.sign_loss_private = None  # passport layer
        self.private_bn, self.projection = None, None  # passport norm
        self.cached_result = None
        if bn == 'bn':
            self.bn = nn.BatchNorm2d(o, affine=False)
        elif bn == 'gn':
            group_map = {
                100: 5,
                257: 1
            }
            num_groups = max(o // 16, 1)
            if o in group_map.keys():
                num_groups = group_map[o]
            self.bn = nn.GroupNorm(num_groups, o, affine=False)
        elif bn == 'in':
            self.bn = nn.InstanceNorm2d(o, affine=False)
        else:
            raise Exception('unknown norm type')

        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None
        if isinstance(passport_kwargs, dict) and passport_kwargs.get('flag', False):
            if GlobalConfig.nvo in ['tdn']:
                if bn == 'bn':
                    self.private_bn = nn.BatchNorm2d(o, affine=False)
                if GlobalConfig.nvo == 'tdn':
                    self.projection = nn.Sequential(
                        nn.Linear(o, o // 2, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Linear(o // 2, o, bias=True)
                    )

            b = passport_kwargs.get('b',
                                    torch.sign(torch.rand(o) - 0.5))  # bit information to store, TODO: hash by default
            if GlobalConfig.nvo == 'tdn':
                b = GlobalConfig.signature
            if isinstance(b, str):
                b = b[:o // 8]
                # print('setting signature:', b)
                if len(b) * 8 > o:
                    raise Exception('invalid bit length')
                bsign = torch.sign(torch.rand(o) - 0.5)
                bitstring = ''.join([format(ord(c), 'b').zfill(8) for c in b])

                for i, c in enumerate(bitstring):
                    bsign[i] = -1 if c == '0' else 1
                b = bsign
            else:
                raise Exception('invalid b type')
            self.register_buffer('b', b)

            self.sign_loss_private = SignLoss(passport_kwargs.get('sign_loss', .1), self.b)

            self.register_buffer('private_gamma_fm', None)
            self.register_buffer('private_beta_fm', None)

        self.public_gamma = nn.Parameter(torch.Tensor(1, o, 1, 1).to(self.device))
        self.public_beta = nn.Parameter(torch.Tensor(1, o, 1, 1).to(self.device))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        init.ones_(self.public_gamma)
        init.zeros_(self.public_beta)
        if self.projection is not None:
            for m in self.projection.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        init.constant_(m.bias, 0)

    def _passport_selection(self, passport_candidates: torch.Tensor):
        b, c, h, w = passport_candidates.size()

        if c == 3:  # input channel
            randb = random.randint(0, b - 1)
            return passport_candidates[randb].unsqueeze(0)

        passport_candidates = passport_candidates.view(b * c, h, w)
        full = False
        flag = [False for _ in range(b * c)]
        channel = c
        passportcount = 0
        bcount = 0
        passport = []

        while not full:
            if bcount >= b:
                bcount = 0

            randc = bcount * channel + random.randint(0, channel - 1)
            while flag[randc]:
                randc = bcount * channel + random.randint(0, channel - 1)
            flag[randc] = True

            passport.append(passport_candidates[randc].unsqueeze(0).unsqueeze(0))

            passportcount += 1
            bcount += 1

            if passportcount >= channel:
                full = True

        passport = torch.cat(passport, dim=1)
        return passport

    def set_private_keys(self, private_beta_fm, private_gamma_fm):
        # with torch.no_grad():
        if self.sign_loss_private is None:
            return
        if int(private_beta_fm.size(0)) != 1:
            private_beta_fm = self._passport_selection(private_beta_fm)
            private_gamma_fm = self._passport_selection(private_gamma_fm)
        # assert private_beta_fm.size(0) == 1, 'only batch size of 1 for key'
        self.register_buffer('private_gamma_fm', private_gamma_fm)
        self.register_buffer('private_beta_fm', private_beta_fm)

    def get_private_param(self, is_gamma, sign_only=False):
        feature_map = self.private_gamma_fm if is_gamma else self.private_beta_fm
        param = self.conv(feature_map)
        b, c = param.size(0), param.size(1)
        if GlobalConfig.nvo == 'tdn':
            param_max, _ = torch.max(param.view(b, c, -1), dim=2)
            param_max, _ = torch.max(param_max, dim=0)
            param_max = param_max.view(1, c, 1, 1)
        param = param.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        param = param.mean(dim=0).view(1, c, 1, 1)
        if sign_only:
            return param.view(-1).sign()
        if is_gamma and self.sign_loss_private is not None:
            self.sign_loss_private.reset()
            self.sign_loss_private.add(param)
        if GlobalConfig.nvo == 'tdn':
            param = param_max
        # passport norm uses non-se param for sign loss, is it incorrect?
        if self.projection is not None:
            param = param.view(1, c)
            param = self.projection(param).view(1, c, 1, 1)

        return param

    def forward(self, x, ind=0):
        if ind == 1 and GlobalConfig.enable_cache and self.sign_loss_private is None:
            return self.cached_result
        x = self.conv(x)
        if self.private_bn is not None:
            if ind == 0:
                x = self.bn(x)
            else:
                x = self.private_bn(x)
        else:
            x = self.bn(x)
        if ind == 0 or self.sign_loss_private is None:
            x = self.public_gamma * x + self.public_beta
        else:
            x = self.get_private_param(is_gamma=True) * x + self.get_private_param(is_gamma=False)
        if self.relu is not None:
            x = self.relu(x)
        if ind == 0 and GlobalConfig.enable_cache:
            self.cached_result = x
        return x




class SignLoss(nn.Module):
    def __init__(self, alpha, b=None):
        super(SignLoss, self).__init__()
        self.alpha = alpha
        self.register_buffer('b', b)
        self.loss = 0
        self.acc = 0
        self.scale_cache = None

    def set_b(self, b):
        self.b.copy_(b)

    def get_acc(self):
        if self.scale_cache is not None:
            acc = (torch.sign(self.b.view(-1)) == torch.sign(self.scale_cache.view(-1))).float().mean()
            return acc
        else:
            raise Exception('scale_cache is None')

    def get_loss(self):
        if self.scale_cache is not None:
            loss = (self.alpha * F.relu(-self.b.view(-1) * self.scale_cache.view(-1) + 0.1)).sum()
            return loss
        else:
            raise Exception('scale_cache is None')

    def add(self, scale):
        self.scale_cache = scale

        # hinge loss concept
        # f(x) = max(x + 0.5, 0)*-b
        # f(x) = max(x + 0.5, 0) if b = -1
        # f(x) = max(0.5 - x, 0) if b = 1

        # case b = -1
        # - (-1) * 1 = 1 === bad
        # - (-1) * -1 = -1 -> 0 === good

        # - (-1) * 0.6 + 0.5 = 1.1 === bad
        # - (-1) * -0.6 + 0.5 = -0.1 -> 0 === good

        # case b = 1
        # - (1) * -1 = 1 -> 1 === bad
        # - (1) * 1 = -1 -> 0 === good

        # let it has minimum of 0.1
        self.loss += self.get_loss()
        self.loss += (0.00001 * scale.view(-1).pow(2).sum())  # to regularize the scale not to be so large
        self.acc += self.get_acc()

    def reset(self):
        self.loss = 0
        self.acc = 0
        self.scale_cache = None

    # def to(self, *args, **kwargs):
    #     self.loss = self.loss.to(args[0])
    #     return super().to(*args, **kwargs)
