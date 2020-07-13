import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class CenterConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', marker='b'):
        super(CenterConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                           stride, padding, dilation, groups, bias, padding_mode)
        self.marker = marker

    def forward(self, input):
        shape = self.weight.size()
        weight = self.weight.view([shape[0], -1])
        weight = weight - weight.mean(dim=1, keepdim=True)
        weight = weight.view(shape)

        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class WnConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', marker='b'):
        super(WnConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups, bias, padding_mode)
        self.marker = marker

    def forward(self, input):
        shape = self.weight.size()
        weight = self.weight.view([shape[0], -1])

        mean = weight.mean(dim=1, keepdim=True)
        std = weight.std(dim=1, keepdim=True)

        weight = (weight - mean) * (np.sqrt(2. / weight.size(1)) / std)
        weight = weight.view(shape)

        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class L2Norm2d(nn.BatchNorm2d):
    def forward(self, x):
        self._check_input_dim(x)
        # NCHW -> CNHW
        y = x.transpose(0, 1)

        # remember the shape
        return_shape = y.shape

        # CNHW -> C, N*H*W
        y = y.contiguous().view(x.size(1), -1)

        z = y ** 2
        sm = torch.sqrt(z.mean(dim=1))

        if self.training is not True:
            y = y * (self.weight.view(-1, 1) / (self.running_var.view(-1, 1) + self.eps))
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * sm
            y = y * (self.weight.view(-1, 1) / (sm.view(-1, 1) + self.eps))

        y += self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0, 1)


class L1Norm2d(nn.BatchNorm2d):
    def forward(self, x):
        self._check_input_dim(x)
        # NCHW -> CNHW
        y = x.transpose(0, 1)

        # remember the shape
        return_shape = y.shape

        # CNHW -> C, N*H*W
        y = y.contiguous().view(x.size(1), -1)

        z = y.abs()
        sm = z.mean(dim=1)

        if self.training is not True:
            y = y * (self.weight.view(-1, 1) / (self.running_var.view(-1, 1) + self.eps))
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * sm
            y = y * (self.weight.view(-1, 1) / (sm.view(-1, 1) + self.eps))

        y += self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0, 1)


class MeanOnlyBN(nn.BatchNorm2d):
    def forward(self, x, scale=None, nonlinearity=None):
        self._check_input_dim(x)
        # NCHW -> CNHW
        y = x.transpose(0, 1)

        # remember the shape
        return_shape = y.shape

        # CNHW -> C, N*H*W
        y = y.contiguous().view(x.size(1), -1)

        mean = y.mean(dim=1)

        if self.training is not True:
            y = y - self.running_mean.view(-1, 1)
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            y = y - mean.view(-1, 1)

        y = self.bias.view(-1, 1) + y
        return y.view(return_shape).transpose(0, 1)


class GhostBN(nn.Module):
    def __init__(self, plane):
        super(GhostBN, self).__init__()
        self.bn = nn.BatchNorm2d(plane * 32)
    def forward(self, x):
        N, C, H, W = x.size()
        x_ = x.view((int(N / 32), C * 32, H, W))
        return self.bn(x_).view(x.size())
