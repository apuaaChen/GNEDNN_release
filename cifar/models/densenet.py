"""
Modified From https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops import MeanOnlyBN


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, norm=None, acf=None, conv=None, wn=False):
        super(Bottleneck, self).__init__()
        if conv is None:
            conv = nn.Conv2d

        self.conv1 = conv(in_planes, 4 * growth_rate, kernel_size=1, bias=norm is None)
        self.conv2 = conv(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=norm is None)

        if wn:
            self.conv1 = nn.utils.weight_norm(self.conv1, name='weight')
            self.conv2 = nn.utils.weight_norm(self.conv2, name='weight')

        if norm is not None:
            self.norm1 = norm(in_planes)
            self.norm2 = norm(4 * growth_rate)
            layers = [self.norm1, acf, self.conv1, self.norm2, acf, self.conv2]
        else:
            layers = [acf, self.conv1, acf, self.conv2]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, norm=None, acf=None, conv=None, wn=False):
        super(Transition, self).__init__()
        if conv is None:
            conv = nn.Conv2d

        self.conv = conv(in_planes, out_planes, kernel_size=1, bias=norm is None, stride=2, padding=1)
        if wn:
            self.conv = nn.utils.weight_norm(self.conv, name='weight')
        if norm is not None:
            self.norm = norm(in_planes)
            layers = [self.norm, acf, self.conv]
        else:
            layers = [acf, self.conv]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out


class DenseNet(nn.Module):
    def __init__(self,reduction=0.5, num_classes=10,
                 init_fn=None, norm_fn=None, activ_fn=None, conv=None):
        super(DenseNet, self).__init__()

        block = Bottleneck
        nblocks = [8, 8, 8]
        growth_rate = 12
        self.growth_rate = growth_rate

        # configure the methods to apply
        if conv is None:
            self.conv = nn.Conv2d
        else:
            self.conv = conv
        if activ_fn is None:
            self.acf = F.relu
        else:
            self.acf = activ_fn

        if norm_fn == 'wn':
            self.norm = MeanOnlyBN
            self.wn = True
        else:
            self.norm = norm_fn
            self.wn = False

        # init convolution
        num_planes = 2 * growth_rate
        self.conv1 = self.conv(3, num_planes, kernel_size=3, padding=1, bias=self.norm is None)
        if self.wn:
            self.conv1 = nn.utils.weight_norm(self.conv1, name='weight')

        # Dense block 1 & trans block 1
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes, self.norm, self.acf, self.conv, self.wn)
        num_planes = out_planes

        # Dense block 2 & trans block 2
        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes, self.norm, self.acf, self.conv, self.wn)
        num_planes = out_planes

        # Dense block 3
        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate

        if self.norm is not None:
            self.norm_last = self.norm(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

        if init_fn is not None:
            self.apply(init_fn)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, self.norm, self.acf, self.conv, self.wn))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        if self.norm is None:
            out = F.avg_pool2d(self.acf(out), 8)
        else:
            out = F.avg_pool2d(self.acf(self.norm_last(out)), 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out