"""
Modified from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ops import MeanOnlyBN, Conv2d_


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm=None, acf=None, conv=None, wn=False, bns=False):
        """
        The basic Residual Block
        :param in_planes: the number of input channels
        :param planes: the number of output channels
        :param stride: the convolving stride
        :param norm: the normalization function
        :param acf: the activation function
        :param conv: the convolution function
        :param wn: bool, whether apply weight normalization
        :param bns: bool, wether add scalar multiplier and bias
        """
        super(BasicBlock, self).__init__()

        # configurations
        self.acf = acf
        self.bns = bns
        self.conv_bias = (norm is None) and (not bns)  # if there is no normalization, and bns is not applied

        if conv is None:
            conv = Conv2d_  # Conv2d_ has an additional marker that can be recognized by the initializer

        # initialize the two convolutions within the block
        self.conv1 = conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=self.conv_bias, marker='b')
        self.conv2 = conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=self.conv_bias, marker='be')

        # apply native weight normalization
        if wn:
            self.conv1 = nn.utils.weight_norm(self.conv1, name='weight')
            self.conv2 = nn.utils.weight_norm(self.conv2, name='weight')

        # configure the downsampling strategy
        self.downsample = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.AvgPool2d(1, stride=stride)

        if self.bns:
            self.bias0 = nn.Parameter(torch.zeros(1))
            self.bias1 = nn.Parameter(torch.zeros(1))
            self.bias2 = nn.Parameter(torch.zeros(1))
            self.scale = nn.Parameter(torch.ones(1))
            self.bias3 = nn.Parameter(torch.zeros(1))
            self.block = nn.Sequential()
        else:
            if norm is not None:
                self.norm1 = norm(planes)
                self.norm2 = norm(planes)
                self.block = nn.Sequential(self.conv1, self.norm1, acf, self.conv2, self.norm2)
            else:
                self.block = nn.Sequential(self.conv1, acf, self.conv2)

    def forward(self, x):
        identity = x
        if self.bns:
            out = self.conv1(x + self.bias0)
            out = self.acf(out + self.bias1)
            out = self.conv2(out + self.bias2)
            out = out * self.scale + self.bias3
        else:
            out = self.block(x)
        if self.downsample is not None:
            if self.bns:
                identity = self.downsample(self.bias0 + x)
            else:
                identity = self.downsample(x)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)
        out += identity
        out = self.acf(out)
        return out


blocks = {
    '20': (BasicBlock, [3, 3, 3]),
    '32': (BasicBlock, [5, 5, 5]),
    '44': (BasicBlock, [7, 7, 7]),
    '56': (BasicBlock, [9, 9, 9]),
}


class ResNet(nn.Module):
    def __init__(self, depth, num_classes=10, init_fn=None, norm_fn=None, activ_fn=None, conv=None):
        super(ResNet, self).__init__()
        self.in_planes = 16
        block, num_blocks = blocks[depth]

        # configuations
        if conv is None:
            self.conv = Conv2d_
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
        # whether add bias and scales
        self.bns = init_fn.bns

        self.conv_bias = (norm_fn is None) and (not self.bns)  # if there is no normalization, and bns is not applied

        self.conv1 = self.conv(3, 16, kernel_size=3, stride=1, padding=1, bias=self.conv_bias, marker='i')
        # for weight normalization
        if self.wn:
            self.conv1 = nn.utils.weight_norm(self.conv1, name='weight')
        # for adding bias and scales
        if self.bns:
            self.bias0 = nn.Parameter(torch.zeros(1))
            self.block0 = nn.Sequential()
        else:
            if self.norm is None:
                self.block0 = nn.Sequential(self.conv1, self.acf)
            else:
                self.norm1 = self.norm(16)
                self.block0 = nn.Sequential(self.conv1, self.norm1, self.acf)

        # build three major blocks
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.bias2 = nn.Parameter(torch.zeros(1))

        # last classifier dense layer
        self.linear = nn.Linear(64 * block.expansion, num_classes)

        # apply the initialization
        if init_fn is not None:
            self.apply(init_fn.initialization)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm, self.acf, self.conv, self.wn, self.bns))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bns:
            out = self.conv1(x)
            out = self.acf(out + self.bias0)
        else:
            out = self.block0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)  # F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        if self.bns:
            out = out + self.bias2
        out = self.linear(out)
        return out
