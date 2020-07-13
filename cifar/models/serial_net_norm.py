import torch.nn as nn
import torch.nn.functional as F

from ops.normalization import MeanOnlyBN


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c, out_c, stride=1, norm=None, acf=None, conv=None, wn=False):
        """
        building a basic block
        :param in_c: number of input channels
        :param out_c: number of output channels
        :param stride: length of CONV stride
        :param norm: function for normalization
        :param acf: activation function
        """
        if conv is None:
            conv = nn.Conv2d
        super(BasicBlock, self).__init__()
        self.conv1 = conv(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=norm is None)
        self.conv2 = conv(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=norm is None)

        if wn:
            self.conv1 = nn.utils.weight_norm(self.conv1, name='weight')
            self.conv2 = nn.utils.weight_norm(self.conv2, name='weight')

        if norm is not None:
            self.norm1 = norm(out_c)
            self.norm2 = norm(out_c)
            self.block = nn.Sequential(self.conv1, self.norm1, acf, self.conv2, self.norm2, acf)
        else:
            self.block = nn.Sequential(self.conv1, acf, self.conv2, acf)

    def forward(self, x):
        return self.block(x)


blocks = {
    '8': (BasicBlock, [1, 1, 1]),
    '14': (BasicBlock, [2, 2, 2]),
    '20': (BasicBlock, [3, 3, 3]),
    '32': (BasicBlock, [5, 5, 5]),
    '44': (BasicBlock, [7, 7, 7]),
    '56': (BasicBlock, [9, 9, 9]),
}


class serialNetNorm(nn.Module):
    def __init__(self, depth, num_classes=10, init_fn=None, norm_fn=None, activ_fn=None, conv=None):
        """
        Initializing a serial network
        :param depth: the depth of the network
        :param num_classes: number of classes
        :param init_fn: function for initialization
        :param norm_fn: function for normalization
        :param activ_fn: the activation function
        """
        super(serialNetNorm, self).__init__()
        self.in_planes = 16
        block, num_blocks = blocks[depth]

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

        # build the first block
        self.conv1 = self.conv(3, 16, kernel_size=3, stride=1, padding=1, bias=self.norm is None)
        if self.wn:
            self.conv1 = nn.utils.weight_norm(self.conv1, name='weight')
        if self.norm is None:
            self.block0 = nn.Sequential(self.conv1, self.acf)
        else:
            self.norm1 = self.norm(16)
            self.block0 = nn.Sequential(self.conv1, self.norm1, self.acf)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

        if init_fn is not None:
            self.apply(init_fn)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm, self.acf, self.conv, self.wn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.block0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
