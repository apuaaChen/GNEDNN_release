import torch
import torch.nn.init as init
import numpy as np
import torch.nn as nn
import math


class Initializer:
    def __init__(self, method, nonlinearity, neg_slope=0.18, manual=-1., fixup=False, L=27., m=2., p=2., bns=False):
        init_fns = {
            'kaiming_norm': self.kaiming_normal,
            'orthogonal': self.delta_orthogonal,
        }
        self.init = init_fns[method]
        self.nonlinearity = nonlinearity
        self.manual = manual
        self.neg_slope = neg_slope

        # configures for fixup initialization
        self.fixup = fixup
        self.L = L
        self.m = m
        self.p = p
        self.bns = bns

    def gain(self):
        optimal = {
            'linear': 1.,
            'tanh': 1.,
            'relu': np.sqrt(2.),
            'leaky_relu': np.sqrt(2. / (1 + np.square(self.neg_slope))),
            'prelu': np.sqrt(2.),
            'sprelu': np.sqrt(2.),
        }
        if self.manual > 0:
            gain_ = self.manual
        else:
            gain_ = optimal[self.nonlinearity]
        if self.fixup:
            scale = np.power(self.L, -self.p/2./self.m)
            gain_ *= scale
        return gain_

    """
    Kaiming Normal initalization
    Introduced in "Delving Deep into Rectifiers: Surpassing Human-Level 
                   Performance on ImageNet Classification"
    Implement is modified from init.kaiming_normal_ of pytorch
    """

    def kaiming_normal(self, tensor, mode='fan_in'):
        fan = init._calculate_correct_fan(tensor, mode)
        gain = self.gain()
        std = gain / math.sqrt(fan)
        # print(std)
        with torch.no_grad():
            return tensor.normal_(0, std)

    """
    Orthogonal initialization
    Introduced in "Dynamical Isometry and a Mean Field Theory of CNNs: 
                   How to Train 10,000-Layer Vanilla Convolutional Neural Networks"
    Implement of "https://github.com/JiJingYu/delta_orthogonal_init_pytorch"
    """

    @staticmethod
    def _orthogonal_matrix(dim):
        """
        Creating orthogonal matrix with QR decomposition
        """
        a = torch.zeros((dim, dim)).normal_(0, 4)
        q, r = torch.qr(a)
        d = torch.diag(r, 0).sign()
        diag_size = d.size(0)
        d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
        q.mul_(d_exp)
        return q

    def delta_orthogonal(self, tensor):
        """
        Delta Orthogonal initialization
        """
        gain = self.gain()
        rows = tensor.size(1)
        cols = tensor.size(0)
        # if rows > cols:
        #     print("In_filters should not be greater than out_filters.")
        tensor.data.fill_(0)
        dim = max(rows, cols)
        q = Initializer._orthogonal_matrix(dim)
        q = torch.t(q)
        mid1 = tensor.size(2) // 2
        mid2 = tensor.size(3) // 2
        with torch.no_grad():
            if rows != 1:
                tensor[:, :, mid1, mid2] = q[:tensor.size(0), :tensor.size(1)]
            else:
                tensor[:, :, mid1, mid2] = 1

            tensor.mul_(gain)

        return tensor

    def initialization(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.Conv2d):
            self.init(m.weight)
        elif isinstance(m, Conv2d_):
            if m.marker is 'b' or 'be':
                self.init(m.weight)
            elif m.marker is 'd':
                init.kaiming_normal_(m.weight, nonlinearity='linear')
            elif m.marker is 'i':
                init.kaiming_normal_(m.weight, nonlinearity='relu')
            else:
                print('Warnning: unlabeled Conv operator')


class Conv2d_(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', marker='b'):
        super(Conv2d_, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias, padding_mode)
        self.marker = marker
