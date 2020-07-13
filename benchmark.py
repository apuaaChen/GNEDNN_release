import torch
import torch.nn as nn
import numpy as numpy
from moment_norm import fusedL2Norm2d
import os
import ctypes
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bm', type=int, default=0, help='benchmarks 0-3')
args = parser.parse_args()

assert torch.cuda.is_available(), "CUDA is not available"
device = torch.device('cuda')


_cudart = ctypes.CDLL('libcudart.so')


def start():
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception("cudaProfilerStart() returned %d" % ret)

def stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception("cudaProfilerStop() returned %d" % ret)


def generate_feature(bm):
    feature = torch.randn(size=bm, requires_grad=True).to(torch.float32)
    return feature.to(device)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv = nn.utils.weight_norm(self.conv, name='weight')
        #self.bn = nn.BatchNorm2d(cout)
        #self.mn = fusedL2Norm2d(cout)
        self.selu = nn.SELU(inplace=False)
        #self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        out = self.conv(x)
        #out = self.bn(out)
        #out = self.mn(out)
        out = self.selu(out)
        #out = self.relu(out)
        return out


def single_test(feature, C):
    model = Block(C, C).to(device)
    grad = torch.rand_like(feature).to(device)
    
    # warm up
    for i in range(5):
        out = model(feature)
        out.backward(grad)
    
    # start profiling
    print("start profiling")
    start()
    for i in range(5):
        out = model(feature)
        out.backward(grad)
    
    torch.cuda.synchronize()
    stop()
    print("finish profiling")

benchmarks = [
    (128, 64, 56, 56), (128, 128, 28, 28),
    (128, 256, 14, 14), (128, 512, 7, 7)
]

feature = generate_feature(benchmarks[args.bm])
single_test(feature, benchmarks[args.bm][1])