# benchmarks for second order moment normalization
import torch
import torch.nn as nn
import numpy as np
from ops import L2Norm2d
from moment_norm import fusedL2Norm2d
import os
import ctypes
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bm', type=int, default=0, help='benchmarks 0-8')
args = parser.parse_args()


# os.environ["CUDA_VISIBLE_DEVICES"] = '2'


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


def single_test(feature, C):
    # L2 normalization
    norm = L2Norm2d(num_features=C).to(device)
    # fused L2 normalization
    fnorm = fusedL2Norm2d(num_features=C).to(device)
    # batch normalization
    bn = nn.BatchNorm2d(num_features=C).to(device)
    
    grad = torch.rand_like(feature)
    
    # warm up
    for i in range(10):
        outn = norm(feature)
        outn.backward(grad)
    
    for i in range(10):
        outf = fnorm(feature)
        outf.backward(grad)
    
    for i in range(10):
        outb = bn(feature)
        outb.backward(grad)
    
    # start profiling
    print("start profiling")
    start()
    for i in range(5):
        outn = norm(feature)
        outn.backward(grad)
    
    for i in range(5):
        outf = fnorm(feature)
        outf.backward(grad)
    
    for i in range(5):
        outb = bn(feature)
        outb.backward(grad)
    
    torch.cuda.synchronize()
    stop()
    print("stop profiling")



benchmarks = [
    (128, 64, 112, 112), (128, 64, 56, 56), (128, 256, 56, 56),
    (128, 128, 28, 28), (128, 512, 28, 28), (128, 256, 14, 14),
    (128, 1024, 14, 14), (128, 512, 7, 7), (128, 2048, 7, 7)
]

feature = generate_feature(benchmarks[args.bm])
single_test(feature, benchmarks[args.bm][1])
