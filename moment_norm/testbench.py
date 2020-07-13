# testbench for second order moment normalization
import torch
import numpy as np
from ops import L2Norm2d
from moment_norm import fusedL2Norm2d
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


assert torch.cuda.is_available(), "CUDA is not available"
device = torch.device('cuda')


def single_test():
    # random input feature map size
    N = np.random.randint(low=1, high=128)
    C = np.random.randint(low=1, high=128)
    H = np.random.randint(low=7, high=128)
    W = np.random.randint(low=7, high=128)
    sigma = np.random.uniform(low=0.2, high=5.0, size=(N, C, H, W))
    
    norm = L2Norm2d(num_features=C).to(device)
    fnorm = fusedL2Norm2d(num_features=C).to(device)
    fnorm.weight.values = norm.weight.values
    fnorm.bias.values = norm.bias.values
    
    feature = torch.randn(size=(N, C, H, W), requires_grad=False) * sigma
    feature = feature.to(torch.float32)
    
    feat_in = feature.clone().to(device).requires_grad_(True)
    feat_in_f = feature.clone().to(device).requires_grad_(True)
    
    feat_out = norm(feat_in)
    feat_out_f = fnorm(feat_in_f)
    
    grad = torch.rand_like(feat_out)
    
    feat_out.backward(grad)
    feat_out_f.backward(grad)
    
    grad_in = feat_in.grad
    grad_in_f = feat_in_f.grad
    
    grad_w_in = norm.weight.grad
    grad_b_in = norm.bias.grad
    
    grad_w_in_f = fnorm.weight.grad
    grad_b_in_f = fnorm.bias.grad
    
    error_out = torch.abs(feat_out_f - feat_out)
    g_error_in = torch.abs(grad_in - grad_in_f)
    g_error_w = torch.abs(grad_w_in - grad_w_in_f)
    g_error_b = torch.abs(grad_b_in - grad_b_in_f)
    
    fnorm.eval()
    norm.eval()
    
    inf_feat_out = norm(feat_in)
    inf_feat_out_f = fnorm(feat_in_f)
    
    inf_error_out = torch.abs(inf_feat_out_f - inf_feat_out)
    
    passed = True
    max_error_out = torch.max(error_out).item()
    if max_error_out > 1e-5 or np.isnan(max_error_out):
        print("[Forward] there are %d different entries in overall %d entries. The maximum difference is %f" 
              % (torch.nonzero(error_out).size(0), error_out.numel(), max_error_out))
        passed = False
    
    max_g_error_in = torch.max(g_error_in).item()
    if max_g_error_in > 1e-5 or np.isnan(max_g_error_in):
        print("[Grad in] there are %d different entries in overall %d entries. The maximum difference is %f" 
              % (torch.nonzero(g_error_in).size(0), g_error_in.numel(), max_g_error_in))
        passed = False
        
    max_g_error_w = torch.max(g_error_w).item()
    if max_g_error_w > 1e-5 or np.isnan(max_g_error_w):
        print("[Grad weight] there are %d different entries in overall %d entries. The maximum difference is %f" 
              % (torch.nonzero(g_error_w).size(0), g_error_w.numel(), max_g_error_w))
        passed = False
        
    max_g_error_b = torch.max(g_error_b).item()
    if max_g_error_b > 1e-5 or np.isnan(max_g_error_b):
        print("[Grad bias] there are %d different entries in overall %d entries. The maximum difference is %f" 
              % (torch.nonzero(g_error_b).size(0), g_error_b.numel(), max_g_error_b))
        passed = False

    max_inf_error_out = torch.max(inf_error_out).item()
    if max_inf_error_out > 1e-5 or np.isnan(max_inf_error_out):
        print("[Inference] there are %d different entries in overall %d entries. The maximum difference is %f" 
              % (torch.nonzero(inf_error_out).size(0), inf_error_out.numel(), max_inf_error_out))
        passed = False
    
    return passed


num_pass = 0;
for i in range(10):
    if single_test():
        num_pass += 1

print("%d out of %d tests passed" % (num_pass, 10))
    
    

