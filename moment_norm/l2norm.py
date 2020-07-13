import torch.nn as nn
import torch
import mnorm

class L2Norm2dfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps, rv, momentum):
        x, y, sm, scale, running_var = mnorm.fused_l2_norm_fv2(x, weight, bias, rv, momentum, eps)   
        ctx.save_for_backward(x, scale, weight, sm)
        ctx.eps = eps
        return y, running_var
    
    @staticmethod
    def backward(ctx, grad_y, grad_rv):
        x, scale, weight, sm = ctx.saved_tensors
        grad_x, grad_weight, grad_bias = mnorm.fused_l2_norm_bv2(x, grad_y, scale, weight, sm, ctx.eps)
        #grad_x_, grad_weight_, grad_bias_ = mnorm.fused_l2_norm_b(x, grad_y, scale, weight, sm, ctx.eps)
        
        return grad_x, grad_weight, grad_bias, None, None, None


def l2norm2dfn_inf(x, sm, weight, bias, eps):
    y = mnorm.fused_l2_norm_inf(x, weight, sm, bias, eps)
    return y

        
l2norm2dfn = L2Norm2dfn.apply


class fusedL2Norm2d(nn.BatchNorm2d):
    def forward(self, x):
        self._check_input_dim(x)

        if self.training is not True:
            y = l2norm2dfn_inf(x, self.running_var, self.weight, self.bias, self.eps)
        else:
            y, self.running_var = l2norm2dfn(x, self.weight, self.bias, self.eps, self.running_var.detach(), self.momentum)                                     
        return y
    

class GhostMN(nn.Module):
    def __init__(self, plane):
        super(GhostMN, self).__init__()
        self.mn = fusedL2Norm2d(plane * 32)
    def forward(self, x):
        N, C, H, W = x.size()
        x_ = x.view((int(N / 32), C * 32, H, W))
        return self.mn(x_).view(x.size())


""" backup
class L2Norm2dfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        z2 = x ** 2
        sm2 = torch.sqrt(z2.sum(dim=(1), keepdim=True) / (x.size(0) * x.size(2) * x.size(3))).view(1, -1, 1, 1)
        scale2 = weight.view(1, -1, 1, 1) / (sm2 + eps)
        y2 = scale2 * x + bias.view(1, -1, 1, 1)
        
        ctx.save_for_backward(x, scale2, weight, 1./(sm2+eps), sm2)
        return y2, sm2
    
    @staticmethod
    def backward(ctx, grad_y, grad_sm_out):
        x, scale2, weight, sme, sm = ctx.saved_tensors
        grad_scale = (grad_y * x).sum(dim=(0, 2, 3), keepdim=True)
        grad_bias = grad_y.sum(dim=(0, 2, 3))
        
        grad_weight = (grad_scale * sme).view(-1)
        grad_sm = -grad_scale * sme * scale2 / 2 / sm
        
        grad_x = grad_sm.view(1, -1, 1, 1) / (x.size(0) * x.size(2) * x.size(3))  * 2 * x + grad_y * scale2
        
        return grad_x, grad_weight, grad_bias, None
"""