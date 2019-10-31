import torch.nn as nn
import torch.nn.functional as F
import torch
from torch._jit_internal import weak_script_method
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve


class sPReLU(nn.PReLU):
    def __init__(self, num_parameters=1, init=0.25):
        super(sPReLU, self).__init__(num_parameters, init)

    @weak_script_method
    def forward(self, input):
        self.weight.clamp(0, 0.5)
        return F.prelu(input, self.weight) / torch.sqrt(1. + self.weight ** 2)


class SeLUv2(nn.Module):
    def __init__(self, gamma_, fixpoint, epsilon_=0.05):
        """
        Implementation of SeLU
        :param gamma_: the std of weights
        :param fixpoint: the fixpoint of the pre-activations' second-order moment
        :param epsilon_: the epsilon
        """
        super(SeLUv2, self).__init__()
        self.g_ = gamma_
        self.fp = fixpoint
        self.e = epsilon_

        v = self.fp * self.g_
        phi_fn = lambda v, l_, a_: \
            (np.square(l_*a_)*np.exp(2.*v)*norm(0, np.sqrt(v)).cdf(-2.*v) + np.square(l_)/2.) * self.g_

        def func(i):
            l_, a_ = i[0], i[1]
            return [
                phi_fn(v, l_, a_) - (1+self.e),
                (np.square(l_*a_)*(np.exp(2.*v)*norm(0, np.sqrt(v)).cdf(-2.*v) - 2. * np.exp(v/2.)*norm(0, np.sqrt(v)).cdf(-v))
                 + 0.5*np.square(l_*a_) + 0.5*l_*l_*v)/self.fp - 1.
            ]

        [self.lamb_, self.alpha_] = fsolve(func, [1., 1.])
        # print(r'$\lambda=%.4f,~~\alpha=%.4f$' % (self.lamb_, self.alpha_))

    def forward(self, input):
        return self.lamb_ * F.elu(input, self.alpha_)
