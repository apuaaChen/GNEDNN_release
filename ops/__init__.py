"""
Normalization operations that will be used in our Models.
"""
from ops import normalization, initialization, mixup, neurons


# initialization ops
Initializer = initialization.Initializer
Conv2d_ = initialization.Conv2d_

# mixup
mixup_data = mixup.mixup_data
mixup_criterion = mixup.mixup_criterion

# normalization ops
CenterConv2d = normalization.CenterConv2d
WnConv2d = normalization.WnConv2d
L2Norm2d = normalization.L2Norm2d
L1Norm2d = normalization.L1Norm2d
MeanOnlyBN = normalization.MeanOnlyBN

# neurons
sPReLU = neurons.sPReLU
SeLUv2 = neurons.SeLUv2
