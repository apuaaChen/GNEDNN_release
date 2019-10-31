from imagenet.models import resnet_mn, mobilenet, fixup_resnet, fixup_resnet_ours
from ops import L2Norm2d, L1Norm2d


ResNetL2 = resnet_mn.resnet50(L2Norm2d, pretrained=False)
ResNetL1 = resnet_mn.resnet50(L1Norm2d, pretrained=False)

ResNet_fixup = fixup_resnet.resnet50(pretrained=False)
ResNet_fixup_ours = fixup_resnet_ours.resnet50(pretrained=False)

MobileNet = mobilenet.MobileNet
