import torch.nn as nn


class MobileNet(nn.Module):
    def __init__(self, dense, norm_fn=None, acf=None, init_fn=None):
        super(MobileNet, self).__init__()
        if acf is None:
            acf = nn.ReLU(inplace=True)

        def conv_bn(inp, oup, stride):
            block = nn.Sequential()
            block.add_module('conv', nn.Conv2d(inp, oup, 3, stride, 1, bias=norm_fn is None))
            if norm_fn is not None:
                block.add_module('norm', norm_fn(oup))
            block.add_module('acf', acf())
            return block

        def conv_dw(inp, oup, stride):
            if dense:
                block = nn.Sequential()
                block.add_module('conv', nn.Conv2d(inp, oup, 3, stride, 1, bias=norm_fn is None))
                if norm_fn is not None:
                    block.add_module('norm', norm_fn(oup))
                block.add_module('acf', acf())
                return block
            else:
                block = nn.Sequential()
                block.add_module('conv_dw', nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=norm_fn is None))
                if norm_fn is not None:
                    block.add_module('norm1', norm_fn(inp))
                block.add_module('acf1', acf())
                block.add_module('conv_pw', nn.Conv2d(inp, oup, 1, 1, 0, bias=norm_fn is None))
                if norm_fn is not None:
                    block.add_module('norm2', norm_fn(oup))
                block.add_module('acf2', acf())
                return block

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)
        if init_fn is not None:
            self.apply(init_fn)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
