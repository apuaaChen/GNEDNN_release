import argparse
import os
import torch
from cifar.utils import Cifar10Provider, Estimator
from cifar.models import ResNet
from ops import Initializer, Conv2d_, sPReLU, CenterConv2d, L2Norm2d, WnConv2d
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm

# Description
parser = argparse.ArgumentParser(description='Experiments in Sec. 7.7: ResNet')

# Basic configurations
parser.add_argument('--gpu', default='0', help='using which GPU')
parser.add_argument('--dataset', choices=['cifar10'], default='cifar10', help='dataset')
parser.add_argument('--size', default='32', type=str, help='model size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=130, type=int, help='total training epochs')
parser.add_argument('--root', default='/', help='location of cifar-10 dataset')

# initialization method
parser.add_argument('--init_fn', choices=['kaiming_norm', 'orthogonal'], default='kaiming_norm',
                    help='the initialization method for models')

# activation function
parser.add_argument('--neuron', choices=['relu', 'tanh', 'leaky_relu', 'prelu', 'sprelu'], default='relu')
parser.add_argument('--neg_slope', default=0.18, type=float, help='negative slope for leaky ReLU')

# normalization function
parser.add_argument('--norm_fn', choices=['none', 'bn', 'l2n', 'wn', 'swn'], default='bn')
parser.add_argument('--scale', default=1., type=float, help='the scaling factor for normalization')

# convolution function
parser.add_argument('--conv', choices=['naive', 'centered', 'normed'], default='naive', help='convolution function')

# mixup initialization
parser.add_argument('--mixup', action='store_true', help='using mixup augmentation')

# logging configures
parser.add_argument('--log', default='log', help='log name of experiment')
parser.add_argument('--list', default='list', help='list name of the experiments')

# gain
parser.add_argument('--gain', type=float, default=-1, help='the gain for parameter initialization')

# for fixup
parser.add_argument('--fixup', action='store_true', help='using fixup initialization')
parser.add_argument('--p', type=float, default=1., help='p factor for fixup initialization')
parser.add_argument('--m', type=float, default=2., help='number of layers in each residual block')
parser.add_argument('--bns', action='store_true', help='apply the scalar scaling and bias in fixup')

args = parser.parse_args()

# setup device
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# get the dataset
data = Cifar10Provider(root=args.root, workers=8)

# neuron configure
neurons = {
    'relu': nn.ReLU(True),
    'leaky_relu': nn.LeakyReLU(negative_slope=args.neg_slope, inplace=True),
    'tanh': nn.Tanh(),
    'prelu': nn.PReLU(),
    'sprelu': sPReLU(),
}

L = (float(args.size) - 2.) / args.m

# initialization method configure
initializer = Initializer(method=args.init_fn, nonlinearity=args.neuron,
                          neg_slope=args.neg_slope, manual=args.gain,
                          fixup=args.fixup, L=L, m=args.m, p=args.p, bns=args.bns)

# normalization method configure
norm_fns = {
    'none': None,
    'bn': nn.BatchNorm2d,
    'l2n': L2Norm2d,
    'wn': 'wn',
}

# convolution method configure
convs = {
    'naive': Conv2d_,
    'centered': CenterConv2d,
    'normed': WnConv2d,
}

if args.conv is 'normed':
    assert args.neuron is 'relu'

# setup model
Net_Module = ResNet
net = Net_Module(depth=args.size, init_fn=initializer,
                 norm_fn=norm_fns[args.norm_fn], activ_fn=neurons[args.neuron], conv=convs[args.conv])

# print the network structure for checking
# print(net)

# upload network to cuda
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    if torch.cuda.device_count() > 1:
        print('Using %d GPUs' % torch.cuda.device_count())
    cudnn.benchmark = True  # for a little speedup


def add_weight_decay(net, weight_decay):
    """
    Filter the weights that should not be decayed
    """
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if 'conv' in name or 'linear' in name:
            decay.append(param)
        else:
            no_decay.append(param)
    assert len(list(net.parameters())) == len(decay) + len(no_decay)
    params = [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0)]
    return params


# setup loss function & optimizer
loss_fn = nn.CrossEntropyLoss()

if not args.bns:
    params = add_weight_decay(net, weight_decay=5e-4)
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)

else:
    print('hhh')
    parameters_bias = [p[1] for p in net.named_parameters() if 'bias' in p[0]]
    parameters_scale = [p[1] for p in net.named_parameters() if 'scale' in p[0]]
    parameters_others = [p[1] for p in net.named_parameters() if not ('bias' in p[0] or 'scale' in p[0])]
    optimizer = optim.SGD(
        [{'params': parameters_bias, 'lr': args.lr / 10.},
         {'params': parameters_scale, 'lr': args.lr / 10.},
         {'params': parameters_others}],
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4)


# setup estimator
estim = Estimator(t_loader=data.trainloader, v_loader=data.testloader, net=net,
                  loss_fn=loss_fn, optim=optimizer, device=device, lr=args.lr, log=args.log, decay=(100, 150),
                  mixup=args.mixup, list=args.list)

# training
pbar = tqdm(range(args.epochs))

for e in pbar:
    train_loss = estim.training()
    acc = estim.inference()
    pbar.set_description('Loss: %.3f|Test Acc: %.3f%%' % (train_loss, acc))

# write result to json file
estim.results.write_result()
