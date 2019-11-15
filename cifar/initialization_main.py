import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from cifar.models import SerialNet
from cifar.utils import Cifar10Provider, Estimator
from ops import Initializer, sPReLU, SeLUv2


# Description
parser = argparse.ArgumentParser(description='Experiments in Sec. 7.3: Initialization in Serial Network;'
                                             'Sec. 7.5: Self-Normalizing Neural Network')

# Basic configurations
parser.add_argument('--gpu', default='0', help='using which GPU')
parser.add_argument('--dataset', choices=['cifar10'], default='cifar10', help='dataset')
parser.add_argument('--size', default='32', type=str, help='model size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=130, type=int, help='total training epochs')
parser.add_argument('--root', default='/', help='location of cifar-10 dataset')

# initialization method
parser.add_argument('--init_fn', choices=['kaiming_norm', 'orthogonal', 'delta_orth'], default='kaiming_norm',
                    help='the initialization method for models')

# activation function
parser.add_argument('--neuron', choices=['relu', 'tanh', 'leaky_relu', 'prelu', 'sprelu', 'selu', 'seluv2'],
                    default='relu')
parser.add_argument('--neg_slope', default=0.18, type=float, help='negative slope for leaky ReLU')

# normalization function
parser.add_argument('--norm_fn', choices=['none', 'bn'], default='bn')

# mixup initialization
parser.add_argument('--mixup', action='store_true', help='using mixup augmentation')

# For SeLU
parser.add_argument('--fixpoint', default=1., type=float, help='fixpoint of 2nd moment in SeLU')
parser.add_argument('--epsilon', default=0.07, type=float, help='epsilon in SeLU')

# logging configures
parser.add_argument('--log', default='log', help='log name of experiment')
parser.add_argument('--list', default='list', help='list name of the experiments')

# gain
parser.add_argument('--gain', type=float, default=-1, help='the gain for parameter initialization')

# weight decay
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')

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
    'selu': nn.SELU(),
    'seluv2': SeLUv2(gamma_=np.square(args.gain), fixpoint=args.fixpoint, epsilon_=args.epsilon)
}

# initialization method configure
initializer = Initializer(method=args.init_fn, nonlinearity=args.neuron,
                          neg_slope=args.neg_slope, manual=args.gain)


# normalization method configure
norm_fns = {
    'none': None,
    'bn': nn.BatchNorm2d
}

# setup model
net = SerialNet(depth=args.size, init_fn=initializer.initialization,
                norm_fn=norm_fns[args.norm_fn], activ_fn=neurons[args.neuron])

# print the network structure for checking
# print(net)

# upload network to cuda
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    if torch.cuda.device_count() > 1:
        print('Using %d GPUs' % torch.cuda.device_count())
    cudnn.benchmark = True  # for a little speedup


def add_weight_decay(net_, weight_decay):
    """
    Filter the weights that should not be decayed
    """
    decay, no_decay = [], []
    for name, param in net_.named_parameters():
        if 'conv' in name or 'linear' in name:
            decay.append(param)
        else:
            no_decay.append(param)
    assert len(list(net_.parameters())) == len(decay) + len(no_decay)
    params_ = [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0)]
    return params_


# setup loss function & optimizer
loss_fn = nn.CrossEntropyLoss()

params = add_weight_decay(net, weight_decay=args.wd)
optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)

# setup estimator
estim = Estimator(t_loader=data.trainloader, v_loader=data.testloader, net=net,
                  loss_fn=loss_fn, optim=optimizer, device=device, lr=args.lr, log=args.log, decay=(80,),
                  mixup=args.mixup, list=args.list)

# training
pbar = tqdm(range(args.epochs))

for e in pbar:
    train_loss = estim.training()
    acc = estim.inference()
    pbar.set_description('Loss: %.3f|Test Acc: %.3f%%' % (train_loss, acc))

# write result to json file
estim.results.write_result()
