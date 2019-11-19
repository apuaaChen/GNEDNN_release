import argparse
import os
import random
import shutil
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from imagenet.models import MobileNet
from tqdm import tqdm
from tensorboardX import SummaryWriter
from imagenet.utils.result_processor import Results
import numpy as np
from ops import SeLUv2, Initializer

# Dataset and model configuration
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', type=str, default='/raid/datasets/ImageNet2012/',
                    help='path to dataset, the imagenet-folder with train and val folders')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

# Training configuration
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# Training-related hyper-parameters
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

# result-related configures
# parser.add_argument('-p', '--print-freq', default=10, type=int,
#                     metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

# for multi-distributed training
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--gpus', default='2,3', help='using which GPU')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# for logging
parser.add_argument('--log', type=str, default='test', help='the name for the log')

# for SeLU
parser.add_argument('--selu', action='store_true', help='using selu instead of BN')
parser.add_argument('--fixpoint', default=1., type=float, help='fixpoint of 2nd moment in SeLU')
parser.add_argument('--epsilon', default=0.03, type=float, help='epsilon in SeLU')
parser.add_argument('--init_fn', choices=['kaiming_norm', 'orthogonal', 'delta_orth'], default='kaiming_norm')
parser.add_argument('--norm_fn', choices=['none', 'bn'], default='bn')
parser.add_argument('--gain', default=1., type=float, help='manual gain for initialization')

# activation function
parser.add_argument('--neuron', choices=['relu', 'tanh', 'leaky_relu', 'prelu', 'sprelu', 'selu', 'seluv2'],
                    default='relu')
parser.add_argument('--neg_slope', default=0.18, type=float, help='negative slope for leaky ReLU')

parser.add_argument('--dense', action='store_true', help='using dense model')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

best_acc1 = 0


def main():
    # configure the random seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # configure the gpus to use
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # If use multiple nodes or multiple process per node, use distributed training
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # count the number of gpus available
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size  # the total number of gpu nodes
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        # If one of the processes exits with a non-zero exit status, the remaining process are killed,
        # an exception will be raised with the cause of termination
        # each gpu has its own process for working
        # args: arguments passed to ''fn''
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


# The main worker
def main_worker(gpu, ngpus_per_node, args_):
    global best_acc1
    args_.gpu = gpu

    if args_.gpu is not None:
        print("Use GPU: {} for training".format(args_.gpu))

    # what is rank
    if args_.distributed:
        if args_.dist_url == "env://" and args_.rank == -1:
            args_.rank = int(os.environ["RANK"])
        if args_.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args_.rank = args_.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args_.dist_backend, init_method=args_.dist_url,
                                world_size=args_.world_size, rank=args_.rank)

    # create model
    class SeLUv2_(SeLUv2):
        def __init__(self):
            super(SeLUv2_, self).__init__(gamma_=np.square(args.gain), fixpoint=args.fixpoint,
                                          epsilon_=args.epsilon)

    class leakyReLU_(nn.LeakyReLU):
        def __init__(self):
            super(leakyReLU_, self).__init__(negative_slope=args.neg_slope, inplace=True)

    # create model
    neurons = {
        'relu': nn.ReLU,
        'leaky_relu': leakyReLU_,
        'tanh': nn.Tanh,
        'prelu': nn.PReLU,
        'selu': nn.SELU,
        'seluv2': SeLUv2_,
    }
    norm_fns = {
        'none': None,
        'bn': nn.BatchNorm2d
    }

    initializer = Initializer(method=args_.init_fn, nonlinearity=args_.neuron,
                              neg_slope=args_.neg_slope, manual=args_.gain)
    model = MobileNet(dense=args_.dense, norm_fn=norm_fns[args_.norm_fn], acf=neurons[args_.neuron],
                      init_fn=initializer.initialization)

    if args_.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args_.gpu is not None:
            torch.cuda.set_device(args_.gpu)
            model.cuda(args_.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args_.batch_size = int(args_.batch_size / ngpus_per_node)
            args_.workers = int((args_.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args_.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args_.gpu is not None:
        torch.cuda.set_device(args_.gpu)
        model = model.cuda(args_.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args_.arch.startswith('alexnet') or args_.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args_.gpu)

    def add_weight_decay(net, weight_decay):
        """
        Filter the weights that should not be decayed
        """
        decay, no_decay = [], []
        for name, param in net.named_parameters():
            if ('conv' in name or 'fc' in name) and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        assert len(list(net.parameters())) == len(decay) + len(no_decay)
        print('decay: %d; not decay: %d' % (len(decay), len(no_decay)))
        params = [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0)]
        return params

    params = add_weight_decay(model, weight_decay=args_.weight_decay)
    optimizer = torch.optim.SGD(params, args_.lr,
                                momentum=args_.momentum)

    """ Original weight decay also decays the bn
    optimizer = torch.optim.SGD(model.parameters(), args_.lr,
                                momentum=args_.momentum,
                                weight_decay=args_.weight_decay)
    """

    # optionally resume from a checkpoint
    if args_.resume:
        if os.path.isfile(args_.resume):
            print("=> loading checkpoint '{}'".format(args_.resume))
            if args_.gpu is None:
                checkpoint = torch.load(args_.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args_.gpu)
                checkpoint = torch.load(args_.resume, map_location=loc)
            args_.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args_.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args_.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args_.data, 'train')
    valdir = os.path.join(args_.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args_.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args_.batch_size, shuffle=(train_sampler is None),
        num_workers=args_.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args_.batch_size, shuffle=False,
        num_workers=args_.workers, pin_memory=True)

    if args_.evaluate:
        validate(val_loader, model, criterion, args_)
        return

    # training progress bar
    pbar = tqdm(range(args_.start_epoch, args_.epochs))

    writer = SummaryWriter(log_dir='./runs/' + args.log)

    result = Results(exp_name=args.log, writer=writer)

    adjust_lr = LearningRate()

    for epoch in pbar:
        if args_.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args_)

        # train for one epoch
        loss, top1_t, top5_t = train(train_loader, model, criterion, optimizer, args_)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args_)

        # set progress bar
        pbar.set_description('Loss: %.3f|Test Acc: %.3f%%' % (loss, acc1))
        # right tensorboard
        result.record(loss, top1_t, top5_t, acc1, acc5, epoch, lr=adjust_lr.lr)

        # remember best acc@1 and save checkpoint
        # if best_acc1 + 0.0002 > acc1:
        #     best_counter += 1
        # else:
        #     best_counter = 0

        # if best_counter > 10:
        #     adjust_lr.decay(optimizer)
        #     best_counter = 0
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args_.multiprocessing_distributed or (args_.multiprocessing_distributed
                                                     and args_.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
    result.write_result()


def train(train_loader, model, criterion, optimizer, args_):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):

        if args_.gpu is not None:
            images = images.cuda(args_.gpu, non_blocking=True)
        target = target.cuda(args_.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        if i % 10 == 0:
            print('iteration: %d| loss: %.2f' % (i, loss.item()))

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, args_):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if args_.gpu is not None:
                images = images.cuda(args_.gpu, non_blocking=True)
            target = target.cuda(args_.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint_m.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_m.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class LearningRate:
    def __init__(self):
        self.lr = args.lr

    def decay(self, optimizer):
        self.lr = self.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr


def adjust_learning_rate(optimizer, epoch, args_):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 60:
        lr = args_.lr
    elif epoch >= 60 and epoch < 75:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
