import os

import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from ops import mixup_data, mixup_criterion
from utils import Results, average


class Estimator:
    def __init__(self, t_loader, v_loader, net, loss_fn, optim, device, lr, log, list, decay=(), mixup=False):
        """
        A class that handles training and inference for a classifier
        :param t_loader: training set loader
        :param v_loader: test set loader
        :param net: the network
        :param loss_fn: loss function
        :param optim: optimizer
        :param device: the device to work on
        :param lr: initial learning rate
        :param decay: learning rate decay epochs
        """
        self.t_loader = t_loader
        self.v_loader = v_loader
        self.optim = optim
        self.net = net
        self.loss_fn = loss_fn
        self.device = device
        self.state = {
            'net': self.net.state_dict(),
            'acc': 0,
            'epoch': 0,
        }
        self.epoch = 0
        self.lr = lr
        self.decay = decay
        self.log = log
        self.writer = SummaryWriter(log_dir='./runs/' + list + '/' + log)
        self.mixup = mixup
        self.results = Results(list_name=list+"_list", exp_name=log, json_dir='./json/' + list + '/')
        self.results.result['grad_l2'] = []

        counter = 0
        for item in self.net.parameters():
            if len(item.size()) > 1:
                counter += 1
        self.results.result['depth'] = counter

        # register the gradient information
        def gradient_process(grad):
            if len(self.results.result['grad_l2']) < self.results.result['depth'] * 391 * 3:
                # the gradient is stored from the gradient
                temp = grad.view(-1)**2
                self.results.result['grad_l2'].append(temp.sum(0).item())

        counter = 0
        for item in self.net.parameters():
            if len(item.size()) > 1:
                item.register_hook(lambda grad: gradient_process(grad))
                counter += 1

    def lr_decay(self):
        if self.epoch in self.decay:
            self.lr = self.lr / 10.
            for param_group in self.optim.param_groups:
                param_group['lr'] = self.lr

    def training(self):
        train_loss = average()
        train_loss.clear()
        self.net.train()  # set the model to train mode
        correct = 0
        total = 0
        self.lr_decay()
        for index, (inputs, labels) in enumerate(self.t_loader, 0):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optim.zero_grad()

            if self.mixup:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 1, True)
                inputs, labels_a, labels_b = map(Variable, (inputs, labels_a, labels_b))
                outputs = self.net(inputs)
                loss = mixup_criterion(self.loss_fn, outputs, labels_a, labels_b, lam)
            else:
                outputs = self.net(inputs)
                loss = self.loss_fn(outputs, labels)
            loss.backward()
            # print(loss.item())
            torch.nn.utils.clip_grad_value_(self.net.parameters(), clip_value=2.)
            self.optim.step()
            train_loss.add(loss)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        self.writer.add_scalar('train_loss', train_loss.value, global_step=self.epoch)
        self.writer.add_scalar('train_acc', float(correct)/float(total), global_step=self.epoch)
        self.writer.add_scalar('lr', self.lr, global_step=self.epoch)

        self.results.result['train_loss'].append(train_loss.value)
        self.results.result['train_acc'].append(float(correct)/float(total))
        self.results.result['epoch'].append(self.epoch)

        self.epoch += 1
        return train_loss.value

    def inference(self):
        self.net.eval()  # set the model to evaluation mode
        test_loss = average()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.v_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # upload data to device
                outputs = self.net(inputs)  # inference
                loss = self.loss_fn(outputs, labels)
                test_loss.add(loss.item())
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        acc = 100. * correct / total
        self.writer.add_scalar('test_acc', acc, global_step=self.epoch - 1)
        self.results.result['test_acc'].append(acc)
        if acc > self.state['acc']:
            self.state['acc'] = acc
            self.state['net'] = self.net.state_dict()
            self.state['epoch'] = self.epoch
            if not os.path.isdir('./checkpoint'):
                os.mkdir('./checkpoint')
            torch.save(self.state, './checkpoint/%s.t7' % self.log)
        return acc
