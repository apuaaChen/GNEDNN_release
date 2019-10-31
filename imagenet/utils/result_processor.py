import json
import os
import sys


class Results:
    """
    Record the results and store them to the json files
    """
    def __init__(self, exp_name, writer):
        self.exp_name = exp_name
        self.writer = writer
        if not os.path.exists('./json/' + exp_name):
            os.mkdir('./json/' + exp_name)

        self.exp_path = './json/' + exp_name + '/result.json'

        # See whether this experiment exists
        if os.path.exists(self.exp_path):
            print('This experiment has ben run.')
            sys.exit()

        # create the dictionary container for results to be recorded

        self.result = {
            'name': exp_name,
            'epoch': [],
            'train_loss': [],
            'train_acc@1': [],
            'train_acc@5': [],
            'test_acc@1': [],
            'test_acc@5': [],
        }

    def record(self, loss, top1_t, top5_t, acc1, acc5, epoch, lr=0.1):
        self.writer.add_scalar('train_loss', loss, global_step=epoch)
        self.writer.add_scalar('train_acc@1', top1_t, global_step=epoch)
        self.writer.add_scalar('train_acc@5', top5_t, global_step=epoch)
        self.writer.add_scalar('test_acc@1', acc1, global_step=epoch)
        self.writer.add_scalar('test_acc@5', acc5, global_step=epoch)
        self.writer.add_scalar('lr', lr, global_step=epoch)
        self.result['train_loss'].append(loss)
        self.result['epoch'].append(epoch)
        self.result['train_acc@1'].append(top1_t)
        self.result['train_acc@5'].append(top5_t)
        self.result['test_acc@1'].append(acc1)
        self.result['test_acc@5'].append(acc5)

    def write_result(self):
        # write json file
        with open(self.exp_path, 'w') as file:
            json.dump(self.result, file)
        self.writer.close()
