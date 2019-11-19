import json
import os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


subplots = [121, 122]
root = '../json/'

exp = ['bn_relu', 'l2norm', 'wn', 'l1norm']

marker = ['k-', 'r-', 'b-', 'g']
label = ['BN', 'SMN', 'sWS', 'L1MN']


def extract_data(name):
    dir = root + name + '/' + name + '_list.json'

    assert os.path.exists(dir)

    with open(dir) as json_file:
        exp_list = json.load(json_file)['exps']
    r_dir = root + name + '/'
    datas = []
    for i in exp_list:
        d_dir = r_dir + i
        assert os.path.exists(d_dir)
        with open(d_dir) as json_file:
            datas.append(json.load(json_file))
    return datas


def statistics(data, key):
    stat_ = None
    for d in data:
        exp = np.array(d[key]).reshape(1, -1)
        if stat_ is None:
            stat_ = exp
        else:
            stat_ = np.concatenate([stat_, exp], axis=0)
    mean = np.mean(stat_, axis=0)[:130]
    std = np.std(stat_, axis=0)[:130]
    return stat_, mean, std


plt.rcParams.update({'font.size': 15})
plt.figure(1, figsize=[7, 4])

for idx_, e in enumerate(exp):
    datas = extract_data(e)
    stat, mean, std = statistics(datas, 'test_acc')
    epochs = datas[0]['epoch'][:130]
    last_ten = mean[-10:]
    last_ten_std = std[-10:]
    plt.plot(epochs, mean, marker[idx_], label=label[idx_] + ' (%.2f%%+%.2f%%)' % (np.mean(last_ten), 2 * np.mean(last_ten_std))
             , linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Test Acc')
plt.ylim(35, 92)
plt.grid(True)
plt.legend(loc='lower right', fontsize=15)
# plt.show()
plt.savefig('./norm_exp.pdf', bbox_inches='tight')
