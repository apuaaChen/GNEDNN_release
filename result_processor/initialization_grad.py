import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

subplots = [231, 232, 233, 234, 235, 236]
root = '../json/'

"""
exp_lists = {
    231: ['bn_tanh', 'km_tanh', 'orth_tanh'],
    232: ['bn_relu', 'km_relu', 'orth_relu'],
    233: ['bn_lrelu', 'km_lrelu', 'orth_lrelu'],
    234: ['bn_prelu', 'km_prelu', 'orth_prelu'],
    235: ['bn_prelu', 'km_sprelu', 'orth_sprelu'],
    236: ['bn_lrelu', 'bn_lrelu3', 'bn_lrelu5', 'orth_lrelu', 'orth_lrelu3', 'orth_lrelu5'],
}
"""

exp_lists = {
    231: ['orth_tanh', 'km_tanh', 'dorth_tanh'],
    232: ['km_relu', 'orth_relu', 'dorth_relu'],
    233: ['km_lrelu2', 'orth_lrelu', 'dorth_lrelu'],
    234: ['dorth_sprelu', 'km_sprelu', 'orth_sprelu'],
    235: ['orth_lrelu', 'orth_lrelu3', 'orth_lrelu5'],
    236: ['dorth_lrelu', 'dorth_lrelu3', 'dorth_lrelu5'],
}

title = ['(a)tanh', '(b)ReLU', r'(c)lReLU, $\gamma=0.18$', '(d)sPReLU', '(e)lReLU', '(f)lReLU']

marker = ['r', 'b', 'y']
label = ['KM', 'Orth', 'Delta Orth']

marker_lrelu = ['r', 'b', 'y', 'y', 'r', 'c']
label_lrelu = [r'Orth$\gamma=0.18$', r'Orth$\gamma=0.3$',
               r'Orth$\gamma=0.5$']
label_dlrelu = [r'Delta Orth$\gamma=0.18$', r'Delta Orth$\gamma=0.3$',
                r'Delta Orth$\gamma=0.5$']


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


def statistics(data, key, c=1):
    stat_ = None
    counter = 0
    for d in data:
        # if counter == c:
        if key is 'grad_l2':
            exp = np.array(d['grad_l2']).reshape([-1, d['depth']])
        else:
            exp = np.array(d[key]).reshape(1, -1)
        if stat_ is None:
            stat_ = exp
        else:
            stat_ = np.concatenate([stat_, exp], axis=0)
        counter += 1
    print(np.shape(stat_))
    mean = np.mean(stat_, axis=0)[::-1]
    std = np.std(stat_, axis=0)[::-1]
    return stat_, mean, std


plt.rcParams.update({'font.size': 13})
plt.figure(1, figsize=[18, 8])

for idx, sub in enumerate(subplots):
    ax_temp = plt.subplot(sub)
    plt.title(title[idx], loc='left')
    print(title[idx])
    for idx_, exp in enumerate(exp_lists[sub]):
        datas = extract_data(exp)
        if sub == 234:
            if idx_ == 0:
                stat, mean, std = statistics(datas, 'grad_l2', c=0)
            else:
                stat, mean, std = statistics(datas, 'grad_l2', c=0)
        else:
            stat, mean, std = statistics(datas, 'grad_l2')
        upper = np.percentile(stat, 85, axis=0)[::-1]
        lower = np.percentile(stat, 15, axis=0)[::-1]
        mean = np.percentile(stat, 50, axis=0)[::-1]
        idx = np.arange(datas[0]['depth'])
        # print(mean)
        # print(std)
        # print(idx)
        if sub == 235:
            plt.fill_between(idx, lower, upper, color=marker_lrelu[idx_], label=label_lrelu[idx_], alpha=0.4)
            plt.plot(idx, mean, marker_lrelu[idx_]+'-'
                     , linewidth=2)
        elif sub == 236:
            plt.fill_between(idx, lower, upper, color=marker_lrelu[idx_], label=label_dlrelu[idx_], alpha=0.4)
            plt.plot(idx, mean, marker_lrelu[idx_]+'-'
                     , linewidth=2)
        else:
            plt.fill_between(idx, lower, upper, color=marker[idx_], label=label[idx_], alpha=0.4)
            plt.plot(idx, mean, marker[idx_]+'-'
                     , linewidth=2)
    plt.xlabel('Layer')
    if sub in [231, 234]:
        print('y label! %d' % sub)
        plt.ylabel(r"$||\Delta \theta||_2^2/\eta^2$")
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=13)

# plt.show()

plt.savefig('./init_grad.pdf', bbox_inches='tight')
