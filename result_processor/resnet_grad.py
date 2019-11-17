import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# subplots = [131, 132, 133]
subplots = [111]
root = '../json/'

# exp_lists = {
#     131: ['res_bn', 'res_bn_mixup'],
#     132: ['res_fixup2s_mixup', 'res_fixup1_5s_mixup'],
#     133: ['res_l2norm', 'res_l2norm_mixup'],
# }

exp_lists = {
    111: ['res_bn', 'res_l2norm', 'res_fixup2s_mixup']
}

marker = ['b', 'r', 'y', 'g']

# label_list = {
#     131: ['BN', 'BN+mixup'],
#     132: ['Fixup@2+b&s+mixip', 'Fixup@1.5+b&s+mixup'],
#     133: ['SMN', 'SMN+mixup']
# }
# title = ['(a)', '(b)', '(c)']
label_list = {
    111: ['BN', 'SMN', 'Fixup@2+b&s+mixip']
}


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
    counter = 0
    for d in data:
        if key is 'grad_l2':
            exp = np.array(d['grad_l2']).reshape([-1, d['depth']])
        else:
            exp = np.array(d[key]).reshape(1, -1)
        if stat_ is None:
            stat_ = exp
        else:
            stat_ = np.concatenate([stat_, exp], axis=0)
        counter += 1
        break
    print(np.shape(stat_))
    mean = np.mean(stat_, axis=0)[::-1]
    std = np.std(stat_, axis=0)[::-1]
    return stat_, mean, std


plt.rcParams.update({'font.size': 13})
plt.figure(1, figsize=[6, 4])

for idx, sub in enumerate(subplots):
    ax_temp = plt.subplot(sub)
    # plt.title(title[idx], loc='left')
    # print(title[idx])
    for idx_, exp in enumerate(exp_lists[sub]):
        datas = extract_data(exp)
        stat, _m, _a= statistics(datas, 'grad_l2')
        upper = np.percentile(stat, 85, axis=0)[::-1]
        lower = np.percentile(stat, 15, axis=0)[::-1]
        mean = np.percentile(stat, 50, axis=0)[::-1]
        idx = np.arange(datas[0]['depth'])
        # print(mean)
        # print(std)
        # print(idx)
        if len(mean) > 60:
            lower = lower[::2]
            upper = upper[::2]
            mean = mean[::2]
        plt.fill_between(idx, lower, upper, color=marker[idx_], label=label_list[sub][idx_], alpha=0.4)
        plt.plot(idx, mean, marker[idx_]+'-'
                 , linewidth=2)
    plt.xlabel('Layer')
    if sub == 131 or 111:
        plt.ylabel(r"$||\Delta \theta||_2^2/\eta^2$")
    plt.yscale('log')
    plt.grid(True)
    if sub == 133:
        plt.legend(loc='upper left', fontsize=13)
    else:
        plt.legend(loc='upper left', fontsize=13)

# plt.show()

plt.savefig('./res_grad.pdf', bbox_inches='tight')