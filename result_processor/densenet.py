import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

subplots = [121, 122]
root = '../json/'

exp_lists = {
    121: ['dn_bn', 'dn_km_relu', 'dn_l2norm'],
    122: ['dn_orth_lrelu3', 'dn_orth_seluv2_g2_e3', 'dn_wn'],
}

marker = ['b', 'r', 'y']
label_list = {
    121: ['BN', 'KM_ReLU', 'SMN'],
    122: ['lReLU', 'SeLU', 'WN'],
}
title = ['(a)', '(b)']


def extract_data(name):
    dir = root + name + '/' + name + '_list.json'
    print(dir)
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
    mean = np.mean(stat_, axis=0)
    std = np.std(stat_, axis=0)
    return stat_, mean, std


plt.rcParams.update({'font.size': 13})
plt.figure(1, figsize=[8, 8])

for idx, sub in enumerate(subplots):
    plt.subplot(sub)
    plt.title(title[idx], loc='left')
    for idx_, exp in enumerate(exp_lists[sub]):
        datas = extract_data(exp)
        stat, mean, std = statistics(datas, 'test_acc')
        epochs = datas[0]['epoch']
        last_ten = mean[-10:]
        last_ten_std = std[-10:]
        plt.plot(epochs, mean, marker[idx_], label=label_list[sub][idx_]+ ' (%.2f%%+%.2f%%)' % (np.mean(last_ten), 2 * np.mean(last_ten_std))
                 , linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Acc')
    plt.ylim(35, 95)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=13)

# plt.show()

plt.savefig('./selu_exp.pdf', bbox_inches='tight')
