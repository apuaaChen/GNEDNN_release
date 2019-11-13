import json
import os
import numpy as np
import matplotlib.pyplot as plt

subplots = [231, 232, 233, 234, 235, 236]
root = '../json/'

exp_lists = {
    231: ['bn_tanh', 'km_tanh', 'orth_tanh'],
    232: ['bn_relu', 'km_relu', 'orth_relu'],
    233: ['bn_lrelu', 'km_lrelu', 'orth_lrelu'],
    234: ['bn_prelu', 'km_prelu', 'orth_prelu'],
    235: ['bn_prelu', 'km_sprelu', 'orth_sprelu'],
    236: ['bn_lrelu', 'bn_lrelu3', 'bn_lrelu5', 'orth_lrelu', 'orth_lrelu3', 'orth_lrelu5'],
}

title = ['(a)tanh', '(b)ReLU', r'(c)lReLU, $\gamma=0.18$', '(d)PReLU', '(e)sPReLU', '(f)lReLU']

marker = ['k-', 'r-', 'b-']
label = ['BN', 'KM', 'Orth']

marker_lrelu = ['k--', 'r--', 'b--', 'k-', 'r-', 'b']
label_lrelu = [r'BN$\gamma=0.18$', r'BN$\gamma=0.3$', r'BN$\gamma=0.5$', r'Orth$\gamma=0.18$', r'Orth$\gamma=0.3$',
               r'Orth$\gamma=0.5$']


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
    mean = np.mean(stat_, axis=0)
    std = np.std(stat_, axis=0)
    return stat_, mean, std


plt.rcParams.update({'font.size': 13})
plt.figure(1, figsize=[14, 8])

for idx, sub in enumerate(subplots):
    plt.subplot(sub)
    plt.title(title[idx], loc='left')
    for idx_, exp in enumerate(exp_lists[sub]):
        datas = extract_data(exp)
        stat, mean, std = statistics(datas, 'test_acc')
        epochs = datas[0]['epoch']
        last_ten = mean[-10:]
        last_ten_std = std[-10:]
        if sub == 236:
            plt.plot(epochs, mean, marker_lrelu[idx_], label=label_lrelu[idx_] + ' (%.2f%%+%.2f%%)' % (np.mean(last_ten), 2 * np.mean(last_ten_std))
                     , linewidth=2)
        else:
            plt.plot(epochs, mean, marker[idx_], label=label[idx_]+ ' (%.2f%%+%.2f%%)' % (np.mean(last_ten), 2 * np.mean(last_ten_std))
                     , linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Acc')
    plt.ylim(35, 92)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=13)

# plt.show()

plt.savefig('./init.pdf', bbox_inches='tight')