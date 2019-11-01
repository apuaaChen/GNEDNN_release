import os
import argparse

# Description
parser = argparse.ArgumentParser(description='exps for NN stability')

parser.add_argument('--dataset_dir', default='./cifar10', help='the directory of dataset')
parser.add_argument('--type', choices=['initialization', 'norm', 'selu', 'densenet', 'resnet'],
                    default='initialization')
parser.add_argument('--repeat', type=int, default=1, help='number of time to run each exp')
parser.add_argument('--start', type=int, default=0, help='start number of exp')

parser.add_argument('--gpu', default='0', help='using which GPU')

args = parser.parse_args()

if not os.path.exists('./json'):
    os.mkdir('./json')

# Experiments in Sec. 5.2: Initialization Techniques
initial_exp = {
    'file': 'initialization_main --size=32 --lr=0.01 --epochs=130 --gpu=' + args.gpu,
    'exps': [
        # baseline models
        '--init_fn=kaiming_norm --neuron=relu --norm_fn=bn --list=bn_relu',
        # baseline model for BN with ReLU
        '--init_fn=kaiming_norm --neuron=tanh --norm_fn=bn --list=bn_tanh',
        # baseline model for BN with tanh
        '--init_fn=kaiming_norm --neuron=leaky_relu --norm_fn=bn --list=bn_lrelu --neg_slope=0.18',
        # baseline model for BN with leaky ReLU
        '--init_fn=kaiming_norm --neuron=prelu --norm_fn=bn --list=bn_prelu',
        # baseline model for BN with PReLU

        # For kaiming initialization
        '--init_fn=kaiming_norm --neuron=relu --norm_fn=none --list=km_relu',
        # kaiming normal + ReLU
        '--init_fn=kaiming_norm --neuron=tanh --norm_fn=none --list=km_tanh',
        # kaiming normal + tanh
        '--init_fn=kaiming_norm --neuron=leaky_relu --norm_fn=none --list=km_lrelu --neg_slope=0.18',
        # kaiming normal + leaky ReLU
        '--init_fn=kaiming_norm --neuron=prelu --norm_fn=none --list=km_prelu',
        # kaiming normali + pReLU
        '--init_fn=kaiming_norm --neuron=sprelu --norm_fn=none --list=km_sprelu',
        # kaiming normali + spReLU

        # For orthogonal initialization
        '--init_fn=orthogonal --neuron=relu --norm_fn=none --list=orth_relu',
        # orthogonal normal + ReLU
        '--init_fn=orthogonal --neuron=tanh --norm_fn=none --list=orth_tanh',
        # orthogonal normal + tanh
        '--init_fn=orthogonal --neuron=leaky_relu --norm_fn=none --list=orth_lrelu --neg_slope=0.18',
        # orthogonal normal + leaky ReLU
        '--init_fn=orthogonal --neuron=prelu --norm_fn=none --list=orth_prelu',
        # orthogonal normal + pReLU
        '--init_fn=orthogonal --neuron=sprelu --norm_fn=none --list=orth_sprelu',
        # orthogonal normal + spReLU

        # For leaky ReLU with different negative slopes
        # baseline models
        '--init_fn=kaiming_norm --neuron=leaky_relu --norm_fn=bn --list=bn_lrelu3 --neg_slope=0.3',
        '--init_fn=kaiming_norm --neuron=leaky_relu --norm_fn=bn --list=bn_lrelu5 --neg_slope=0.5',
        # negative slope=0.3
        '--init_fn=orthogonal --neuron=leaky_relu --norm_fn=none --list=orth_lrelu3 --neg_slope=0.3',
        # negative slope=0.5
        '--init_fn=orthogonal --neuron=leaky_relu --norm_fn=none --list=orth_lrelu5 --neg_slope=0.5',
    ]
}


# Experiments in Sec. 5.3: Normalization Techniques
normal_exp = {
    'file': 'normalization_main --lr=0.01 --epochs=130 --gpu=' + args.gpu,
    'exps': [
        # For weight normalization
        '--size=32 --init_fn=kaiming_norm --neuron=relu --norm_fn=wn --conv=naive --list=wn32 --epochs=4',
        # For l2normalization
        '--size=32 --init_fn=kaiming_norm --neuron=relu --norm_fn=l2n --conv=centered --list=l2norm',
        # For weight normalization
        '--size=32 --init_fn=kaiming_norm --neuron=relu --norm_fn=none --conv=normed --list=wn',
        # For l1normalization
        '--size=32 --init_fn=kaiming_norm --neuron=relu --norm_fn=l1n --conv=centered --list=l1norm'
    ]
}

# Experiments in Sec. 5.4: Self-Normalizing Neural Network
selu_exp = {
    'file': 'initialization_main --size=32 --lr=0.01 --epochs=130 --gpu=' + args.gpu,
    'exps': [
        '--init_fn=kaiming_norm --neuron=selu --norm_fn=none --list=km_selu --gain=1.',
        # baseline model for original SeLU

        '--init_fn=kaiming_norm --neuron=seluv2 --norm_fn=none '
        '--fixpoint=1. --gain=1. --epsilon=0.0716 --list=km_seluv2_g1_e7',
        # reproduce the baseline model

        '--init_fn=kaiming_norm --neuron=seluv2 --norm_fn=none '
        '--fixpoint=1. --gain=1. --epsilon=0.03 --list=km_seluv2_g1_e3',
        # epsilon=3

        '--init_fn=kaiming_norm --neuron=seluv2 --norm_fn=none '
        '--fixpoint=1. --gain=1.4142535 --epsilon=0.03 --list=km_seluv2_g2_e3',
        # gain=2, epsilon=3

        # with orthogonal initialization

        '--init_fn=orthogonal --neuron=selu --norm_fn=none --list=orth_selu --gain=1.',
        # baseline model for original SeLU

        '--init_fn=orthogonal --neuron=seluv2 --norm_fn=none '
        '--fixpoint=1. --gain=1. --epsilon=0.0716 --list=orth_seluv2_g1_e7',
        # reproduce the baseline model

        '--init_fn=orthogonal --neuron=seluv2 --norm_fn=none '
        '--fixpoint=1. --gain=1. --epsilon=0.03 --list=orth_seluv2_g1_e3',
        # epsilon=3

        '--init_fn=orthogonal --neuron=seluv2 --norm_fn=none '
        '--fixpoint=1. --gain=1.4142535 --epsilon=0.03 --list=orth_seluv2_g2_e3',
        # epsilon=3

    ]
}

densenet_exp = {
    'file': 'densenet_main --epochs=130 --gpu=' + args.gpu,
    'exps': [
        '--init_fn=kaiming_norm --neuron=relu --norm_fn=bn --list=dn_bn --lr=0.1',
        # A baseline model
        '--init_fn=kaiming_norm --neuron=relu --norm_fn=none --list=dn_km_relu --lr=0.1',
        # A kaiming norm baseline

        '--init_fn=orthogonal --neuron=leaky_relu --norm_fn=none --list=dn_orth_lrelu3 --neg_slope=0.3 --lr=0.1',
        # leaky ReLU + orthogonal initialization + negative slope=0.3

        '--init_fn=kaiming_norm --neuron=relu --norm_fn=l2n --conv=centered --list=dn_l2norm --lr=0.1',
        # second moment normalization

        '--init_fn=kaiming_norm --neuron=relu --norm_fn=none --conv=normed --list=dn_wn --lr=0.1',
        # weight normalization

        '--init_fn=orthogonal --neuron=seluv2 --norm_fn=none '
        '--fixpoint=1. --gain=1.4142535 --epsilon=0.03 --list=dn_orth_seluv2_g2_e3 --lr=0.1',
        # SeLU + orthogonal + epsilon=3 + gamma_0=2
    ]
}

resnet_exp = {
    'file': 'resnet_main --size=56 --epochs=200 --gpu=' + args.gpu,
    'exps': [
        # baseline model
        '--init_fn=kaiming_norm --neuron=relu --norm_fn=bn --list=res_bn --lr=0.1',

        # baseline model + mixup
        '--init_fn=kaiming_norm --neuron=relu --norm_fn=bn --list=res_bn_mixup --lr=0.1 --mixup',


        # fixup init p=2 + scale + mixup
        '--init_fn=kaiming_norm --neuron=relu --norm_fn=none '
        '--list=res_fixup2s_mixup --lr=0.1 --fixup --p=2 --m=2 --mixup --bns',

        # fixup init p=2 + scale
        '--init_fn=kaiming_norm --neuron=relu --norm_fn=none '
        '--list=res_fixup2s --lr=0.1 --fixup --p=2 --m=2 --bns',

        # fixup init p=2
        '--init_fn=kaiming_norm --neuron=relu --norm_fn=none '
        '--list=res_fixup2 --lr=0.1 --fixup --p=2 --m=2',

        # fixup init p=1.5 + scale + mixup
        '--init_fn=kaiming_norm --neuron=relu --norm_fn=none '
        '--list=res_fixup1_5s_mixup --lr=0.1 --fixup --p=1.5 --m=2 --mixup --bns',


        # 2nd moment normalization
        '--init_fn=kaiming_norm --neuron=relu --norm_fn=l2n --conv=centered --list=res_l2norm',

        # 2nd moment normalization + mixup
        '--init_fn=kaiming_norm --neuron=relu --norm_fn=l2n --conv=centered --list=res_l2norm_mixup --mixup',
    ]
}

types = {
    'initialization': initial_exp,
    'norm': normal_exp,
    'selu': selu_exp,
    'densenet': densenet_exp,
    'resnet': resnet_exp,
}


def run():
    exp_dict = types[args.type]
    for i in range(args.repeat):
        for exp in exp_dict['exps']:
            cmd = 'python -m' + ' cifar.' + exp_dict['file'] + ' ' + exp\
                  + ' --log=exp%d' % (i + args.start) + ' --root=' + args.dataset_dir
            os.system(cmd)


run()
