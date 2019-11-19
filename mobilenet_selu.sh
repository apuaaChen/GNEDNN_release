python -m imagenet.mobilenet_main /raid/datasets/ImageNet2012/ -b=512 --gpus='2,3' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:2003' --rank=0 --world-size=1 -j=16 --epochs=90 --log=$1 --init_fn='kaiming_norm' --norm_fn='none' --neuron='seluv2' --selu --fixpoint=1. --epsilon=$2 --gain=1. --epochs=90 --lr=0.02 --dense