#### Code for "A Comprehensive and Modularized Statistical Framework for Gradient Norm Equality in Deep Neural Network"
***
##### CIFAR-10

All the experiments on CIFAR-10 are summarized in cifar/exp_list.py. To run each segment of experiments, we provide ```cifar10.sh```. For instance, to run the experiments in initialization:
```
bash cifar10.sh cifar10_dir initialization 1 4 0
```
```cifar10_dir ``` is the directory to the dataset. ```initialization``` indecates the segment to run. ```1``` suggests that the experiment is running on gpu 1. ```4``` means that each experiment is run for 4 times. ```0``` suggests that the experiments' index starts from ```0```.
***
##### ImageNet

**MobileNet**

For BN baseline:
```
python -m imagenet.mobilenet_main imagenet_dir -b=512 --gpus='0,1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:2003' --rank=0 --world-size=1 -j=16 --epochs=90 --log=mobilenet_bn_dense --norm_fn='bn' --epochs=90 --lr=0.02 --dense
```
For ReLU + Gaussian initialization:
```
python -m imagenet.mobilenet_main imagenet_dir -b=512 --gpus='0,1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:2003' --rank=0 --world-size=1 -j=16 --epochs=90 --log=mobilenet_relu_dense --init_fn='kaiming_norm' --norm_fn='none' --neuron='relu' --epochs=90 --lr=0.02 --dense
```
For SeLU baseline:
```
python -m imagenet.mobilenet_main imagenet_dir -b=512 --gpus='0,1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:2003' --rank=0 --world-size=1 -j=16 --epochs=90 --log=mobilenet_selu_dense --init_fn='kaiming_norm' --norm_fn='none' --neuron='selu' --epochs=90 --lr=0.02 --dense
```
For SeLU epsilon=0.06:
```
python -m imagenet.mobilenet_main imagenet_dir -b=512 --gpus='0,1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:2003' --rank=0 --world-size=1 -j=16 --epochs=90 --log=mobilenet_seluv2_0.06 --init_fn='kaiming_norm' --norm_fn='none' --neuron='seluv2' --selu --fixpoint=1. --epsilon=0.065 --gain=1. --epochs=90 --lr=0.02 --dense
```
For SeLU epsilon=0.03:
```
python -m imagenet.mobilenet_main imagenet_dir -b=512 --gpus='0,1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:2003' --rank=0 --world-size=1 -j=16 --epochs=90 --log=mobilenet_seluv2_0.03 --init_fn='kaiming_norm' --norm_fn='none' --neuron='seluv2' --selu --fixpoint=1. --epsilon=0.03 --gain=1. --epochs=90 --lr=0.02 --dense
```
For leaky ReLU + Gaussian initialization:
```
python -m imagenet.mobilenet_main imagenet_dir -b=512 --gpus='0,1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:2003' --rank=0 --world-size=1 -j=16 --epochs=90 --log=mobilenet_lrelu_dense --init_fn='kaiming_norm' --norm_fn='none' --neuron='leaky_relu' --neg_slope=0.3 --epochs=90 --lr=0.02 --dense
```
For leaky ReLU + Orthogonal initialization:
```
python -m imagenet.mobilenet_main imagenet_dir -b=512 --gpus='0,1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:2003' --rank=0 --world-size=1 -j=16 --epochs=90 --log=mobilenet_lrelu_dense_orth --init_fn='orthogonal' --norm_fn='none' --neuron='leaky_relu' --neg_slope=0.3 --epochs=90 --lr=0.02 --dense
```
For leaky ReLU + Delta Orthogonal initialization:
```
python -m imagenet.mobilenet_main imagenet_dir -b=512 --gpus='0,1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:2003' --rank=0 --world-size=1 -j=16 --epochs=90 --log=mobilenet_lrelu_dense_dorth --init_fn='delta_orth' --norm_fn='none' --neuron='leaky_relu' --neg_slope=0.3 --epochs=90 --lr=0.02 --dense
```

**ResNet 50**

For BN baseline:
```
python -m imagenet.resnet50_main imagenet_dir -a=resnet50 -b=256 --gpus='0,1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:5005' --rank=0 --world-size=1 -j=8 --epochs=90 --log=Resnet50_baseline
```
For BN + mixup:
```
python -m imagenet.resnet50_mixup_main imagenet_dir -a=resnet50 -b=256 --gpus='0,1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:5005' --rank=0 --world-size=1 -j=8 --epochs=90 --log=Resnet50_mixup --alpha=0.2
```
For Second moment normalization
```
python -m imagenet.resnet50_mn_main imagenet_dir -b=256 --gpus='0,1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:5005' --rank=0 --world-size=1 -j=8 --epoch=90 --log=resnet50_smn
```
For Second moment normalization + mixup
```
python -m imagenet.resnet50_mn_main imagenet_dir -b=256 --gpus='0,1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:5005' --rank=0 --world-size=1 -j=8 --epoch=90 --log=resnet50_smn_mixup_0.2 --alpha=0.2
```
For L1-MN + mixup
```
python -m imagenet.resnet50_mn_main imagenet_dir -b=256 --gpus='0,1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:5005' --rank=0 --world-size=1 -j=8 --epoch=90 --log=resnet50_l1mn_mixup_0.2 --alpha=0.2 --l1
```
For Original Fixup initialization
```
python -m imagenet.fixup_main imagenet_dir -b=256 --gpus='0, 1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:5005' --rank=0 --world-size=1 -j=8 --epochs=100 --log=ResNet50_fixup
```
For our Fixup initialization
```
python -m imagenet.fixup_main imagenet_dir -b=256 --gpus='0, 1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:5005' --rank=0 --world-size=1 -j=8 --epochs=100 --log=ResNet50_fixup_ours --ours
```