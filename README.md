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
python -m imagenet.mobilenet_main imagenet_dir -b=512 --gpus='0,1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:3003' --rank=0 --world-size=1 -j=16 --epochs=90 --log=mobilenet_bn_dense --norm_fn='bn' --epochs=90 --lr=0.02 --dense
```
For ReLU + Gaussian initialization
```
python -m imagenet.mobilenet_main imagenet_dir -b=512 --gpus='0,1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:2003' --rank=0 --world-size=1 -j=16 --epochs=90 --log=mobilenet_relu_dense --init_fn='kaiming_norm' --norm_fn='none' --neuron='relu' --epochs=90 --lr=0.02 --dense
```
For SeLU baseline
```
python -m imagenet.mobilenet_main imagenet_dir -b=512 --gpus='0,1' --multiprocessing-distributed --dist-url='tcp://127.0.0.1:2003' --rank=0 --world-size=1 -j=16 --epochs=90 --log=mobilenet_selu_dense --init_fn='kaiming_norm' --norm_fn='none' --neuron='selu' --epochs=90 --lr=0.02 --dense
```
For SeLU $\epsilon=0.065$
