DATADIR = $1
python -m cifar.explist --dataset_dir=$DATADIR --type=initialization --repeat=1 --start=0 --gpu=1