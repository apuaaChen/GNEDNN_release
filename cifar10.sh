echo "Data dir: $1"
python -m cifar.exp_list --dataset_dir=$1 --type=initialization --repeat=1 --start=0 --gpu=1