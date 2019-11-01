echo "Data dir: $1"
echo "Exp: $2"
python -m cifar.exp_list --dataset_dir=$1 --type=$2 --repeat=1 --start=0 --gpu=1