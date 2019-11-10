echo "Data dir: $1"
echo "Exp: $2"
echo "GPU: $3"
echo "repeat: $4"
echo "start: $5"
python -m cifar.exp_list --dataset_dir=$1 --type=$2 --repeat=$4 --start=$5 --gpu=$3