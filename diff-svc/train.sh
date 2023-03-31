#/bin/bash
org=$1
vessl_project_name=$2
exp_name=$3
nproc_per_node=$4
max_epoch=$5

echo $org
echo $exp_name
echo $nproc_per_node
python preprocessing/preprocess.py --exp_name $exp_name
python preprocessing/binarize.py --config config.yaml --exp_name $exp_name
python -m torch.distributed.launch --nproc_per_node $nproc_per_node run.py --config training/config.yaml --exp_name $exp_name --org $org --vessl_project_name $vessl_project_name --log_interval 100 --max_epoch $max_epoch --reset