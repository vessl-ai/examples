#/bin/bash
org=$1
vessl_project_name=$2
exp_name=$3

pip install -r requirements.txt
apt update
DEBIAN_FRONTEND=noninteractive apt -y install ffmpeg
python preprocessing/preprocess.py --exp_name $exp_name --infer
python run.py --config training/config.yaml --exp_name $exp_name --org $org --vessl_project_name $vessl_project_name --infer --infer_ckpt_epoch 1000