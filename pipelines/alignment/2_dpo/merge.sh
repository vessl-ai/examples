peft_type=${FINETUNING_METHOD,,}

python merge.py --model-name-or-path /root/model \
    --adapter-name-or-path /root/training_results \
    --output-dir /root/results \
    --peft $peft_type
