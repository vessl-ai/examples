peft_type=${PEFT_METHOD,,}

python merge.py --model-name-or-path /root/model \
    --adapter-name-or-path /root/training_results \
    --output-dir /root/results/rlhf_model \
    --peft $peft_type
