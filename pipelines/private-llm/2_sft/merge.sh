peft_type=${PEFT_METHOD,,}

python merge.py --model-name-or-path /root/model \
    --adapter-name-or-path /root/results/sft_intermediate \
    --output-dir /root/results/sft_model \
    --peft $peft_type

# cleanup intermediate results
rm -rf /root/results/sft_intermediate
