if [ "$FINETUNING_METHOD" == "Full" ]; then
    accelerate launch dpo.py \
    --model_name_or_path /root/model \
    --attn_implementation flash_attention_2 \
    --dataset_name /root/dataset \
    --loss_type sigmoid \
    --num_train_epochs 1 \
    --logging_steps 25 \
    --output_dir /root/training_results \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_r 256 \
    --lora_target_modules all-linear \
    --lora_task_type CausalLM \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 20
elif [ "$FINETUNING_METHOD" == "LoRA" ]; then
    accelerate launch dpo.py \
    --model_name_or_path /root/model \
    --attn_implementation flash_attention_2 \
    --dataset_name /root/dataset \
    --loss_type sigmoid \
    --num_train_epochs 1 \
    --logging_steps 25 \
    --output_dir /root/training_results \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_r 256 \
    --lora_target_modules all-linear \
    --lora_task_type CausalLM \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 20
elif [ "$FINETUNING_METHOD" = "QLoRA" ]; then
    accelerate launch dpo.py \
    --model_name_or_path /root/model \
    --attn_implementation flash_attention_2 \
    --dataset_name /root/dataset \
    --loss_type sigmoid \
    --num_train_epochs 1 \
    --logging_steps 25 \
    --output_dir /root/training_results \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --load_in_4bit True \
    --bnb_4bit_quant_type nf4 \
    --use_bnb_nested_quant True \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_r 256 \
    --lora_target_modules all-linear \
    --lora_task_type CausalLM \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 20
else
echo "Invalid FINETUNING_METHOD: $FINETUNING_METHOD"
    exit 1
fi
