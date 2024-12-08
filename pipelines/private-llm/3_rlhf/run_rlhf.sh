if [ -z "$1" ]; then
    model_path="/root/model"
else
    model_path="$1"
fi

# Common parameters for all methods
COMMON_ARGS="
    --model_name_or_path $model_path \
    --reward_model_path /root/reward_model \
    --trust_remote_code True \
    --attn_implementation flash_attention_2 \
    --dataset_name /root/dataset \
    --warmup_ratio 0.1 \
    --learning_rate $LEARNING_RATE \
    --logging_steps 25 \
    --output_dir /root/training_results \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --bf16 \
    --torch_dtype bfloat16 \
    --total_episodes $TOTAL_EPISODES \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --response_length 64 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 20 \
    --eval_strategy no \
    --max_length 2048"

# Method-specific parameters
case "$PEFT_METHOD" in
    "Full")
        METHOD_ARGS="--gradient_checkpointing True"
        ;;
    "LoRA")
        METHOD_ARGS="
        --gradient_checkpointing False \
        --use_peft True \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_r 8 \
        --lora_target_modules all-linear \
        --lora_task_type CAUSAL_LM"
        ;;
    "QLoRA")
        METHOD_ARGS="
        --gradient_checkpointing False \
        --load_in_4bit True \
        --bnb_4bit_quant_type nf4 \
        --use_bnb_nested_quant True \
        --use_peft True \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_r 8 \
        --lora_target_modules all-linear \
        --lora_task_type CAUSAL_LM"
        ;;
    *)
        echo "Invalid PEFT_METHOD: $PEFT_METHOD"
        exit 1
        ;;
esac

# Execute the command
accelerate launch rlhf.py $COMMON_ARGS $METHOD_ARGS
