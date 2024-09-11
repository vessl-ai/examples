import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from arguments import DatasetArguments, ModelArguments, PeftArguments


def set_attn():
    try:
        import flash_attn

        print("Using flash attention...")
        return "flash_attention_2"
    except Exception:
        return "eager"


def load_model_and_tokenizer(model_args: ModelArguments, data_args: DatasetArguments):
    quantization_config = None

    if model_args.load_in_4bit:
        compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
        store_dtype = getattr(torch, model_args.bnb_4bit_quant_storage)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.bnb_4bit_use_double_quant,
            bnb_4bit_quant_storage=store_dtype,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=model_args.load_in_8bit)

    if model_args.use_unsloth:
        from unsloth import FastLanguageModel

        model, _ = FastLanguageModel.from_pretrained(
            model_name=model_args.model_name_or_path,
            max_seq_length=data_args.max_seq_length,
            dtype=None,
            load_in_4bit=model_args.load_in_4bit,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=quantization_config,
            trust_remote_code=True,
            attn_implementation=set_attn(),
            device_map="auto",
            torch_dtype="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_peft_config(peft_args: PeftArguments):
    from peft import LoraConfig

    target_modules = (
        peft_args.lora_target_modules[0]
        if peft_args.lora_target_modules[0] == "all-linear"
        else peft_args.lora_target_modules
    )

    peft_config = LoraConfig(
        task_type=peft_args.task_type,
        r=peft_args.lora_r,
        lora_alpha=peft_args.lora_alpha,
        bias=peft_args.lora_bias,
        lora_dropout=peft_args.lora_dropout,
        target_modules=target_modules,
    )

    return peft_config


def get_unsloth_peft_model(
    model,
    peft_args: PeftArguments,
    training_args: TrainingArguments,
    data_args: DatasetArguments,
):
    from unsloth import FastLanguageModel

    target_modules = (
        peft_args.lora_target_modules[0]
        if peft_args.lora_target_modules[0] == "all-linear"
        else peft_args.lora_target_modules
    )

    model = FastLanguageModel.get_peft_model(
        model,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        r=peft_args.lora_r,
        target_modules=target_modules,
        use_gradient_checkpointing=training_args.gradient_checkpointing,
        max_seq_length=data_args.max_seq_length,
    )

    return model
