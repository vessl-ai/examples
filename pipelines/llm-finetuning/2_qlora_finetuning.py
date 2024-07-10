import os

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

MAX_STEPS = int(os.environ.get("MAX_STEPS", 300))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")


def main():
    dataset = load_dataset("vessl/insurance-policies", split="train")

    base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=2048,  # Supports RoPE Scaling internally, so choose any!
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        base_model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=8,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",     # Supports any, but = "none" is optimized
        use_gradient_checkpointing=True,
    )

    # Hyperparamters
    training_arguments = TrainingArguments(
        output_dir="./output",                              # Directory for storing output files
        logging_dir="./logs",                               # Directory for storing logs
        per_device_train_batch_size=24,                     # Number of samples per batch per device during training
        num_train_epochs=2,                                 # Total number of training epochs to perform
        max_steps=MAX_STEPS,                                # Maximum number of steps to train the model, overrides `num_train_epochs`
        gradient_accumulation_steps=1,                      # Number of updates steps to accumulate before performing a backward/update pass
        warmup_steps=1,                                     # Linear warmup over warmup_steps
        learning_rate=2.5e-4,                               # Learning rate for the optimizer
        optim="paged_adamw_8bit",                           # The optimizer to use
        gradient_checkpointing=True,                        # If True, use gradient checkpointing to save memory at the expense of slower backward pass
        fp16=not torch.cuda.is_bf16_supported(),            # Whether to use 16-bit (mixed) precision training instead of 32-bit training
        bf16=torch.cuda.is_bf16_supported(),                # Whether to use bfloat16 precision training instead of 32-bit training
        logging_steps=25,                                   # Log every X updates steps
        save_steps=50,                                      # Save checkpoint every X updates steps
        save_strategy="steps",                              # The checkpoint saving strategy to adopt during training
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Setting sft parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1536,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    trainer.train()

    # Save the model
    new_model_name = "kb-insurance-policies-7b-qlora"
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, new_model_name))
    model.config.use_cache = True


if __name__ == "__main__":
    main()
