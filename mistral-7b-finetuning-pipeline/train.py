import argparse

import pandas as pd
import torch

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from tqdm import tqdm

def print_trainable_parameters(model):
    # Prints the number of trainable parameters in the model.
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

def main(args):
    # Model configurations & tokenizer
    print("*** Loading tokenizer...")
    base_model_name = "davidkim205/komt-mistral-7b-v1"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Tokenize training dataset
    print("*** Tokenizing training dataset...")
    data = pd.read_csv(args.dataset_path)
    max_length = 128

    formatted_data = []
    for _, row in tqdm(data.iterrows()):
        for q_col in ['질문_1', '질문_2']:
            for a_col in ['답변_1', '답변_2', '답변_3', '답변_4', '답변_5']:
                # Concatenating question and answer with EOS token
                input_text = row[q_col] + tokenizer.eos_token + row[a_col]
                input_ids = tokenizer.encode(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
                formatted_data.append(input_ids)

    tokenized_train_dataset = torch.cat(formatted_data, dim=0)
    print("Finished tokenizing training dataset")
    print("Dataset size: ", tokenized_train_dataset.shape)

    # Prepare quantized base model
    print("*** Preparing quantized base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map={"":0},
    )
    base_model.gradient_checkpointing_enable()
    base_model = prepare_model_for_kbit_training(base_model)

    # Build PEFT model
    print("*** Building PEFT model for LoRA adapter...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    ft_model = get_peft_model(base_model, lora_config)
    print_trainable_parameters(ft_model)

    # Set up trainer
    print("*** Setting up trainer...")
    tokenizer.pad_token = tokenizer.eos_token

    trainer = Trainer(
        model=ft_model,
        train_dataset=tokenized_train_dataset,
        args=TrainingArguments(
            output_dir=args.checkpoint_path,
            warmup_steps=1,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            max_steps=args.max_steps,
            learning_rate=2.5e-4, # Want a small lr for finetuning
            #bf16=True,
            optim="paged_adamw_8bit",
            logging_steps=args.logging_steps,  # When to start reporting loss
            logging_dir="./logs",              # Directory for storing logs
            save_strategy="steps",             # Save the model checkpoint every logging step
            save_steps=25,                     # Save checkpoints every 50 steps
            # evaluation_strategy="steps",     # Evaluate the model every logging step
            # eval_steps=25,                   # Evaluate and save checkpoints every 50 steps
            # do_eval=True,                    # Perform evaluation at the end of training
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    ft_model.config.use_cache = False # silence the warnings. Please re-enable for inference!

    # Train the model
    print("*** Training the model...")
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="/root/dataset/train.csv")
    parser.add_argument("--checkpoint-path", type=str, default="/root/checkpoint")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=25)

    args = parser.parse_args()
    main(args)
