import argparse

import torch
import vessl
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer

# keys which are in summary logs
SUMMARY_KEYS = {
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    "train_loss",
    "total_flos",
}


# TODO(sanghyuk): replace this with vessl.VesslCallback as soon as it is fixed
class VesslLogCallback(TrainerCallback):
    def on_log(
        self, args, state, control, logs: dict[str, float] = None, **kwargs
    ) -> None:
        # log metrics in the main process only
        if state.is_world_process_zero:
            # if the current step is the max step, and the summary keys are subset of log keys, just skip logging.
            # It is summary log and it will be handled in on_train_end logs
            log_keys = set(logs.keys())
            if state.max_steps != state.global_step or not SUMMARY_KEYS.issubset(
                log_keys
            ):
                vessl.log(payload=logs, step=state.global_step)


def main(
    dataset_name: str, output_dir: str, batch_size: int, max_seq_length: int = 512
):
    model_name = "meta-llama/Meta-Llama-3-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
            "lm_head",
        ],
        bias="none",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )

    dataset = load_dataset(dataset_name)
    dataset = dataset.filter(lambda example: len(example["prompt"]) <= max_seq_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        num_train_epochs=3,
        learning_rate=2e-4,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=10,
        eval_steps=10,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="prompt",
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        packing=True,
        callbacks=[VesslLogCallback],
    )

    print("Training...")
    trainer.train()

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="the name of dataset to train on")
    parser.add_argument("--output-dir", help="output directory of training")
    parser.add_argument("--batch-size", help="batch size", type=int)
    parser.add_argument(
        "--max-seq-length", help="max sequance length per each sample", type=int
    )
    args = parser.parse_args()

    main(args.dataset, args.output_dir, args.batch_size, args.max_seq_length)
