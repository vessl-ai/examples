import json
from pathlib import Path
from typing import Callable, Mapping

from datasets import Dataset
from peft import get_peft_config, get_peft_model, LoraConfig, PeftConfig, PeftModel
from torch import float32, nn, exp
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


# dataset codes
# https://github.com/jeremyarancio/llm-tolkien/blob/main/llm/prepare_dataset.py

def prepare_dataset(
    model_name: str,
    dataset_path: Path,
    min_length: int,
    context_length: int,
    test_size: float,
    shuffle: bool,
) -> None:
    """Prepare dataset for training and push it to the hub."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = preprocess_data(
        dataset_path=dataset_path, min_length=min_length, tokenizer=tokenizer
    )
    dataset = Dataset.from_dict({"text": [text]})

    # tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "context_length": context_length},
        remove_columns=dataset.column_names,
    )
    tokenized_dataset_dict = tokenized_dataset.train_test_split(
        test_size=test_size, shuffle=shuffle
    )

    return tokenized_dataset_dict

def preprocess_data(
    dataset_path: Path, min_length: int, tokenizer: PreTrainedTokenizer
) -> str:
    """Prepare dataset for training from the jsonl file.

    Args:
        dataset_path (Path): Extracted text from the book
        min_length (int): Filter pages without text
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer

    Yields:
        str: text of the pages
    """
    with open(dataset_path, "r") as f:
        grouped_text = ""
        for line in f:
            elt = json.loads(line)
            text: str = list(elt.values())[0]
            if len(text) > min_length:
                grouped_text += text
        # End of paragraphs defined by ".\n is transformed into EOS token"
        grouped_text = grouped_text.replace(".\n", "." + tokenizer.eos_token)
        return preprocess_text(grouped_text)


def preprocess_text(text: str) -> str:
    text = text.replace("\n", " ")
    return text


def tokenize(element: Mapping, tokenizer: Callable, context_length: int) -> str:
    inputs = tokenizer(
        element["text"],
        truncation=True,
        return_overflowing_tokens=True,
        return_length=True,
        max_length=context_length,
    )
    inputs_batch = []
    for length, input_ids in zip(inputs["length"], inputs["input_ids"]):
        if (
            length == context_length
        ):  # We drop the last input_ids that are shorter than max_length
            inputs_batch.append(input_ids)
    return {"input_ids": inputs_batch}


# training_util

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(float32)


def prepare_model(model):
    for param in model.parameters():
      param.requires_grad = False  # freeze the model - train adapters later
      if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(float32)
    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)
    return model


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"


model_name = "/ckpt/llama_2_7b_hf"

# load dataset
dataset_dict = prepare_dataset(
    model_name=model_name,
    dataset_path=Path("./extracted_text.jsonl"),
    min_length=100,
    context_length = 1024,
    test_size = 0.1,
    shuffle = True,
)

lora_config = {
    "task_type": "CAUSAL_LM",
    "r": 16, # attention heads
    "lora_alpha": 32, # alpha scaling
    "lora_dropout": 0.05,
    "bias": "none",
}

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
model = prepare_model(model)
model = get_peft_model(model, LoraConfig(**lora_config))

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token='[PAD]'
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


trainer_config = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "warmup_steps": 100,
    "weight_decay": 0.1,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "fp16": True,
    "logging_steps": 1,
    "output_dir": "/output",
    "overwrite_output_dir": True,
    "evaluation_strategy": "no",
    "save_strategy": "no",
}

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    
training_args = TrainingArguments(**trainer_config)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    data_collator=data_collator,
)

model.config.use_cache = False
trainer.train()
model.config.use_cache = True
model.save_pretrained("/ckpt/llm_tolkien_llama_2_7B_local")

config = PeftConfig.from_pretrained("/ckpt/llm_tolkien_llama_2_7B_local")
trained_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
# tokenizer = AutoTokenizer.from_pretrained("JeremyArancio/llm-tolkien")
# Load the Lora model
trained_model = PeftModel.from_pretrained(trained_model, "/ckpt/llm_tolkien_llama_2_7B_local")

# Generate text 1
prompt = 'The hobbits were so suprised seeing their friend'
inputs = tokenizer(prompt, return_tensors="pt")
tokens = trained_model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.75,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
print('prompt: The hobbits were so suprised seeing their friend')
print(tokenizer.decode(tokens[0]))

# Generate text 2
prompt = 'Aragorn picked up the sword, and'
inputs = tokenizer(prompt, return_tensors="pt")
tokens = trained_model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.75,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
print('prompt: Aragorn picked up the sword, and')
print(tokenizer.decode(tokens[0]))

# Generate text 3
prompt = 'The orks are gathering to Rohan, and the hobbits are'
inputs = tokenizer(prompt, return_tensors="pt")
tokens = trained_model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.75,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
print('prompt: The orks are gathering to Rohan, and the hobbits are')
print(tokenizer.decode(tokens[0]))
