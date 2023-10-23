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


model_name = "/ckpt/llama_2_7b_hf"
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

config = PeftConfig.from_pretrained("/ckpt_diff/llm_tolkien_llama_2_7B_local")
trained_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
# tokenizer = AutoTokenizer.from_pretrained("JeremyArancio/llm-tolkien")
# Load the Lora model
trained_model = PeftModel.from_pretrained(trained_model, "/ckpt_diff/llm_tolkien_llama_2_7B_local")


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
