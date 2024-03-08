# %%
import torch
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    MixtralForCausalLM,
)


# %%
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = MixtralForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb_config,
)

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="right")
tokenizer.pad_token = tokenizer.eos_token

# %%
instruction = "How can I create a new model via web?"
inputs = tokenizer.apply_chat_template([{"role": "user", "content": instruction}], return_tensors="pt").to(
    model.device
)

# %%
generation_config = GenerationConfig(
    max_length=512,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)

# %%
LORA_PATH = "/lora/checkpoint-100"
QLORA_PATH = "/qlora/checkpoint-100"

# %%
input_len = inputs.shape[1]

outputs = model.generate(inputs, generation_config=generation_config)
print("==BASE MODEL==")
print(tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True))

lora_model = PeftModel.from_pretrained(model, "LORA_PATH")
lora_output = lora_model.generate(inputs, generation_config=generation_config)
print("==LORA MODEL==")
print(tokenizer.decode(lora_output[0][input_len:], skip_special_tokens=True))

qlora_model = PeftModel.from_pretrained(model, "QLORA_PATH")
qlora_output = qlora_model.generate(inputs, generation_config=generation_config)
print("==QLORA MODEL==")
print(tokenizer.decode(qlora_output[0][input_len:], skip_special_tokens=True))
# %%
