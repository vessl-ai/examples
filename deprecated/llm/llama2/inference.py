import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "/ckpt/"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, device_map="auto", torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, peft_model_id, device_map="auto")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, legacy=False)
tokenizer.pad_token = tokenizer.unk_token
target_max_length = 128
tokenized = tokenizer("There was a hobbit named David.")
tokenized["input_ids"] = torch.tensor(tokenized["input_ids"]).unsqueeze(0)
tokenized["attention_mask"] = torch.ones(tokenized["input_ids"].size(1)).unsqueeze(0)
outputs = model.generate(
    input_ids=tokenized["input_ids"],
    max_new_tokens=1024,
    attention_mask=tokenized["attention_mask"],
)
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
with open("/output/out.txt", "w") as f:
    f.write(result)
