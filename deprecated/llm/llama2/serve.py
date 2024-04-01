from __future__ import annotations

import typing as t

import bentoml
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class Llama2(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        peft_model_id = "/ckpt/"
        max_memory = {0: "80GIB", 1: "80GIB", "cpu": "30GB"}
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, device_map="auto", torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(
            model, peft_model_id, device_map="auto", max_memory=max_memory
        )
        model.eval()
        self.model = model
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path, legacy=False
        )
        tokenizer.pad_token = tokenizer.unk_token
        self.target_max_length = 128
        self.tokenizer = tokenizer

    @bentoml.Runnable.method(batchable=False)
    def generate(self, input_text: str) -> bool:
        tokenized = self.tokenizer(input_text)
        tokenized["input_ids"] = (
            torch.tensor(tokenized["input_ids"]).unsqueeze(0).to("cuda")
        )
        tokenized["attention_mask"] = (
            torch.ones(tokenized["input_ids"].size(1)).unsqueeze(0).to("cuda")
        )
        outputs = self.model.generate(
            input_ids=tokenized["input_ids"],
            max_new_tokens=1024,
            attention_mask=tokenized["attention_mask"],
        )
        result = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return result


llama2_runner = t.cast("RunnerImpl", bentoml.Runner(Llama2, name="llama2"))

svc = bentoml.Service("serve_llama2", runners=[llama2_runner])


@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON())
async def infer(text: str) -> str:
    result = await llama2_runner.generate.async_run(text)
    return result
