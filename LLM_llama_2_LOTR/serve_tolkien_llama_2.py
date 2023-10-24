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

import time
import typing as t
from typing import TYPE_CHECKING

import bentoml
from bentoml.io import JSON
from bentoml.io import Text


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


class Llama2(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        model_name = "/ckpt/llama_2_7b_hf"
        model_diff_name = "/ckpt_diff/llm_tolkien_llama_2_7B_local"
        lora_config = {
            "task_type": "CAUSAL_LM",
            "r": 16, # attention heads
            "lora_alpha": 32, # alpha scaling
            "lora_dropout": 0.05,
            "bias": "none",
        }
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, load_in_8bit=True)
        model = prepare_model(model)
        model = get_peft_model(model, LoraConfig(**lora_config))
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        tokenizer.pad_token='[PAD]'
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        config = PeftConfig.from_pretrained(model_diff_name, local_files_only=True)
        trained_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
        self.trained_model = trained_model
        # Load the Lora model
        trained_model = PeftModel.from_pretrained(self.trained_model, model_diff_name, local_files_only=True)
        self.tokenizer = tokenizer
        self.trained_model = trained_model

    @bentoml.Runnable.method(batchable=False)
    def generate(self, input_text: str) -> bool:
        inputs = self.tokenizer(input_text, return_tensors="pt")
        tokens = self.trained_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.75,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        result = self.tokenizer.decode(tokens[0])
        return result
    
llama2_runner = t.cast(
    "RunnerImpl", bentoml.Runner(Llama2, name="llama2")
)

svc = bentoml.Service('ROTR_serve_llama2', runners=[llama2_runner])
@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON())
async def infer(text: str) -> str:
    result = await llama2_runner.generate.async_run(text)
    return result
