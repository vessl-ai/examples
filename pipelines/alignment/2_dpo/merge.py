import argparse
import copy
import json
import os

import bitsandbytes as bnb
import torch
from bitsandbytes.functional import dequantize_4bit
from peft import PeftModel
from peft.utils import _get_submodules
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def save_model(model, tokenizer, output_dir):
    print(f"Saving dequantized model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "config.json"), "r") as fp:
        config_data = json.load(fp)

    config_data.pop("quantization_config", None)
    config_data.pop("pretraining_tp", None)

    with open(os.path.join(output_dir, "config.json"), "w") as config:
        config.write(json.dumps(config_data, indent=2))


def dequantize_model(
    model,
    tokenizer,
    dtype=torch.bfloat16,
    device="cpu",
):
    BNBLinear4bit = bnb.nn.Linear4bit

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, BNBLinear4bit):
                quant_state = copy.deepcopy(module.weight.quant_state)

                quant_state[2] = dtype

                weights = dequantize_4bit(
                    module.weight.data, quant_state=quant_state, quant_type="nf4"
                ).to(dtype)

                new_module = torch.nn.Linear(
                    module.in_features, module.out_features, bias=None, dtype=dtype
                )
                new_module.weight = torch.nn.Parameter(weights)
                new_module.to(device=device, dtype=dtype)

                parent, target, target_name = _get_submodules(model, name)
                setattr(parent, target_name, new_module)

        model.is_loaded_in_4bit = False

        return model


def main(model_name_or_path, adapter_name_or_path, output_dir, peft_type):
    if peft_type == "qlora":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quant_config = None

    print(f"Starting to load the model {model_name_or_path} into memory")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if peft_type == "qlora":
        model = dequantize_model(model, tokenizer)

    model = PeftModel.from_pretrained(model=model, model_id=adapter_name_or_path)
    model = model.merge_and_unload()

    save_model(model, tokenizer, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--adapter-name-or-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--peft", type=str, choices=("lora", "qlora"), default="lora")
    args = parser.parse_args()

    main(
        model_name_or_path=args.model_name_or_path,
        adapter_name_or_path=args.adapter_name_or_path,
        output_dir=args.output,
        peft_type=args.peft,
    )
