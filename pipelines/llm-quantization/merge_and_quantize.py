import torch
import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM

def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description="Merge and quantize a fine-tuned language model.")
    parser.add_argument('--base-model-name', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Base model name.")
    parser.add_argument('--adapter-name', type=str, default="JoPmt/llama-3.1-adapter", help="Adapter model name or path.")
    parser.add_argument('--merged-model-name', type=str, default="./model-merged", help="Output path for the merged model.")
    parser.add_argument('--quantized-model-name', type=str, default="./output", help="Output path for the quantized model.")
    parser.add_argument('--use-flash-attn', dest='use_flash_attn', action='store_true', help="Use flash attention if available.")
    parser.add_argument('--no-use-flash-attn', dest='use_flash_attn', action='store_false', help="Do not use flash attention.")
    parser.set_defaults(use_flash_attn=True)
    args = parser.parse_args()

    # Load the fine-tuned model and merge it
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        return_dict=True,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.use_flash_attn else None,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_name)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.merged_model_name)

    # Load merged model for quantization
    model = AutoAWQForCausalLM.from_pretrained(
        args.merged_model_name, device_map="cuda", **{"low_cpu_mem_usage": True, "use_cache": False}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)

    # Quantize the merged model
    quant_config = {"zero_point": False, "q_group_size": 128, "w_bit": 4, "version": "Marlin"}
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(args.quantized_model_name)
    tokenizer.save_pretrained(args.quantized_model_name)

    print(f'Model is quantized and saved at "{args.quantized_model_name}"')

if __name__ == "__main__":
    main()
