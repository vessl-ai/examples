import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_model(
    base_model_name: str,
    lora_adapter_path: str,
    output_path: str,
    torch_dtype: str = "bfloat16",
    load_in_4bit: bool = True,
):
    """
    Merge LoRA adapter with base model and save the merged model.
    
    Args:
        base_model_name: Name or path of the base model
        lora_adapter_path: Path to the LoRA adapter
        output_path: Path to save the merged model
        torch_dtype: Torch dtype for model loading
        load_in_4bit: Whether to load model in 4-bit precision
    """
    print(f"Loading base model: {base_model_name}")
    
    # Convert string dtype to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )
    
    print(f"Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, 
        trust_remote_code=True
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading LoRA adapter from: {lora_adapter_path}")
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    
    print("Merging LoRA adapter with base model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # Save merged model and tokenizer
    merged_model.save_pretrained(
        output_path, 
        safe_serialization=True,
        max_shard_size="5GB"
    )
    tokenizer.save_pretrained(output_path)
    
    # Save model configuration info
    with open(os.path.join(output_path, "merge_info.txt"), "w") as f:
        f.write(f"Base model: {base_model_name}\n")
        f.write(f"LoRA adapter: {lora_adapter_path}\n")
        f.write(f"Torch dtype: {torch_dtype}\n")
        f.write(f"Load in 4bit: {load_in_4bit}\n")
    
    print("✅ Model merge completed successfully!")
    print(f"Merged model saved at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--base_model", 
        type=str, 
        default="openai/gpt-oss-20b",
        help="Base model name or path"
    )
    parser.add_argument(
        "--lora_adapter", 
        type=str, 
        required=True,
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="/merged_model",
        help="Output path for merged model"
    )
    parser.add_argument(
        "--torch_dtype", 
        type=str, 
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model loading"
    )
    parser.add_argument(
        "--load_in_4bit", 
        action="store_true",
        default=True,
        help="Load model in 4-bit precision"
    )
    
    args = parser.parse_args()
    
    merge_lora_model(
        base_model_name=args.base_model,
        lora_adapter_path=args.lora_adapter,
        output_path=args.output_path,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
    )