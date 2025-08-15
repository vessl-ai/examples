import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_model(
    base_model_name: str,
    lora_adapter_path: str,
    output_path: str,
):
    """
    Merge LoRA adapter with base model and save the merged model.

    Args:
        base_model_name: Name or path of the base model
        lora_adapter_path: Path to the LoRA adapter
        output_path: Path to save the merged model
    """
    print(f"Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print(f"Loading base model: {base_model_name}")
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype="auto",
        use_cache=True,
        device_map="auto"
    )
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)

    print(f"Loading LoRA adapter from: {lora_adapter_path}")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print("Merging LoRA adapter with base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    # Save merged model and tokenizer
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

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

    args = parser.parse_args()

    merge_lora_model(
        base_model_name=args.base_model,
        lora_adapter_path=args.lora_adapter,
        output_path=args.output_path,
    )
