from awq import AutoAWQForCausalLM

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

def load_model(model_name, quantization_method):
    """Load the model and tokenizer from the specified path."""
    if quantization_method == "awq":
        model = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False)
    elif quantization_method == "none":
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported quantization method: {quantization_method}, supported methods are 'awq' and 'none'")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    return model, tokenizer, streamer

def evaluate_model(model, tokenizer, streamer, prompts, max_length=2048):
    """Generate responses from the model for each prompt."""
    for prompt in prompts:
        prompt_template = "[INST] {prompt} [/INST]"
        tokens = tokenizer(prompt_template.format(prompt=prompt),return_tensors='pt').input_ids.cuda()
        generation_output = model.generate(
            tokens,
            streamer=streamer,
            max_new_tokens=512
        )
        print(generation_output)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load and evaluate a language model.")
    parser.add_argument('--model-name', type=str, required=True, help="Name of the model to load from Hub")
    parser.add_argument('--prompts', type=str, nargs='+', required=True, help="Prompts to evaluate the model with.")
    parser.add_argument('--quantization', type=str, default="none", help="Quantization method to use.")
    args = parser.parse_args()

    # Load the model and tokenizer
    model, tokenizer, streamer = load_model(args.model_name, args.quantization)

    # Evaluate the model with the provided prompts
    evaluate_model(model, tokenizer, streamer, args.prompts)

if __name__ == "__main__":
    main()
