import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_and_tokenizer(model_path):
    """Load the model and tokenizer from the specified path."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    return model, tokenizer

def evaluate_model(model, tokenizer, prompts, max_length=2048):
    """Generate responses from the model for each prompt."""
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}\nResponse: {response}\n")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load and evaluate a language model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model directory.")
    parser.add_argument('--prompts', type=str, nargs='+', required=True, help="Prompts to evaluate the model with.")
    args = parser.parse_args()

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # Evaluate the model with the provided prompts
    evaluate_model(model, tokenizer, args.prompts)

if __name__ == "__main__":
    main()
