import argparse
import os
from threading import Thread
from time import sleep

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
import torch

import gradio as gr

class InferenceApp:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate(self, query, history):
        print(history)
        prompt = "<s>[INST] " + query + "[/INST]"
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=2048,
            temperature=0.9,
            top_k=1,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=2,
            num_return_sequences=1
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        for new_text in streamer:
            yield generated_text + new_text
            generated_text += new_text
            if "</s>" in generated_text:
                generated_text = generated_text[: generated_text.find("</s>")]
                streamer.end()
                yield generated_text
                return
        return generated_text


def close_app():
    gr.Info("Terminated the app!")
    sleep(1)
    os._exit(0)

def main(args):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load base model
    print("*** Loading base model...")
    base_model_name = "davidkim205/komt-mistral-7b-v1"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Load fine-tuned layer and tokenizer
    print("*** Loading fine-tuned model...")
    fine_tuned_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, add_bos_token=True, trust_remote_code=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fine_tuned_model.to(device)
    base_model.config.use_cache = False
    fine_tuned_model.config.use_cache = False

    BaseModelInferenceApp = InferenceApp(base_model, tokenizer)
    FineTunedModelInferenceApp = InferenceApp(fine_tuned_model, tokenizer)

    css = "footer {visibility: hidden}"
    with gr.Blocks(css=css, title="Base model vs Fine-tuned") as demo:
        with gr.Row():
            gr.Markdown("<h2>Comparing Mistral-7B Base model vs fine-tuned</h2>")
        with gr.Row():
            with gr.Column():
                gr.Markdown("<h3>Base Model</h3>")
                gr.ChatInterface(BaseModelInferenceApp.generate)
            with gr.Column():
                gr.Markdown("<h3>Fine-tuned Model</h3>")
                gr.ChatInterface(FineTunedModelInferenceApp.generate)
        with gr.Row():
            close_button = gr.Button("Close the app", variant="stop")
            close_button.click(fn=lambda: gr.update(interactive=False), outputs=[close_button]).then(fn=close_app)

    demo.queue().launch(server_name="0.0.0.0").queue()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the inference app for comparing base model and fine-tuned model"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to the fine-tuned adapter"
    )
    args = parser.parse_args()

    main(args)
