import argparse
import os
from time import sleep
from typing import List
from threading import Thread

import gradio as gr
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

class LLMChatHandler():
    def __init__(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype="auto",
            device_map="auto")
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def chat_history_to_prompt(self, message: str, history: List[List[str]]) -> str:
        conversation = []
        for h in history:
            user_text, assistant_text = h
            conversation += [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]

        conversation.append({"role": "user", "content": message})
        prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        return prompt

    def chat_function(self, message, history):
        streamer = TextIteratorStreamer(self.tokenizer)
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            eos_token_id=self.terminators,
            max_length=2048,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            streamer=streamer
        )
        prompt = self.chat_history_to_prompt(message, history)
        t = Thread(target=pipe, args=(prompt,))
        t.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield generated_text

        t.join()

def close_app():
    gr.Info("Terminated the app!")
    sleep(1)
    os._exit(0)

def main(args):
    print(f"Loading the model {args.model_id}...")
    hdlr = LLMChatHandler(args.model_id)

    with gr.Blocks(title="Mistral Chatbot on vLLM", fill_height=True) as demo:
        with gr.Row():
            gr.Markdown(f"<h2>Chatbot with {args.model}</h2>")
        gr.ChatInterface(hdlr.chat_function)
        with gr.Row():
            close_button = gr.Button("Close the app", variant="stop")
            close_button.click(
                fn=lambda: gr.update(interactive=False), outputs=[close_button]
            ).then(fn=close_app)

    demo.queue().launch(server_name="0.0.0.0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="OSS Chatbot",
        description="Run open source LLMs from HuggingFace with a simple chat interface")

    parser.add_argument("--model-id", default="casperhansen/llama-3-8b-instruct-awq", help="HuggingFace model name for LLM.")
    args = parser.parse_args()

    main(args)
