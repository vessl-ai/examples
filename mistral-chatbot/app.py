import os
from time import sleep
from typing import List

import gradio as gr
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_NAME = os.environ.get("MODEL_NAME", "TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
MODEL = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    quantization="awq",
    dtype="auto",
)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)


def history_to_prompt(message: str, history: List[List[str]]) -> str:
    conversation = []
    for h in history:
        user_text, assistant_text = h
        conversation += [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]

    conversation.append({"role": "user", "content": message})
    prompt = TOKENIZER.apply_chat_template(conversation, tokenize=False)

    return prompt


def close_app():
    gr.Info("Terminated the app!")
    sleep(1)
    os._exit(0)


def chat_function(message, history):
    prompt = history_to_prompt(message, history)
    sampling_params = SamplingParams(
        max_tokens=512,
        top_k=10,
        top_p=0.95,
        temperature=0.8,
    )
    outputs = MODEL.generate(prompt, sampling_params)
    return outputs[0].outputs[0].text


def main():
    with gr.Blocks(title="Mistral Chatbot on vLLM", fill_height=True) as demo:
        with gr.Row():
            gr.Markdown("<h2>Mistral Chatbot</h2>")
        gr.ChatInterface(chat_function)
        with gr.Row():
            close_button = gr.Button("Close the app", variant="stop")
            close_button.click(
                fn=lambda: gr.update(interactive=False), outputs=[close_button]
            ).then(fn=close_app)

    demo.queue().launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
