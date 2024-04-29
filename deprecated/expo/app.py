import argparse
import os
from time import sleep
from urllib.parse import urlparse

import gradio as gr
from openai import OpenAI


def refine_endpoint(url: str):
    r = urlparse(url)
    if not r.scheme:
        endpoint = f'https://{os.environ.get("ENDPOINT")}/v1'
    else:
        endpoint = f'{os.environ.get("ENDPOINT")}/v1'

    return endpoint


class ChatHandler:
    def __init__(self, mode: str):
        self.endpoint = refine_endpoint(os.environ.get("ENDPOINT"))
        self.model_name = "tgi" if mode == "vanilla" else os.environ.get("MODEL_NAME")
        self.client = OpenAI(base_url=self.endpoint, api_key="-")

        self.stop_tokens = [
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "<|reserved_special_token|>",
        ]

    def build_messages_with_prompt(self, message, history):
        messages = []
        for h in history:
            user_text, assistant_text = h
            messages += [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]

        messages.append({"role": "user", "content": message})

        return messages

    def chat_function(self, message, history):
        messages = self.build_messages_with_prompt(message, history)
        stream = self.client.chat.completions.create(
            model=self.model_name, messages=messages, stream=True, stop=self.tokens
        )

        partial_message = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                partial_message = partial_message + chunk.choices[0].delta.content
                yield partial_message


def close_app():
    gr.Info("Terminated the app!")
    sleep(1)
    os._exit(0)


def main(args):
    print(f"Launching with {args.mode}...")
    hdlr = ChatHandler(args.mode)

    with gr.Blocks(title=f"Llama 3 Chatbot ({args.mode})", fill_height=True) as demo:
        with gr.Row():
            gr.Markdown(
                f"<h2>Llama 3 Chatbot ({args.mode})</h2>"
                "<h3>Interact with LLM using chat interface!<br></h3>"
                # f"<h3>Original model: <a href='https://huggingface.co/{args.model_id}' target='_blank'>{args.model_id}</a></h3>"
            )

        gr.ChatInterface(hdlr.chat_function)

        with gr.Row():
            close_button = gr.Button("Close the app", variant="stop")
            close_button.click(
                fn=lambda: gr.update(interactive=False), outputs=[close_button]
            ).then(fn=close_app)

    demo.queue().launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="OSS Chatbot",
        description="Run open source LLMs from HuggingFace with a simple chat interface",
    )
    parser.add_argument(
        "--mode",
        choices=("vanilla", "vllm"),
        help="Connect to service whether vanilla or vllm server",
        required=True,
    )
    parser.add_argument(
        "--port",
        default=7860,
        help="Port to open",
    )
    args = parser.parse_args()

    main(args)
