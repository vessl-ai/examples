import argparse
import base64
import os
from time import sleep

import gradio as gr
from openai import OpenAI


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def chat_function(message, history):
    history_openai_format = []
    for h in history:
        if isinstance(h["content"], tuple):
            imgs = [encode_image(img) for img in h["content"]]
            msg = {
                "role": h["role"],
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in imgs
                ],
            }
        else:
            msg = {
                "role": h["role"],
                "content": [
                    {"type": "text", "text": h["content"]}
                ],
            }
        history_openai_format.append(msg)
    
    msg = {"role": "user", "content": [{"type": "text", "text": message["text"]}]}
    if message["files"]:
        for f in message["files"]:
            base64_img = encode_image(f)
            msg["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                }
            )
    history_openai_format.append(msg)

    client = OpenAI(base_url=args.llm_endpoint)

    stream = client.chat.completions.create(
        model=args.model_id,
        messages=history_openai_format,
        stream=True,
    )

    partial_message = ""
    for chunk in stream:
        partial_message += (chunk.choices[0].delta.content or "")
        yield partial_message


def close_app():
    gr.Info("Terminated the app!")
    sleep(1)
    os._exit(0)


def main(args):
    print(f"Loading the model {args.model_id}...")

    with gr.Blocks(title=f"🤗 Chatbot with {args.model_id}", fill_height=True) as demo:
        with gr.Row():
            gr.Markdown(
                f"<h2>Chatbot with 🤗 {args.model_id} 🤗</h2>"
                "<h3>Interact with LLM using chat interface!<br></h3>"
                f"<h3>Original model: <a href='https://huggingface.co/{args.model_id}' target='_blank'>{args.model_id}</a></h3>")
        gr.ChatInterface(chat_function, multimodal=True, type="messages")
        with gr.Row():
            close_button = gr.Button("Close the app", variant="stop")
            close_button.click(
                fn=lambda: gr.update(interactive=False), outputs=[close_button]
            ).then(fn=close_app)

    demo.queue().launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="OSS Chatbot",
        description="Run open source LLMs from HuggingFace with a simple chat interface")

    parser.add_argument("--model-id", default="microsoft/Phi-3.5-vision-instruct",
                        help="HuggingFace model name for LLM.")
    parser.add_argument("--port", default=7860, type=int, help="Port number for the Gradio app.")
    parser.add_argument("--llm-endpoint", default="http://localhost:8000/v1", help="OpenAI or compatible API endpoint.")
    args = parser.parse_args()

    main(args)