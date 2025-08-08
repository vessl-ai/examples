import argparse
import logging
import os
import shutil
import sys
from time import sleep
from typing import List, Optional

import gradio as gr
from openai import AsyncOpenAI
import torch


class LLMChatHandler():
    def __init__(self, model_id: str, llm_host: str, llm_api_key: str):
        self.llm_client = AsyncOpenAI(
            base_url=llm_host,
            api_key=llm_api_key,
        )
        self.model_id = model_id

    async def chat_function(self, message, history):
        history.append({"role": "user", "content": message})
        response = await self.llm_client.chat.completions.create(
            model=self.model_id,
            messages=history,
        )
        yield response.choices[0].message.content


def main(args: argparse.Namespace):
    hdlr = LLMChatHandler(
        model_id=args.model_id,
        llm_host=args.llm_host,
        llm_api_key=args.llm_api_key,
    )

    with gr.Blocks(title=f"OpenAI Chatbot with {args.model_id}", fill_height=True) as demo:
        with gr.Row():
            gr.Markdown(
                f"<h2>Chatbot with {args.model_id}</h2>"
                "<h3>Interact with LLM using chat interface!<br></h3>"
            )
        gr.ChatInterface(hdlr.chat_function, type="messages")

    demo.queue().launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="OpenAI Chatbot",
        description="Chat with OpenAI LLMs with a simple interface"
    )

    parser.add_argument(
        "--port",
        default=7860,
        type=int,
        help="Port to run the Gradio server.",
    )
    parser.add_argument(
        "--model-id",
        default="openai/gpt-oss-20b",
        help="Model name for LLM.",
    )
    parser.add_argument(
        "--llm-host",
        default="http://localhost:8000/v1",
        help="OpenAI or compatible API endpoint.",
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="API key for OpenAI-compatible LLM API.",
    )

    args = parser.parse_args()

    main(args)