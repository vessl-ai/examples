import argparse
import os
from time import sleep
from typing import List

import gradio as gr
from transformers import AutoTokenizer

from vllm import ModelRegistry
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid

from vllm_solar import SolarForCausalLM


class LLMChatHandler():
    def __init__(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        engine_args = AsyncEngineArgs(
            model=model_id,
            tokenizer=None,
            tokenizer_mode="auto",
            trust_remote_code=True,
            quantization="awq" if "awq" in model_id or "AWQ" in model_id else None,
            dtype="auto",
        )
        self.vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)

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

    async def chat_function(self, message, history):
        prompt = self.chat_history_to_prompt(message, history)
        response_generator = self.chat_function_vllm(prompt)
        async for text in response_generator:
            yield text

    async def chat_function_vllm(self, prompt):

        sampling_params = SamplingParams(
            stop_token_ids=self.terminators,
            max_tokens=2048,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2)
        results_generator = self.vllm_engine.generate(prompt, sampling_params, random_uuid())
        async for request_output in results_generator:
            response_txt = ""
            for output in request_output.outputs:
                if output.text not in self.terminators:
                    response_txt += output.text
            yield response_txt


def close_app():
    gr.Info("Terminated the app!")
    sleep(1)
    os._exit(0)


def main(args):
    print(f"Loading the model {args.model_id}...")
    hdlr = LLMChatHandler(args.model_id)

    with gr.Blocks(title=f"🤗 Chatbot with {args.model_id}", fill_height=True) as demo:
        with gr.Row():
            gr.Markdown(
                f"<h2>Chatbot with 🤗 {args.model_id} 🤗</h2>"
                "<h3>Interact with LLM using chat interface!<br></h3>"
                f"<h3>Original model: <a href='https://huggingface.co/{args.model_id}' target='_blank'>{args.model_id}</a></h3>")
        gr.ChatInterface(hdlr.chat_function)
        with gr.Row():
            close_button = gr.Button("Close the app", variant="stop")
            close_button.click(
                fn=lambda: gr.update(interactive=False), outputs=[close_button]
            ).then(fn=close_app)

    demo.queue().launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    ModelRegistry.register_model("SolarForCausalLM", SolarForCausalLM)

    parser = argparse.ArgumentParser(
        prog="OSS Chatbot",
        description="Run open source LLMs from HuggingFace with a simple chat interface")

    parser.add_argument("--model-id", default="upstage/solar-pro-preview-instruct",
                        help="HuggingFace model name for LLM.")
    parser.add_argument("--port", default=7860, type=int, help="Port number for the Gradio app.")
    parser.add_argument("--tensor-parallelism", default=1, type=int, help="Number of tensor parallelism.")
    args = parser.parse_args()

    main(args)