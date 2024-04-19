import argparse
import os
from time import sleep
from typing import List
from threading import Thread

import gradio as gr
from transformers import AutoTokenizer

class LLMChatHandler():
    def __init__(self, model_id: str, use_vllm: bool = False):
        self.use_vllm = use_vllm
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        if use_vllm:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            engine_args = AsyncEngineArgs(
                model=model_id,
                tokenizer=None,
                tokenizer_mode="auto",
                trust_remote_code=True,
                quantization="awq" if "awq" in model_id or "AWQ" in model_id else None,
                dtype="auto",
            )
            self.vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        else:
            from transformers import AutoModelForCausalLM
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                torch_dtype="auto",
                device_map="auto")

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
        if self.use_vllm:
            response_generator = self.chat_function_vllm(prompt)
            async for text in response_generator:
                yield text
        else:
            response_generator = self.chat_function_hf(prompt)
            for text in response_generator:
                yield text

    async def chat_function_vllm(self, prompt):
        from vllm import SamplingParams
        from vllm.utils import random_uuid
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

    def chat_function_hf(self, prompt):
        from transformers import pipeline, TextIteratorStreamer
        streamer = TextIteratorStreamer(self.tokenizer)
        pipe = pipeline(
            "text-generation",
            model=self.hf_model,
            tokenizer=self.tokenizer,
            eos_token_id=self.terminators,
            max_length=2048,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            return_full_text=False,
            streamer=streamer
        )
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
    hdlr = LLMChatHandler(args.model_id, args.use_vllm)

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
    parser = argparse.ArgumentParser(
        prog="OSS Chatbot",
        description="Run open source LLMs from HuggingFace with a simple chat interface")

    parser.add_argument("--model-id", default="casperhansen/llama-3-8b-instruct-awq", help="HuggingFace model name for LLM.")
    parser.add_argument("--port", default=7860, type=int, help="Port number for the Gradio app.")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM instead of HuggingFace AutoModelForCausalLM.")
    parser.add_argument("--tensor-parallelism", default=1, type=int, help="Number of tensor parallelism.")
    args = parser.parse_args()

    main(args)
