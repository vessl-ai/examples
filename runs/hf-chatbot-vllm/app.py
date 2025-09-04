import argparse
from typing import Optional

import gradio as gr
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid


class LLMChatHandler():
    def __init__(self, model_id: str, max_num_seqs: int, max_model_len: int, dtype: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        def _guess_quantization(model_id) -> Optional[str]:
            if "awq" in model_id or "AWQ" in model_id:
                return "awq"
            if "bnb" in model_id:
                return "bitsandbytes"
            return None

        def _guess_load_format(model_id) -> Optional[str]:
            if "bnb" in model_id:
                return "bitsandbytes"
            return "auto"

        engine_args = AsyncEngineArgs(
            model=model_id,
            task="generate",
            tokenizer=None,
            tokenizer_mode="auto",
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            load_format=_guess_load_format(model_id=model_id),
            quantization=_guess_quantization(model_id=model_id),
            dtype=dtype,
        )
        self.vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)


    async def chat_function(self, message, history):
        history.append({"role": "user", "content": message})
        prompt = self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
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


def main(args):
    print(f"Loading the model {args.model_id}...")
    hdlr = LLMChatHandler(model_id=args.model_id, max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len, dtype=args.dtype)

    with gr.Blocks(title=f"🤗 Chatbot with {args.model_id}", fill_height=True) as demo:
        with gr.Row():
            gr.Markdown(
                f"<h2>Chatbot with 🤗 {args.model_id} 🤗</h2>"
                "<h3>Interact with LLM using chat interface!<br></h3>"
                f"<h3>Original model: <a href='https://huggingface.co/{args.model_id}' target='_blank'>{args.model_id}</a></h3>")
        gr.ChatInterface(hdlr.chat_function, type="messages")

    demo.queue().launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="OSS Chatbot",
        description="Run open source LLMs from HuggingFace with a simple chat interface")

    parser.add_argument("--model-id", default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit", help="HuggingFace model name for LLM.")
    parser.add_argument("--port", default=7860, type=int, help="Port number for the Gradio app.")
    parser.add_argument("--dtype", default="auto", type=str, help="Data type for model weights and activations.")
    parser.add_argument("--max-num-seqs", default=16, type=int, help="")
    parser.add_argument("--max-model-len", default=32767, type=int, help="")
    args = parser.parse_args()

    main(args)