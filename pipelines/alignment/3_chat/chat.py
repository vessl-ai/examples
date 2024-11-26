import argparse
import os
import time
from threading import Thread

import gradio as gr
from transformers import AutoTokenizer


class LLMChatHander:
    def __init__(self, model_name_or_path: str, use_vllm: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        if use_vllm:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine

            engine_args = AsyncEngineArgs(
                model=model_name_or_path,
                tokenizer=None,
                tokenizer_mode="auto",
                trust_remote_code=True,
                quantization="awq"
                if "awq" in model_name_or_path or "AWQ" in model_name_or_path
                else None,
                dtype="auto",
            )
            self.vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        else:
            from transformers import AutoModelForCausalLM

            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                torch_dtype="auto",
                device_map="auto",
            )

    def chat_history_to_prompt(self, message: str, history: list[list[str]]) -> str:
        conversation = []
        for h in history:
            user_text, assistant_text = h
            conversation += [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]

        conversation.append({"role": "user", "content": message})
        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        return prompt

    async def chat_function(self, message: str, history: list[list[str]]):
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

        sampling_params = SamplingParams(max_tokens=4096, temperature=1.0, top_p=0.9)
        generator = self.vllm_engine.generate(prompt, sampling_params, random_uuid())

        async for response in generator:
            response_text = ""
            for output in response.outputs:
                response_text += output.text
            yield response_text

    def chat_function_hf(self, prompt):
        from transformers import TextIteratorStreamer, pipeline

        streamer = TextIteratorStreamer(self.tokenizer)
        pipe = pipeline(
            "text-generation",
            model=self.hf_model,
            tokenizer=self.tokenizer,
            max_length=4096,
            temperature=1.0,
            top_p=0.9,
            return_full_text=False,
            streamer=streamer,
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
    time.sleep(1)
    os._exit(0)


def main(model_name_or_path: str, port: int, use_vllm: bool):
    handler = LLMChatHander(model_name_or_path=model_name_or_path, use_vllm=use_vllm)

    with gr.Blocks(title="Chat for human evaluation") as demo:
        with gr.Row():
            gr.Markdown(f"<h3>Model: {model_name_or_path}</h3>")
        gr.ChatInterface(handler.chat_function)
        with gr.Row():
            close_button = gr.Button("Close", variant="stop")
            close_button.click(
                fn=lambda: gr.update(interactive=False), outputs=[close_button]
            ).then(fn=close_app)

    demo.queue().launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--use-vllm", action="store_true")
    args = parser.parse_args()

    main(args.model_name_or_path, args.port, args.use_vllm)
