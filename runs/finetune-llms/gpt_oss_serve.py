# Simple GPT-OSS API server for serving the merged model
# Once vLLM support LoRA adapter serving for GPT-OSS, this will be deprecated

import argparse
import json
import time
from typing import List, Dict, Any, Optional, Iterator
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
import torch
import threading


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "gpt-oss-20b"
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = None
    top_k: int = None
    stream: bool = False


class ChatResponse(BaseModel):
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


def load_gpt_oss_model(base_model_name: str, lora_adapter_path: str = None):
    """Load GPT-OSS model with optional LoRA adapter."""
    print(f"Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print(f"Loading base model: {base_model_name}")
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype="auto",
        use_cache=True,
        device_map="auto"
    )
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)

    if lora_adapter_path:
        print(f"Loading LoRA adapter from: {lora_adapter_path}")
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        print("Merging LoRA adapter with base model...")
        model = model.merge_and_unload()
    else:
        model = base_model

    print("✅ Model loaded successfully!")
    return model, tokenizer


def stream_chat_response(model, tokenizer, input_ids, gen_kwargs, model_name) -> Iterator[str]:
    """Stream chat response using TextIteratorStreamer."""
    # Setup streaming
    streamer = TextIteratorStreamer(
        tokenizer,
        timeout=30.0,
        skip_prompt=True,
        skip_special_tokens=True
    )
    gen_kwargs["streamer"] = streamer
    
    # Ensure proper EOS handling for streaming
    if "eos_token_id" not in gen_kwargs:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if "early_stopping" not in gen_kwargs:
        gen_kwargs["early_stopping"] = True

    # Start generation in a separate thread
    thread = threading.Thread(target=model.generate, args=(input_ids,), kwargs=gen_kwargs)
    thread.start()

    # Create a unique ID for this response
    response_id = f"chatcmpl-{int(time.time())}"

    # Send each token as it's generated
    total_tokens = 0
    for new_text in streamer:
        total_tokens += 1
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": new_text
                    },
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Send final chunk
    final_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": input_ids.shape[1],
            "completion_tokens": total_tokens,
            "total_tokens": input_ids.shape[1] + total_tokens
        }
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

    thread.join()


def create_app(model, tokenizer):
    """Create FastAPI app with GPT-OSS model."""
    app = FastAPI(title="GPT-OSS API Server", version="1.0.0")

    @app.get("/")
    async def root():
        return {"message": "GPT-OSS API Server is running! Go do /docs to see the API docs."}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "gpt-oss-20b",
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "openai"
                }
            ]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        try:
            # Convert messages to format expected by tokenizer
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]

            # Apply chat template
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

            # Generation parameters
            gen_kwargs = {
                "max_new_tokens": request.max_tokens,
                "do_sample": True,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "early_stopping": True,
            }

            # Handle streaming vs non-streaming
            if request.stream:
                return StreamingResponse(
                    stream_chat_response(model, tokenizer, input_ids, gen_kwargs, request.model),
                    media_type="text/plain"
                )
            else:
                # Generate response
                output_ids = model.generate(input_ids, **gen_kwargs)

                # Decode response
                response_ids = output_ids[0][input_ids.shape[1]:]
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

                # Count tokens
                input_tokens = input_ids.shape[1]
                output_tokens = len(response_ids)

                return {
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens
                    }
                }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


def main(
    base_model: str,
    lora_adapter: str = None,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """Main function to start GPT-OSS API server."""
    print("🚀 Starting GPT-OSS API Server...")

    # Load model and tokenizer
    model, tokenizer = load_gpt_oss_model(base_model, lora_adapter)

    # Create FastAPI app
    app = create_app(model, tokenizer)

    print(f"🌐 Server starting on http://{host}:{port}")
    print(f"📖 API docs available at http://{host}:{port}/docs")

    # Run server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-OSS API Server with LoRA support")
    parser.add_argument(
        "--base_model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Base model name or path"
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter (optional)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server"
    )

    args = parser.parse_args()

    main(
        base_model=args.base_model,
        lora_adapter=args.lora_adapter,
        host=args.host,
        port=args.port,
    )
