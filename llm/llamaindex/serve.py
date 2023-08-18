from __future__ import annotations

import typing as t
from typing import Any, List

import bentoml
import torch
from InstructorEmbedding import INSTRUCTOR
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.base import BaseEmbedding
from llama_index.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# custom LLM class for llamaindex
class Llama2Model(CustomLLM):
    def __init__(self):
        model = AutoModelForCausalLM.from_pretrained(
            "/data/llama-2-7b-hf", device_map="auto", torch_dtype=torch.float16
        )
        model.eval()
        self.model = model
        tokenizer = AutoTokenizer.from_pretrained("/data/llama-2-7b-hf", legacy=False)
        tokenizer.pad_token = tokenizer.unk_token
        self.tokenizer = tokenizer

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(name="custom-llama2")

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        tokenized = self.tokenizer(prompt)
        tokenized["input_ids"] = (
            torch.tensor(tokenized["input_ids"]).unsqueeze(0).to("cuda")
        )
        tokenized["attention_mask"] = (
            torch.ones(tokenized["input_ids"].size(1)).unsqueeze(0).to("cuda")
        )
        outputs = self.model.generate(
            input_ids=tokenized["input_ids"],
            max_new_tokens=1024,
            attention_mask=tokenized["attention_mask"],
        )
        result = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return CompletionResponse(text=result)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError()


# custom Embedder class for llamaindex
class InstructorEmbeddings(BaseEmbedding):
    def __init__(
        self,
        instructor_model_name: str = "hkunlp/instructor-large",
        instruction: str = "Represent a document for semantic search:",
        **kwargs: Any,
    ) -> None:
        self._model = INSTRUCTOR(instructor_model_name)
        self._instruction = instruction
        super().__init__(**kwargs)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, query]])
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, text]])
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode([[self._instruction, text] for text in texts])
        return embeddings


class LlamaIndex(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        context_window = 2048
        num_output = 1024
        documents = SimpleDirectoryReader("/docs/vessl-docs-dataset/").load_data()
        llm = Llama2Model()
        service_context = ServiceContext.from_defaults(
            llm=llm,
            context_window=context_window,
            num_output=num_output,
            embed_model=InstructorEmbeddings(embed_batch_size=2),
            chunk_size=512,
        )
        self.index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )

    @bentoml.Runnable.method(batchable=False)
    def generate(self, input_text: str) -> bool:
        result = self.index.as_query_engine().query(input_text)
        return result


llamaindex_runner = t.cast("RunnerImpl", bentoml.Runner(LlamaIndex, name="llamaindex"))
svc = bentoml.Service("llamaindex_service", runners=[llamaindex_runner])


@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON())
async def infer(text: str) -> str:
    result = await llamaindex_runner.generate.async_run(text)
    return result
