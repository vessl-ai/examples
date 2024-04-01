import logging
import sys
import torch
import bentoml
import llama_index

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM

import typing as t
from typing import Any, List
from InstructorEmbedding import INSTRUCTOR

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.embeddings.base import BaseEmbedding
from llama_index.prompts import PromptTemplate


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# Setup for Custom Enbedding
class InstructorEmbeddings(BaseEmbedding):
    _model: INSTRUCTOR = PrivateAttr()
    _instruction: str = PrivateAttr()

    def __init__(
        self,
        instructor_model_name: str = "hkunlp/instructor-large",
        instruction: str = "Represent a document for semantic search:",
        **kwargs: Any,
    ) -> None:
        self._model = INSTRUCTOR(instructor_model_name)
        self._instruction = instruction
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, query]])
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, text]])
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            [[self._instruction, text] for text in texts]
        )
        return embeddings

    
class LlamaIndex(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu",)
    SUPPORTS_CPU_MULTI_THREADING = False
    
    def __init__(self):
        # load documents
        documents = SimpleDirectoryReader("/docs/vessl-docs-dataset/").load_data()

        # This will wrap the default prompts that are internal to LlamaIndex
        query_wrapper_prompt = PromptTemplate(
            "Given the context information and not prior knowledge, answer the query.\n\n"
            "### Instruction:\n{query_str}\n\n### Response:"
        )

        context_window = 1024
        max_length = 1024
        num_output = 256
        embed_batch_size = 2
        chunk_size = 256

        llm = HuggingFaceLLM(
            context_window=context_window,
            max_new_tokens=num_output,
            generate_kwargs={"temperature": 0.25, "do_sample": False},
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name="/data/llama-2-7b-hf",
            model_name="/data/llama-2-7b-hf",
            device_map="auto",
            tokenizer_kwargs={"max_length": max_length},
            # uncomment this if using CUDA to reduce memory usage
            model_kwargs={"torch_dtype": torch.float16}
        )

        service_context = ServiceContext.from_defaults(llm=llm, context_window=context_window, num_output=num_output, embed_model=InstructorEmbeddings(embed_batch_size=embed_batch_size), chunk_size=chunk_size)
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        query_engine = index.as_query_engine()
        self.query_engine = query_engine
        
    @bentoml.Runnable.method(batchable=False)
    def generate(self, input_text: str) -> bool:
        # set Logging to DEBUG for more detailed outputs
        result = self.query_engine.query(input_text)
        print("Query: " + input_text)
        print("Answer: ")
        print(result)
        return result
    

llamaindex_runner = t.cast("RunnerImpl", bentoml.Runner(LlamaIndex, name="llamaindex"))

svc = bentoml.Service("llamaindex_service", runners=[llamaindex_runner])
@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON())
async def infer(text: str) -> str:
    result = await llamaindex_runner.generate.async_run(text)
    return result
