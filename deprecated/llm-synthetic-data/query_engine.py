import os
from typing import Any

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


def load_query_engine(
    docs_dir: str,
    persist_dir: str = "./storage",
    model: str = "gpt-4-turbo-preview",
    temperature: float = 0,
    max_tokens: int = 4096,
    additional_kwargs: dict[str, Any] = None,
):
    if os.path.exists(persist_dir):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader(docs_dir, recursive=True).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)

    if additional_kwargs is None:
        additional_kwargs = {}
    Settings.llm = OpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        additional_kwargs=additional_kwargs,
    )
    Settings.embed_model = OpenAIEmbedding()

    query_engine = index.as_query_engine()

    return query_engine
