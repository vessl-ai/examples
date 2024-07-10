from chromadb.config import Settings as ChromaDBSettings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
import torch

def get_embedding_model(model_name: str) -> Embeddings:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs={"device": device})

def get_chroma_vector_store(
    chroma_endpoint: str,
    chroma_port: int,
    chroma_collection: str,
    embedding_model: Embeddings) -> VectorStore:

    chroma_client_settings = None
    if chroma_endpoint:
        chroma_client_settings=ChromaDBSettings(
            chroma_api_impl="chromadb.api.fastapi.FastAPI",
            chroma_server_host=chroma_endpoint,
            chroma_server_http_port=str(chroma_port))
    vector_store = Chroma(
        collection_name=chroma_collection,
        embedding_function=embedding_model,
        client_settings=chroma_client_settings)

    return vector_store
