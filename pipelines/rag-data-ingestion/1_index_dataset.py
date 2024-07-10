import argparse
import logging
from typing import List
import os
import sys

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from tqdm import tqdm

from util import get_embedding_model, get_chroma_vector_store

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def generate_text_chunks(
    file_path: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 256) -> List[Document]:

    loader = PyMuPDFLoader(file_path=file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_documents(documents)

    return text_chunks

def main(args):
    # Conform vector store connect to Chroma collection
    logger.info(f"Prepraing embedding model {args.embedding_model}")
    embeddings = get_embedding_model(args.embedding_model)

    logger.info(f"Connecting to Chroma at {args.chroma_endpoint}:{args.chroma_port}/collection/{args.chroma_collection_name}")
    vector_store = get_chroma_vector_store(
        chroma_endpoint=args.chroma_endpoint,
        chroma_port=args.chroma_port,
        chroma_collection=args.chroma_collection_name,
        embedding_model=embeddings
    )

    # Load documents and split into chunks
    logger.info(f"Loading documents from dataset at {args.dataset_path}")
    document_paths = []
    for root, _, files in os.walk(args.dataset_path):
        for file in files:
            if file.endswith(".pdf"):
                document_paths.append(os.path.join(root, file))

    logger.info(f"Found {len(document_paths)} documents in the dataset")

    # Ingest the chunks into the vector store using tqdm
    # TODO: consider idempotence (Do not re-ingest the same documents)
    total_document_size = len(document_paths)
    print("Current collections:", vector_store._client.get_collection(args.chroma_collection_name).count())
    with tqdm(total = total_document_size) as pbar:
        for doc in document_paths:
            documents = generate_text_chunks(
                file_path=doc,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            total_document_size += len(documents)
            pbar.total = total_document_size
            pbar.refresh()
            for document in documents:
                vector_store.add_documents([document])
                pbar.update(1)
            pbar.update(1)

    logger.info("Ingested all documents into the vector store")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma-endpoint", type=str)
    parser.add_argument("--chroma-port", type=int, default=0)
    parser.add_argument("--chroma-collection-name", type=str, default="langflow")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=256)
    parser.add_argument("--dataset-path", type=str, default="./test")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3")

    args = parser.parse_args()
    main(args)
