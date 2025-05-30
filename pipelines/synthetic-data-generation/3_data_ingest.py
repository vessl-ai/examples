import argparse
import os

import chromadb
import pandas as pd
import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def ingest_data(data_path: str, embedding_model: str, collection_name: str, chroma_path: str):
    df = pd.read_csv(data_path)

    # Initialize Chroma
    client = chromadb.PersistentClient(chroma_path)
    _collection = client.get_or_create_collection(collection_name)

    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "trust_remote_code": True,
            },
        ),
    )

    # Add documents to Chroma
    for index, row in df.iterrows():
        document = Document(id=index, page_content=row["question"], metadata={"answer": row["answer"], "code_example": row["code_example"], "reference": row["reference"]})
        vector_store.add_documents([document])

    print(f"** Ingested {len(df)} documents into {collection_name} **")


def main():
    parser = argparse.ArgumentParser(description="Ingest FAQ samples into Chroma.")
    parser.add_argument("--data-path", type=str, required=True, help="Data path to load")
    parser.add_argument("--chroma-path", type=str, required=True, help="Path to the Chroma database")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3", help="Name of the embedding to use")
    parser.add_argument("--collection-name", type=str, default="synthetic_data", help="Name of the collection to use")
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        raise ValueError(f"Data path {data_path} does not exist.")
    if not data_path.endswith(".csv"):
        raise ValueError(f"Data path {data_path} is not a CSV file.")
    
    print(f"** Ingesting data from {args.data_path} into Chroma **")
    ingest_data(
        data_path=args.data_path,
        embedding_model=args.embedding_model,
        collection_name=args.collection_name,
        chroma_path=args.chroma_path,
    )

    print("** Data ingestion complete **")

if __name__ == "__main__":
    main()
