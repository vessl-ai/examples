import argparse
import os

import torch

from datasets import load_from_disk
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    faithfulness, 
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)


def main(args: argparse.Namespace):
    # load RAG results
    dataset_path = os.path.join(args.data_path, f"dataset-{args.rag_pattern}")
    print(f"Loading RAG results from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    # initialize evaluation models
    print("evaluation endpoint:", args.evaluation_endpoint)
    base_url = os.path.join(args.evaluation_endpoint, "v1")
    evaluation_model = ChatOpenAI(
        base_url=base_url,
        model=evaluation_model,
        temperature=0.0,
        max_tokens=4096,
        streaming=True,
    )
    evaluation_embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "trust_remote_code": True,
        },
    )

    # evaluate RAG results
    print("Evaluating RAG results...")
    results = evaluate(
        dataset, 
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ],
        llm=evaluation_model,
        embeddings=evaluation_embeddings,
    )
    print(f"{args.rag_pattern} RAG evaluation results:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/data")
    parser.add_argument("--evaluation-endpoint", type=str, default="https://api.openai.com")
    parser.add_argument("--evaluation-model", type=str, default="gpt-4o")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--rag-pattern", type=str)
    args = parser.parse_args()
    main(args)
