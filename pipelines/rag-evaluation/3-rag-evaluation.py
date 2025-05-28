import argparse
import os

import torch

from datasets import load_from_disk
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import VLLM
from ragas import evaluate
from ragas.metrics import (
    faithfulness, 
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)


def main(args: argparse.Namespace):

    # initialize evaluation models
    evaluation_model = VLLM(
        model=args.evaluation_model,
        vllm_kwargs={
            "gpu_memory_utilization": 0.8,
            "max_model_len": 4096,
        },
        temperature=0.0,
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
    for rag_pattern in ["naive", "hyde", "reranking", "hyde-reranking"]:
        # load RAG results
        dataset_path = os.path.join(rag_pattern, f"dataset-{rag_pattern}")
        print(f"Loading RAG results from {dataset_path}...")
        dataset = load_from_disk(dataset_path)
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
        print(f"{rag_pattern} RAG evaluation results:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-model", type=str, default="gpt-4o")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3")
    args = parser.parse_args()
    main(args)
