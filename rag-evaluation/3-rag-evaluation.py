import os

from datasets import load_from_disk
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas.metrics import (
    faithfulness, 
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from ragas import evaluate


evaluation_endpoint = os.environ.get("EVALUATION_ENDPOINT", "https://api.openai.com")
evaluation_model = os.environ.get("EVALUATION_MODEL", "gpt-4o")
embedding_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
rag_pattern = os.environ.get("RAG_PATTERN", "naive")
data_path = os.environ.get("DATA_PATH", "/data")


# load RAG results
dataset_path = os.path.join(data_path, f"dataset-{rag_pattern}")
print(f"Loading RAG results from {dataset_path}...")
dataset = load_from_disk(dataset_path)

# initialize evaluation models
print("evaluation endpoint:", evaluation_endpoint)
base_url = os.path.join(evaluation_endpoint, "v1")
evaluation_model = ChatOpenAI(
    base_url=base_url,
    model=evaluation_model,
    temperature=0.0,
    max_tokens=1024,
    streaming=True,
)
evaluation_embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model,
    model_kwargs={"device": "cuda"},
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
print(f"{rag_pattern} RAG evaluation results:", results)
