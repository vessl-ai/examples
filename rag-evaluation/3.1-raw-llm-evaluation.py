import os

from datasets import load_from_disk
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas.metrics import answer_correctness
from ragas import evaluate


evaluation_endpoint = os.environ.get("EVALUATION_ENDPOINT", "https://api.openai.com")
evaluation_model = os.environ.get("EVALUATION_MODEL", "gpt-4o")
embedding_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
data_path = os.environ.get("DATA_PATH", "/data")


# load LLM results
dataset_path = os.path.join(data_path, "dataset-raw")
print(f"Loading raw LLM results from {dataset_path}...")
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

# evaluate LLM results
print("Evaluating raw LLM results...")
results = evaluate(
    dataset, 
    metrics=[answer_correctness],
    llm=evaluation_model,
    embeddings=evaluation_embeddings,
)
print("raw LLM evaluation results:", results)
