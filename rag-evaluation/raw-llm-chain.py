import os
import pickle

import chromadb
from datasets import Dataset
from langchain_chroma import Chroma
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


data_path = os.environ.get("DATA_PATH", "/data")
llm_endpoint = os.environ.get("LLM_ENDPOINT", "https://api.openai.com")
llm_model = os.environ.get("LLM_MODEL", "gpt-4o")
output_path = os.environ.get("OUTPUT_PATH", "/output")


# load claims
print("loading claims...")
claims_path = os.path.join(data_path, "claims.pkl")
with open(claims_path, "rb") as fr:
    claims = pickle.load(fr)
print(f"claims length: {len(claims)}")

# initialize LLM chain
prompt = PromptTemplate.from_template(
    "As an AI assistant, you are assigned to perform verification tasks. Please answer whether the given claim is true or false.\n"
    "Presented Claim: {question}"
)

base_url = os.path.join(llm_endpoint, "v1")
llm = ChatOpenAI(
    base_url=base_url,
    model=llm_model,
    temperature=0,
    max_tokens=4096,
    streaming=True,
)

rag_chain_from_docs = (
    prompt
    | llm
    | StrOutputParser()
)

rag_chain = RunnablePassthrough.assign(
    answer=rag_chain_from_docs
)

# get LLM results
print("Getting LLM results...")
inputs = [{"question": c["claim"]} for c in claims]

results = rag_chain.batch(inputs)

# save LLM results
dataset = Dataset.from_dict({
    "question": [r["question"] for r in results],
    "answer": [r["answer"] for r in results],
    "ground_truth": ["TRUE" if c["label"] == "SUPPORTS" else "FALSE" for c in claims]
})

dataset_path = os.path.join(output_path, f"dataset-raw")
os.makedirs(output_path, exist_ok=True)
dataset.save_to_disk(dataset_path)
print(f"Saved RAG results to {dataset_path}.")
