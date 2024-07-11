import argparse
import os
import pickle

from datasets import Dataset
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def load_dataset(data_path):
    print("loading claims...")
    claims_path = os.path.join(data_path, "claims.pkl")
    with open(claims_path, "rb") as fr:
        claims = pickle.load(fr)
    print(f"claims length: {len(claims)}")

    return claims


def initialize_llm_chain(llm_endpoint, llm_model):
    prompt = PromptTemplate.from_template(
        "As an AI assistant, you are assigned to perform verification tasks. Please answer whether the given claim is true or false.\n"
        "Presented Claim: {question}"
    )

    print("LLM endpoint:", llm_endpoint)
    base_url = os.path.join(llm_endpoint, "v1")
    llm = ChatOpenAI(
        base_url=base_url,
        model=llm_model,
        temperature=0,
        max_tokens=4096,
        streaming=True,
    )

    llm_chain_from_prompt = (
        prompt
        | llm
        | StrOutputParser()
    )

    llm_chain = RunnablePassthrough.assign(
        answer=llm_chain_from_prompt
    )

    return llm_chain


def llm_chain_batch(llm_chain, claims):
    print("Getting raw LLM results...")
    inputs = [{"question": c["claim"]} for c in claims]

    results = llm_chain.batch(inputs)

    # save LLM results
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "ground_truth": ["TRUE" if c["label"] == "SUPPORTS" else "FALSE" for c in claims]
    })

    return dataset


def main(args: argparse.Namespace):
    claims = load_dataset(args.data_path)
    llm_chain = initialize_llm_chain(args.llm_endpoint, args.llm_model)
    llm_results = llm_chain_batch(llm_chain, claims)

    dataset_path = os.path.join(args.output_path, f"dataset-raw-{args.llm_model}")
    os.makedirs(args.output_path, exist_ok=True)
    llm_results.save_to_disk(dataset_path)
    print(f"Saved LLM results to {dataset_path}.")


if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/data")
    parser.add_argument("--llm-endpoint", type=str, default="https://api.openai.com")
    parser.add_argument("--llm-model", type=str, default="gpt-4o")
    parser.add_argument("--output-path", type=str, default="/output")
    args = parser.parse_args()
    main(args)
