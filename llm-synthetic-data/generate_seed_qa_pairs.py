import argparse
import json
import os
import re

from dotenv import load_dotenv

from query_engine import load_query_engine

load_dotenv()

PERSIST_DIR = "./storage"

Q1 = "How can I upload output files explicitly using VESSL Python SDK?"
A1 = 'Use `vessl.upload` function. For example, If you want to upload `outputs` folder, do `vessl.upload("./outputs")`.'

INPUT_QUERY = f"""List the 20 questions and corresponding answers that users will possibly ask while they are using VESSL.
Please follow these conditions.
- The answer should contain an exact command or function if it is available in context.
- Listing all questions that start with 'how' should be avoided.
- Do not provide the document only but explain how user should behave.

[List]
Q1. {Q1}
A1. {A1}
"""


def generate_seed_qa_pairs(
    output_dir: str,
    docs_dir: str,
    model_name: str = "gpt-4-turbo-preview",
    temperature: float = 1.0,
):
    query_engine = load_query_engine(
        docs_dir=docs_dir,
        persist_dir=PERSIST_DIR,
        model=model_name,
        temperature=temperature,
        max_tokens=4096,
    )

    response = query_engine.query(INPUT_QUERY)
    pattern = re.compile(r"Q\d+. (.*?)\nA\d+. (.*?)\n\n")

    matches = pattern.findall(response.response)
    qa_pairs = [{"question": Q1, "answer": A1}] + [
        {"question": q, "answer": a} for q, a in matches
    ]

    with open(os.path.join(output_dir, "seed.json"), "w") as fp:
        json.dump(qa_pairs, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--docs-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    generate_seed_qa_pairs(
        args.output_dir,
        args.docs_dir,
        model_name=args.model,
        temperature=args.temperature,
    )
