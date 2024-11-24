import argparse
import csv
import glob
import json
import os
import re

from openai import OpenAI


prompt = [
    {
        "role": "system",
        "content": """
Role & Task: You are a support engineer at VESSL AI. Your role is to assist clients by creating an FAQ (Frequently Asked Questions) to help them understand VESSL AI and its services (VESSL Service and VESSL Pipeline) more effectively.

Requirements:
* Generate as many question-answer pairs as user's requsest.
* Each Q&A pair must follow strict JSON format.
* Answers must be detailed and easily validated by clients using the provided document.
* Include links to the relevant sections of the document to support the answers, ensuring customers can cross-reference.
* Where applicable, provide example code snippets to demonstrate SDK or CLI usage.
* Ensure no additional messages like "here is..." are included. Only JSON should be returned.

Format:
[
    {
        "question": "<question_1>",
        "answer": "<detailed_answer_1>",
        "code_example": "<optional_code_snippet_if_required>",
        "reference": "<link_to_document>"
    },
    {
        "question": "<question_2>",
        "answer": "<detailed_answer_2>",
        "code_example": "<optional_code_snippet_if_required>",
        "reference": "<link_to_document>"
    }
    ...
]
        """,
    },
    {
        "role": "system",
        "content": """<examples>
[
  {
    "question": "How to create VESSL Run via Python SDK?",
    "answer": "Here are the sample code to create VESSL Run via SDK. Here is the example:",
    "code_example": "```python\nimport vessl\n\nsample_json={\"test\":\"escaped_quote\"}\n```",
    "reference": "https://docs.vessl.ai/guides/get-started/quickstart"
  },
  {
    "question": "What are the options available for defining input variables in VESSL Pipeline?",
    "answer": "For input variables, you can define the following options: Name, Description, Type (either String or Choice), and Value (default value of the variable). The Choice type provides a dropdown menu in the pipeline execution, while both options are treated as strings.",
    "code_example": "",
    "reference": "https://docs.vessl.ai/guides/pipeline/pipeline-steps"
  }
]</examples>""",
    },
]


def generate_faq_samples(
    openai_client: OpenAI, llm_model: str, doc: str, sample_count: int
) -> list[dict]:
    prompt_with_doc = prompt.copy()
    prompt_with_doc.append(
        {
            "role": "user",
            "content": f"Generate {sample_count} faq samples with following link and document as a context: <document>\n{doc}\n</document>",
        }
    )

    completion = openai_client.chat.completions.create(
        model=llm_model,
        messages=prompt_with_doc,
    )

    message = completion.choices[0].message.content
    samples = []
    try:
        # Use regex to extract JSON content from response
        json_str = re.search(r"\[[\S\n ]+\]", message).group()
        samples = json.loads(json_str)
    except Exception as e:
        print(f"Failed to parse the response as JSON: {e}.")
        print(f"Original response: {message}")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate FAQ samples using a specified LLM model."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/data",
        help="Path of the original data",
    )
    parser.add_argument(
        "--llm-endpoint",
        type=str,
        default="https://api.openai.com",
        help="URL of the LLM Endpoint to use for generation",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/llama-3.2-90b-vision-instruct",
        help="Name of the model to use for generation",
    )
    parser.add_argument(
        "--samples-per-doc",
        type=int,
        default=10,
        help="Number of samples to generate per document",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/data/sample.jsonl",
        help="Output file to save the generated samples",
    )
    args = parser.parse_args()

    # Load OpenAI client
    base_url = os.path.join(args.llm_endpoint, "v1")
    client = OpenAI(
        base_url=base_url,
    )

    # Load documents
    documents = glob.glob("*.md", recursive=True)

    # Open CSV file to save results
    with open(args.output_path, mode="w", newline="\n") as file:
        writer = csv.DictWriter(
            file, fieldnames=["question", "answer", "code_example", "reference"]
        )
        writer.writeheader()

        for doc in documents:
            print(f"Original doc: {doc}")
            with open(doc, "r") as f:
                doc_str = f.read()
            samples = generate_faq_samples(
                client, args.model_name, doc_str, args.samples_per_doc
            )
            try:
                writer.writerows(samples)
            except Exception as e:
                print(f"Failed to write the sample to csv: {e}")
                print(f"Original sample: {samples}")
            file.flush()
            for sample in samples[:3]:
                print(f"Question: {sample['question']}")
                print(f"Answer: {sample['answer']}")
                if sample["code_example"]:
                    print(f"Code example: {sample['code_example']}")
                print("\n")
            print("\n...\n")


if __name__ == "__main__":
    main()
