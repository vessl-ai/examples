import csv
import os
import argparse
import re
import json

from cryptography.hazmat.primitives.serialization import load_pem_private_key
from llama_index.readers.web import SimpleWebPageReader
from snowflake.snowpark import Session
from snowflake.cortex import Complete

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
        """
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
]</examples>"""
    },
]

def generate_faq_samples(session, doc, sample_count, llm_model="llama3.1-8b"):
    prompt_with_doc = prompt.copy()
    prompt_with_doc.append({
        "role": "user",
        "content": f"Generate {sample_count} faq samples with following link and document as a context: <link>{doc.id_}</link><document>\n{doc.text}\n</document>"
    })
    response = Complete(
        llm_model,
        prompt_with_doc,
        options={
            "temperature": 0.7,
            "max_tokens": 32000,
        },
        session=session,
    )

    message = eval(response)["choices"][0]["messages"]
    samples = []
    try:
        # Use regex to extract JSON content from response
        json_str = re.search(r'\[[\S\n ]+\]', eval(response)["choices"][0]["messages"]).group()
        samples = json.loads(json_str)
    except Exception as e:
        print(f"Failed to parse the response as JSON: {e}.")
        print(f"Original response: {message}")
    return samples

def main():
    parser = argparse.ArgumentParser(description="Generate FAQ samples using a specified LLM model.")
    parser.add_argument('--model-name', type=str, default="llama3.1-405b", help="Name of the model to use for generation")
    parser.add_argument('--samples-per-doc', type=int, default=10, help="Number of samples to generate per document")
    parser.add_argument('--output-file', type=str, default="samples.csv", help="Output file to save the generated samples")
    args = parser.parse_args()

    # Initialize session
    account = os.environ["SNOWFLAKE_ACCOUNT"]
    user = os.environ["SNOWFLAKE_USER"]
    private_key = load_pem_private_key(os.environ["SNOWFLAKE_PRIVATE_KEY"].replace("\\n", "\n").encode(), password=None)
    warehouse = os.environ["SNOWFLAKE_WAREHOUSE"]

    connection_parameters = {
        "account": account,
        "user": user,
        "private_key": private_key,
        "warehouse": warehouse
    }
    session = Session.builder.configs(connection_parameters).create()

    # Load documents
    documents = SimpleWebPageReader(html_to_text=True).load_data(
        [
            "https://docs.vessl.ai/guides/get-started/llama3-deployment",
            "https://docs.vessl.ai/guides/get-started/serverless-tgi-deployment",
            "https://docs.vessl.ai/guides/serve/overview",
            "https://docs.vessl.ai/guides/serve/create-a-service",
            "https://docs.vessl.ai/guides/serve/managing-api-keys",
            "https://docs.vessl.ai/guides/serve/migration",
            "https://docs.vessl.ai/guides/serve/overview-dashboard",
            "https://docs.vessl.ai/guides/serve/revisions",
            "https://docs.vessl.ai/guides/serve/logs",
            "https://docs.vessl.ai/guides/serve/metrics",
            "https://docs.vessl.ai/guides/serve/settings",
            "https://docs.vessl.ai/reference/yaml/serve-yaml",
            "https://docs.vessl.ai/reference/cli/serve",
            "https://docs.vessl.ai/guides/pipeline/overview",
            "https://docs.vessl.ai/guides/pipeline/create-new-pipeline",
            "https://docs.vessl.ai/guides/pipeline/revisions",
            "https://docs.vessl.ai/guides/pipeline/pipeline-steps",
            "https://docs.vessl.ai/guides/pipeline/pipeline-variables",
            "https://docs.vessl.ai/guides/pipeline/executions",
            "https://docs.vessl.ai/guides/pipeline/triggers"
        ]
    )

    # Open CSV file to save results
    with open(args.output_file, mode='w', newline='\n') as file:
        writer = csv.DictWriter(file, fieldnames=["question", "answer", "code_example", "reference"])
        writer.writeheader()

        for doc in documents:
            print(f"Original doc: {doc.id_}")
            samples = generate_faq_samples(session, doc, args.samples_per_doc, llm_model=args.model_name)
            try:
                writer.writerows(samples)
            except Exception as e:
                print(f"Failed to write the sample to csv: {e}")
                print(f"Original sample: {samples}")
            file.flush()
            for sample in samples[:3]:
                print(f"Question: {sample['question']}")
                print(f"Answer: {sample['answer']}")
                if sample['code_example']:
                    print(f"Code example: {sample['code_example']}")
                print("\n")
            print("\n...\n")

    session.close()

if __name__ == "__main__":
    main()
