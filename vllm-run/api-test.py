import argparse
import os

from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="https://api.openai.com")
parser.add_argument("--api-key", default="token-abc123")
parser.add_argument("--model-name", default="gpt-3.5-turbo")
args = parser.parse_args()

client = OpenAI(
    base_url=os.path.join(args.base_url, "v1"),
    api_key=args.api_key,
)

completion = client.chat.completions.create(
    model=args.model_name,
    messages=[
        {"role": "user", "content": "What is the capital of South Korea?"},
    ],
)

print(completion.choices[0].message)