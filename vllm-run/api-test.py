import os

from openai import OpenAI

base_url = os.getenv("BASE_URL", "https://api.openai.com")
api_key = os.getenv("API_KEY", "token-abc123")
model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

client = OpenAI(
    base_url=os.path.join(base_url, "v1"),
    api_key=api_key,
)

completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "user", "content": "What is the capital of South Korea?"},
    ],
)

print(completion.choices[0].message)