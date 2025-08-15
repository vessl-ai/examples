#!/usr/bin/env python3
"""
Simple streaming test script for GPT-OSS API
"""
import openai
from datetime import datetime

# Configure client for your GPT-OSS server
client = openai.OpenAI(
    base_url="{YOUR_API_URL}/v1",
    api_key="dummy"  # Not needed for our server
)

# OpenAI Harmony format system prompt
current_date = datetime.now().strftime("%Y-%m-%d")
system_prompt = f"""
<|start|>system<|message|>You are VESSL-GPT, a large language model fine-tuned on VESSL.
Knowledge cutoff: 2024-06
Current date: {current_date}
Reasoning: low
# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>
"""

def test_streaming():
    print("🚀 Testing GPT-OSS Streaming...")
    print("=" * 50)

    try:
        stream = client.chat.completions.create(
            model="gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Write a haiku about artificial intelligence"}
            ],
            max_tokens=1024,
            temperature=0.7,
            stream=True
        )

        print("🤖 GPT-OSS: ", end="", flush=True)

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)

        print("\n" + "=" * 50)
        print("✅ Streaming test completed!")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_streaming()
