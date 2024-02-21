import argparse
import os
import openai
import time
import statistics

# Set up OpenAI API client
client = openai.OpenAI(base_url=os.getenv('OPENLLM_ENDPOINT', 'http://localhost:3000').rstrip('/') + '/v1', api_key='na')
models = client.models.list()
model = models.data[0].id  # Assuming the first model is the one we want to use

def main():
    parser = argparse.ArgumentParser(description='Load test the OpenAI-compatible LLM API')
    parser.add_argument('--prompt', type=str,
                        default="Share your thoughts on the importance of education in today's society.",
                        help="The prompt to use for the completion")
    args = parser.parse_args()
    prompt = "<user>" + args.prompt.strip() + "</user><assistant>"
    print(f"Prompt: {args.prompt}")
    print("Response:", end=' ')

    start_time = last_token_time = time.time()
    token_latencies = []
    completions = client.completions.create(prompt=prompt.strip(), model=model, max_tokens=1024, stream=True)
    for chunk in completions:
        text = chunk.choices[0].text
        if text:
            print(text, flush=True, end='')

        token_end_time = time.time()
        token_latency = token_end_time - last_token_time  # Measure token latency from the start of the request
        token_latencies.append(token_latency)
        last_token_time = token_end_time

    latency = time.time() - start_time
    avg_token_latency = statistics.mean(token_latencies) if token_latencies else 0
    token_latency_90pct = statistics.quantiles(token_latencies, n=100)[89] if token_latencies else 0
    token_latency_95pct = statistics.quantiles(token_latencies, n=100)[94] if token_latencies else 0
    token_latency_99pct = statistics.quantiles(token_latencies, n=100)[98] if token_latencies else 0

    print(f"\n\n----- Generation Complete! -----")
    print(f"\nRequest Latency: {latency:.2f}s")

    print(f"Avg Token Latency: {round(avg_token_latency * 1000, 3)} milliseconds")
    print(f"90th Percentile Token Latency: {round(token_latency_90pct * 1000, 3)} milliseconds")
    print(f"95th Percentile Token Latency: {round(token_latency_95pct * 1000, 3)} milliseconds")
    print(f"99th Percentile Token Latency: {round(token_latency_99pct * 1000, 3)} milliseconds\n")

if __name__ == "__main__":
    main()
