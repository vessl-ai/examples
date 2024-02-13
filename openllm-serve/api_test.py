# Usage: python3 api_test.py --prompts-file ./sample_prompts.txt --max-tokens 64 --num-iterations 10

import argparse
import os
import openai
import time
import statistics

# Set up OpenAI API client
client = openai.OpenAI(base_url=os.getenv('OPENLLM_ENDPOINT', 'http://localhost:3000') + '/v1', api_key='na')
models = client.models.list()
model = models.data[0].id  # Assuming the first model is the one we want to use

# Function to make requests and record metrics
def benchmark_api(prompts_file, max_tokens=64, num_iterations=10):
    # Load prompts from file
    with open(prompts_file, 'r') as file:
        prompts = file.readlines()

    # Augment or split prompts to reach num_iterations
    num_prompts = len(prompts)
    if num_prompts < num_iterations:
        prompts = (prompts * (num_iterations // num_prompts + 1))[:num_iterations]
    elif num_prompts > num_iterations:
        prompts = prompts[:num_iterations]

    # Initialize metrics lists
    request_latencies = []
    token_latencies = []
    successful_requests = 0
    total_tokens = 0

    # Make requests
    print(f"Making {num_iterations} requests...")
    for prompt in prompts:
        start_time = token_time = time.time()
        try:
            completions = client.completions.create(prompt=prompt.strip(), model=model, max_tokens=max_tokens, stream=True)
            full_response = ""
            for chunk in completions:
                token_end_time = time.time()
                token_latency = token_end_time - token_time
                token_latencies.append(token_latency)
                text = chunk.choices[0].text
                if text:
                    full_response += text
                token_time = token_end_time
                total_tokens += 1
            print(f"* Prompt: {prompt.strip()}")
            print(f"* Response: {full_response}")
            successful_requests += 1
        except Exception as e:
            print(f"Error: {e}")
        end_time = time.time()

        # Record latency
        latency = end_time - start_time
        request_latencies.append(latency)

    # Calculate metrics
    avg_latency = statistics.mean(request_latencies)
    pct_90_latency = statistics.quantiles(request_latencies, n=100)[90]
    pct_95_latency = statistics.quantiles(request_latencies, n=100)[95]
    avg_token_latency = statistics.mean(token_latencies)
    pct_90_token_latency = statistics.quantiles(token_latencies, n=100)[90]
    pct_95_token_latency = statistics.quantiles(token_latencies, n=100)[95]
    throughput = total_tokens / sum(token_latencies)

    # Print metrics - cut to 2 decimal places
    print(f"Avg Request Latency: {round(avg_latency*1000, 3)} milliseconds")
    print(f"90th Percentile Request Latency: {round(pct_90_latency*1000, 3)} milliseconds")
    print(f"95th Percentile Request Latency: {round(pct_95_latency*1000, 3)} milliseconds")
    print(f"Avg Output Token Latency: {round(avg_token_latency*1000, 3)} milliseconds")
    print(f"90th Percentile Output Token Latency: {round(pct_90_token_latency*1000, 3)} milliseconds")
    print(f"95th Percentile Output Token Latency: {round(pct_95_token_latency*1000, 3)} milliseconds")
    print(f"Successful Requests: {successful_requests}/{len(prompts)}")
    print(f"Token Throughput: {round(throughput, 3)} tokens/second")


def main():
    parser = argparse.ArgumentParser(description='Load test the OpenLLM API')
    parser.add_argument('--prompts-file', type=str, help='Path to a file with prompts to use for the load test')
    parser.add_argument('--max-tokens', type=int, default=64, help='Maximum number of tokens to generate per request')
    parser.add_argument('--num-iterations', type=int, default=10, help='Number of requests to make')
    args = parser.parse_args()

    benchmark_api(args.prompts_file, args.max_tokens, args.num_iterations)

if __name__ == "__main__":
    main()
