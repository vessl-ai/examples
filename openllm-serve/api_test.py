import argparse
import os
import openai
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up OpenAI API client
client = openai.OpenAI(base_url=os.getenv('OPENLLM_ENDPOINT', 'http://localhost:3000') + '/v1', api_key='na')
models = client.models.list()
model = models.data[0].id  # Assuming the first model is the one we want to use

# Function to handle individual request and measure token latencies
def handle_request(idx, prompt, model, max_tokens):
    token_latencies = []
    total_tokens = 0
    start_request_time = last_token_time = time.time()  # Start time for the entire request
    try:
        completions = client.completions.create(prompt=prompt.strip(), model=model, max_tokens=max_tokens, stream=True)
        full_response = ""
        for chunk in completions:
            text = chunk.choices[0].text
            if text:
                full_response += text
                token_end_time = time.time()
                token_latency = token_end_time - last_token_time  # Measure token latency from the start of the request
                token_latencies.append(token_latency)
                total_tokens += 1
                last_token_time = token_end_time  # Reset start time for the next token
        request_latency = time.time() - start_request_time  # Total request time
        return {"success": True, "index": idx, "response": full_response, "token_latencies": token_latencies, "total_tokens": total_tokens, "request_latency": request_latency}
    except Exception as e:
        return {"success": False, "index": idx, "error": str(e)}

# Function to make requests and record metrics
def benchmark_api(prompts_file, max_tokens=64, num_iterations=10, num_workers=5):
    # Load prompts from file
    with open(prompts_file, 'r') as file:
        prompts = file.readlines()

    # Augment or split prompts to reach num_iterations
    num_prompts = len(prompts)
    if num_prompts < num_iterations:
        prompts = (prompts * (num_iterations // num_prompts + 1))[:num_iterations]
    elif num_prompts > num_iterations:
        prompts = prompts[:num_iterations]

    token_latencies = []
    request_latencies = []  # List to store request latencies

    # Make requests in parallel
    print(f"Making {num_iterations} requests in parallel with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(handle_request, idx+1, prompt, model, max_tokens) for idx, prompt in enumerate(prompts)]

        successful_requests = 0
        total_tokens = 0
        for future in as_completed(futures):
            result = future.result()
            if result["success"]:
                successful_requests += 1
                token_latencies.extend(result['token_latencies'])
                total_tokens += result['total_tokens']
                request_latencies.append(result['request_latency'])  # Collect total request latency
                print(f"[{result['index']}] Success: {result['total_tokens']} output tokens")
            else:
                print(f"[{result['index']}] Failure: {result['error']}")

    # Calculate metrics
    avg_token_latency = statistics.mean(token_latencies) if token_latencies else 0
    avg_request_latency = statistics.mean(request_latencies) if request_latencies else 0
    throughput = total_tokens / sum(request_latencies) if request_latencies else 0

    # Calculate percentiles for token latencies
    token_latency_90pct = statistics.quantiles(token_latencies, n=100)[89] if token_latencies else 0
    token_latency_95pct = statistics.quantiles(token_latencies, n=100)[94] if token_latencies else 0
    token_latency_99pct = statistics.quantiles(token_latencies, n=100)[98] if token_latencies else 0

    # Calculate percentiles for request latencies
    request_latency_90pct = statistics.quantiles(request_latencies, n=100)[89] if request_latencies else 0
    request_latency_95pct = statistics.quantiles(request_latencies, n=100)[94] if request_latencies else 0
    request_latency_99pct = statistics.quantiles(request_latencies, n=100)[98] if request_latencies else 0

    # Print metrics - cut to 2 decimal places
    print(f"Avg Token Latency: {round(avg_token_latency * 1000, 3)} milliseconds")
    print(f"90th Percentile Token Latency: {round(token_latency_90pct * 1000, 3)} milliseconds")
    print(f"95th Percentile Token Latency: {round(token_latency_95pct * 1000, 3)} milliseconds")
    print(f"99th Percentile Token Latency: {round(token_latency_99pct * 1000, 3)} milliseconds\n")
    print(f"Avg Request Latency: {round(avg_request_latency, 3)} seconds")
    print(f"90th Percentile Request Latency: {round(request_latency_90pct, 3)} seconds")
    print(f"95th Percentile Request Latency: {round(request_latency_95pct, 3)} seconds")
    print(f"99th Percentile Request Latency: {round(request_latency_99pct, 3)} seconds\n")
    print(f"Successful Requests: {successful_requests}/{num_iterations}")
    print(f"Token Throughput: {round(throughput, 3)} tokens/second")

def main():
    parser = argparse.ArgumentParser(description='Load test the OpenAI-compatible LLM API')
    parser.add_argument('--prompts-file', type=str, default='./sample_prompts.txt', help='Path to a file with prompts to use for the load test')
    parser.add_argument('--max-tokens', type=int, default=1024, help='Maximum number of tokens to generate per request')
    parser.add_argument('--num-iterations', type=int, default=100, help='Number of requests to make')
    parser.add_argument('--num-workers', type=int, default=5, help='Number of parallel workers for requests')
    args = parser.parse_args()

    print(f"Test configuration: {args}")
    benchmark_api(args.prompts_file, args.max_tokens, args.num_iterations, args.num_workers)

if __name__ == "__main__":
    main()
