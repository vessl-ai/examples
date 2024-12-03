import pandas as pd
import json
import requests
import time
import argparse
import os
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# Single constant prompt
SYSTEM_PROMPT = """I will give you user reviews for a product. Generate a json summary of the
reviews, with focus on the common positive and negative aspects across all of
the reviews. Use the exact same output format as in the example (list of
positive highlights, list of negative aspects, summary).
In the summary, you must create json result only with same field like below example, and must follow field order. 

IMPORTANT:
- JSON objects only contains the fields that shown in the below Examples json.
- Return ONLY the JSON object, nothing else.
- Do not add any explanation or additional text.
- Ensure the output is valid JSON that can be parsed by python json.loads()

// Examples json
{
  "product_id": "" // ID of product
  "score": 0 // review score of product
  "raw_review": "" // raw string review text
  "summarized_review": "" // simple summarized review
  "positive_features": [""] // important positive features when summarized review.
  "negative_features": [""] // importantnegative features when summarized review
}
"""

class VLLMProcessor:
    def __init__(self, host: str = "localhost", port: int = 8000, num_workers: int = 4):
        self.system_prompt = SYSTEM_PROMPT
        self.base_url = f"{host}/generate"
        self.num_workers = num_workers
        self.session = requests.Session()

    def fix_escaped_json(self, text: str) -> str:
        """Fix over-escaped JSON strings"""
        text = text.replace('\\\\\\\\', '\\\\')
        text = text.replace('\\\\"', '\\"')
        return text

    def check_vllm_connection(self) -> bool:
        """Test if vLLM server is running and accessible"""
        try:
            response = self.session.post(
                self.base_url,
                json={
                    "prompt": "test",
                    "max_tokens": 10,
                    "temperature": 0,
                },
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of records"""
        processed_batch = []

        for record in batch:
            try:
                # Convert record to JSON string, preserving escapes
                record_str = json.dumps(record, ensure_ascii=False)

                # Create prompt with system prompt and user input
                full_prompt = f"{SYSTEM_PROMPT}\n\nInput:\n{record_str}\n\nOutput:"

                response = self.session.post(
                    self.base_url,
                    json={
                        "prompt": full_prompt,
                        "max_tokens": 1000,
                        "temperature": 0,  # Use deterministic output
                        "stop": ["\n\n"]  # Stop at empty lines
                    },
                    timeout=10
                )
                response.raise_for_status()

                # Extract the generated text from the vLLM response
                response_data = response.json()

                # Handle vLLM response format where text is a list
                if isinstance(response_data, dict) and 'text' in response_data:
                    text_content = response_data['text']
                    if isinstance(text_content, list):
                        result = text_content[0]  # Take the first element if it's a list
                    else:
                        result = text_content
                else:
                    print(f"\nWarning: Unexpected response format: {response_data}")
                    processed_batch.append(record)
                    continue

                # Find the JSON content in the response
                try:
                    # Try to find JSON content between Output: and the next newline
                    output_start = result.find('Output:')
                    if output_start != -1:
                        result = result[output_start + 7:].strip()

                    # Fix escaped characters and parse
                    fixed_result = self.fix_escaped_json(result)
                    processed_record = json.loads(fixed_result)
                except json.JSONDecodeError as e:
                    # If that fails, try cleaning up the response
                    try:
                        # Remove any markdown code blocks or extra text
                        cleaned_result = result.replace('```json', '').replace('```', '').strip()
                        # Try to extract just the JSON part
                        json_start = cleaned_result.find('{')
                        json_end = cleaned_result.rfind('}') + 1
                        if json_start != -1 and json_end != -1:
                            cleaned_result = cleaned_result[json_start:json_end]
                        fixed_cleaned_result = self.fix_escaped_json(cleaned_result)
                        processed_record = json.loads(fixed_cleaned_result)
                    except json.JSONDecodeError:
                        print(f"\nWarning: JSON parsing failed: {str(e)}\nResponse: {result[:200]}...")
                        processed_record = record

                processed_batch.append(processed_record)

            except Exception as e:
                print(f"\nWarning: Error in batch processing: {str(e)}")
                processed_batch.append(record)

        return processed_batch


class CSVProcessor:
    def __init__(self, input_file: str, output_file: str, vllm_processor: VLLMProcessor,
                 batch_size: int = 5):
        self.input_file = input_file
        self.output_file = output_file
        self.vllm_processor = vllm_processor
        self.batch_size = batch_size
        self.max_rows = 10

    def clean_text(self, text: str) -> str:
        """Clean text while preserving necessary escapes"""
        if not isinstance(text, str):
            return str(text) if text is not None else ""

        # Basic cleaning while preserving escapes
        text = str(text).strip()
        text = text.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
        # Don't modify existing escapes
        return text

    def clean_record(self, record: Dict) -> Dict:
        """Clean all values in a record"""
        return {k: self.clean_text(v) for k, v in record.items()}

    def process_batch_with_executor(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of records with cleaning"""
        try:
            cleaned_batch = [self.clean_record(record) for record in batch]
            return self.vllm_processor.process_batch(cleaned_batch)
        except Exception as e:
            print(f"Error in batch executor: {str(e)}")
            return batch

    def process_csv(self) -> bool:
        try:
            # Read specified number of rows
            if self.max_rows:
                df = pd.read_csv(self.input_file, nrows=self.max_rows, 
                               encoding='utf-8', on_bad_lines='skip', dtype=str)
                total_rows = len(df)
            else:
                # Count total rows if no limit specified
                total_rows = sum(1 for _ in pd.read_csv(self.input_file, chunksize=1000))
                df = pd.read_csv(self.input_file, encoding='utf-8', 
                               on_bad_lines='skip', dtype=str)
            
            print(f"Processing {total_rows} records...")
            
            # Convert to records
            records = df.to_dict(orient='records')
            processed_records = []
            
            # Create batches
            batches = [records[i:i + self.batch_size] 
                      for i in range(0, len(records), self.batch_size)]
            
            # Process with progress bar
            with ThreadPoolExecutor(max_workers=self.vllm_processor.num_workers) as executor:
                with tqdm(total=total_rows, desc="Processing records") as pbar:
                    futures = [
                        executor.submit(self.process_batch_with_executor, batch)
                        for batch in batches
                    ]
                    
                    for future in as_completed(futures):
                        try:
                            batch_results = future.result()
                            processed_records.extend(batch_results)
                            pbar.update(len(batch_results))
                            
                            # Save progress periodically
                            if len(processed_records) % 100 == 0:
                                self.save_progress(processed_records, f"{self.output_file}.part")
                        except Exception as e:
                            print(f"Error processing batch: {str(e)}")
            
            # Save final results
            self.save_progress(processed_records, self.output_file)
            
            # Clean up partial file
            part_file = f"{self.output_file}.part"
            if os.path.exists(part_file):
                os.remove(part_file)
            
            return True
            
        except Exception as e:
            print(f"Error processing CSV: {str(e)}")
            return False

    def save_progress(self, records: list, filename: str):
        """Save progress to file with error handling"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save progress to {filename}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Process CSV to JSON using vLLM')

    parser.add_argument('--input', '-i',
                      required=True,
                      help='Input CSV file path')

    parser.add_argument('--output', '-o',
                      required=True,
                      help='Output JSON file path')

    parser.add_argument('--workers', '-w',
                      type=int,
                      default=4,
                      help='Number of worker threads')

    parser.add_argument('--batch-size', '-b',
                      type=int,
                      default=5,
                      help='Batch size for processing')

    parser.add_argument('--debug', '-d',
                      action='store_true',
                      help='Print debug information')

    parser.add_argument('--host',
                      default='localhost',
                      help='vLLM server host')

    parser.add_argument('--port',
                      type=int,
                      default=8000,
                      help='vLLM server port')

    args = parser.parse_args()

    vllm_processor = VLLMProcessor(
        host=args.host,
        port=args.port,
        num_workers=args.workers
    )

    if not vllm_processor.check_vllm_connection():
        print(f"Error: Cannot connect to vLLM server at {args.host}:{args.port}")
        print("Please ensure the vLLM server is running. You can start it with:")
        print(f"python -m vllm.entrypoints.api_server --model <your-model> --port {args.port}")
        return

    csv_processor = CSVProcessor(args.input, args.output, vllm_processor,
                               batch_size=args.batch_size)

    print(f"Starting CSV to JSON processing with vLLM...")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Number of workers: {args.workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"vLLM server: {args.host}:{args.port}")

    if args.debug:
        print("\nSystem Prompt:")
        print(SYSTEM_PROMPT)
        print("\nStarting processing...")

    success = csv_processor.process_csv()

    if success:
        print("Processing completed successfully!")
    else:
        print("Processing failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
