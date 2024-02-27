# Generating Q&A Dataset from documents using LLM

In this example, we are going to generate question and answer pairs from documents using Llama-index and OpenAI GPT-4 Turbo. Overall generation process is brought from [algorithm of alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/generate_instruction.py).

1. Set your OpenAI API key to the environment variable or `.env` file.
    ```sh
    OPENAI_API_KEY=...
    ```

1. Install dependencies.
    ```sh
    pip install -r requirements
    ```

1. Prepare your docs. In this example, we're going to use docs for VESSL.

1. Generate seed Q&A pairs and review them. The seed Q&A pairs will be saved to `{output_dir}/seed.json
    ```sh
    python generate_seed_qa_pairs.py \
    --output-dir ./outputs \
    --docs-dir ./vessl-docs \
    --model gpt-4-turbo-preview \
    --temperature 1.0
    ```

1. Generate Q&A pairs by running `generate_qa_pairs.py`.
    ```sh
     python generate_qa_pairs.py \
    --output-dir ./outputs \
    --seed-qa-path ./outputs/seed.json \
    --docs-dir ./vessl-docs \
    --n-pairs 500 \
    --model gpt-4-turbo-preview \
    --temperature 1.0
    ```
    The Q&A pairs will be saved to `{output_dir}/synthetic_data.json`.
