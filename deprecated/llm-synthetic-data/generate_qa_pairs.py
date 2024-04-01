# https://github.com/tatsu-lab/stanford_alpaca/blob/main/generate_instruction.py
import argparse
import json
import os
import random
import re
from functools import partial
from multiprocessing import Pool

import numpy as np
from dotenv import load_dotenv
from rouge_score import rouge_scorer

from query_engine import load_query_engine

load_dotenv()


def encode_prompt(qa_pairs: list[dict[str, str]]) -> str:
    prompt = (
        "You are asked to come up with a set of 20 diverse task instructions. "
        "These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.\n\n"
        "List of 20 tasks:\n"
    )
    qa_template = "###\n" "{idx}. Question: {question}\n" "{idx}. Answer: {answer}\n"

    for idx, qa_pair in enumerate(qa_pairs):
        prompt += qa_template.format(
            idx=idx + 1, question=qa_pair["question"], answer=qa_pair["answer"]
        )

    prompt += "###\n"
    prompt += f"{idx+2}. Question: "

    return prompt


def generate_qa_pairs_from_seed(
    output_dir: str,
    seed_qa_path: str,
    docs_dir: str,
    n_pairs_to_gen: int = 100,
    model_name: str = "gpt-4-turbo-preview",
    temperature: float = 1.0,
    resume_from_path: str = None,
):
    # make output directory
    os.makedirs(output_dir, exist_ok=True)

    # load seed question/answer pairs
    with open(seed_qa_path) as fp:
        seed_qa_pairs = json.load(fp)

    # similarity score and tokenize seed
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    all_questions = [p["question"] for p in seed_qa_pairs]
    all_question_tokens = [scorer._tokenizer.tokenize(q) for q in all_questions]

    # init query engine
    openai_api_kwargs = {"logit_bias": {"50256": -100}, "stop": ["\n21.", "21."]}
    query_engine = load_query_engine(
        docs_dir=docs_dir,
        persist_dir="./storage",
        model=model_name,
        temperature=temperature,
        additional_kwargs=openai_api_kwargs,
    )

    # resuming
    if resume_from_path is None:
        generated_qa_pairs = []
    elif os.path.exists(resume_from_path):
        with open(resume_from_path) as fp:
            generated_qa_pairs = json.load(fp)
    else:
        print(f"{resume_from_path} does not exists.")
        generated_qa_pairs = []

    # starting generation loop
    iter_idx = 0
    consecutive_zero_count = 0

    n_sample_from_seed = 3

    # loop until the sufficient number of pairs are generated or only similar questions are generated
    while len(generated_qa_pairs) < n_pairs_to_gen and consecutive_zero_count < 5:
        # encode prompt
        sampled_pairs = random.sample(seed_qa_pairs, n_sample_from_seed)
        prompt = encode_prompt(sampled_pairs)

        # get question and answer pairs which are single strings
        response = query_engine.query(prompt)
        text = (
            prompt.removesuffix(f"{n_sample_from_seed+1}. Question: ")
            + response.response
        )
        raw_pairs = text.split("###\n")[1:]

        # extract question and answer pairs from string
        pattern = re.compile(r"\d+\. Question: (.*?)\n\d+\. Answer: (.*?)\n?$")
        batch_qa_pairs = []
        for _, raw_pair in enumerate(raw_pairs):
            try:
                question, answer = pattern.findall(raw_pair)[0]
                batch_qa_pairs.append(
                    {"question": question.strip(), "answer": answer.strip()}
                )
            except IndexError:
                print("Error while parsing. Skipping...")
                print(raw_pair)

        batch_len = len(batch_qa_pairs)
        added_len = batch_len

        # compute similarity between new question and the previous questions
        for qa_pair in batch_qa_pairs:
            new_instruction_tokens = scorer._tokenizer.tokenize(qa_pair["question"])
            with Pool(16) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_question_tokens,
                )

            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_questions = {
                all_questions[i]: rouge_scores[i]
                for i in np.argsort(rouge_scores)[-10:][::-1]
            }

            # if there is a similar question, exclude this pair
            if max(rouge_scores) > 0.7:
                added_len -= 1
                continue

            qa_pair["most_similar_questions"] = most_similar_questions
            qa_pair["avg_similarity_score"] = float(np.mean(rouge_scores))
            generated_qa_pairs.append(qa_pair)

            all_questions.append(qa_pair["question"])
            all_question_tokens.append(new_instruction_tokens)

        iter_idx += 1
        print(f"Iteration {iter_idx}")
        print(
            f"{batch_len} pairs generated, {added_len} pairs added, total {len(generated_qa_pairs)} pairs so far"
        )
        print("------")

        if added_len == 0:
            consecutive_zero_count += 1
        else:
            consecutive_zero_count = 0

    if len(generated_qa_pairs) < n_pairs_to_gen and consecutive_zero_count > 0:
        print("Iteration loop finished because of too many consecutive zeros")

    with open(os.path.join(output_dir, "./synthetic_data.json"), "w") as fp:
        synthetic_data = [
            {"question": p["question"], "answer": p["answer"]}
            for p in generated_qa_pairs
        ]
        json.dump(synthetic_data, fp, indent=2)

    with open(os.path.join(output_dir, "./checkpoint.json"), "w") as fp:
        json.dump(generated_qa_pairs, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-qa-path", required=True)
    parser.add_argument("--docs-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    generate_qa_pairs_from_seed(
        args.output_dir,
        args.seed_qa_path,
        args.docs_dir,
        args.n_pairs,
        model_name=args.model,
        temperature=args.temperature,
    )
