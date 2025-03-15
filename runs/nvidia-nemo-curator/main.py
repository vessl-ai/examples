import argparse
import asyncio
import json
import os

from openai import AsyncOpenAI
from nemo_curator import AsyncOpenAIClient
from nemo_curator.synthetic import AsyncNemotronGenerator


async def generate_math_questions(
    llm_endpoint,
    model_name,
    n_topics,
    n_subtopics,
    n_questions,
    output_file,
    school_level="university",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1024,
    max_concurrent_requests=10
):
    """
    Use NeMo Curator to generate math questions and save them as a jsonl file
    """
    print(f"LLM endpoint: {llm_endpoint}")
    print(f"Model name: {model_name}")
    print(f"Number of topics to generate: {n_topics}")
    print(f"Number of subtopics per topic: {n_subtopics}")
    print(f"Number of questions per subtopic: {n_questions}")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Environment variable OPENAI_API_KEY is not set. Please set it using the command 'export OPENAI_API_KEY=your_api_key'.")
    
    openai_client = AsyncOpenAI(
        base_url=os.path.join(llm_endpoint, "v1"),
        api_key=api_key
    )
    
    curator_client = AsyncOpenAIClient(openai_client)
    
    generator = AsyncNemotronGenerator(curator_client, max_concurrent_requests=max_concurrent_requests)
    
    model_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    
    print("Generating math questions using run_math_pipeline...")
    
    question_titles = await generator.run_math_pipeline(
        n_macro_topics=n_topics,
        school_level=school_level,
        n_subtopics=n_subtopics,
        n_openlines=n_questions,
        model=model_name,
        base_model_kwargs=model_kwargs,
        conversion_model_kwargs=model_kwargs,
        ignore_conversion_failure=True
    )
    
    print(f"Generated question titles: {question_titles}")
    
    math_data = []
    
    if os.path.exists(output_file):
        os.remove(output_file)
    
    for question_idx, question_title in enumerate(question_titles):
        print(f"Processing question {question_idx+1}/{len(question_titles)}: {question_title}")
        
        problem_prompt = f"""Please provide a detailed mathematical problem based on the following title:
        
Title: {question_title}

Your response should include:
1. A detailed problem statement
2. Necessary context and background information
3. Step-by-step solution with explanations
4. Final answer

Format your response in a clear, educational manner suitable for {school_level}-level mathematics."""

        responses = await curator_client.query_model(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": problem_prompt,
                }
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        problem_text = responses[0] if responses else "문제 생성에 실패했습니다."
        
        question_data = {
            "question_title": question_title,
            "question_text": problem_text,
        }
        
        math_data.append(question_data)
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(question_data, ensure_ascii=False) + '\n')
    
    print(f"Total {len(math_data)} math questions generated and saved to {output_file}")
    return math_data


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate math questions using NeMo Curator')
    
    parser.add_argument('--endpoint', type=str, required=True,
                        help='LLM endpoint URL')
    parser.add_argument('--model', type=str, required=True,
                        help='LLM model name')
    parser.add_argument('--topics', type=int, default=3,
                        help='Number of topics to generate')
    parser.add_argument('--subtopics', type=int, default=2,
                        help='Number of subtopics per topic')
    parser.add_argument('--questions', type=int, default=3,
                        help='Number of questions per subtopic')
    parser.add_argument('--output', type=str, default='math_questions.jsonl',
                        help='Output JSONL file path')
    parser.add_argument('--school-level', type=str, default='university',
                        choices=['elementary', 'middle', 'high', 'university'],
                        help='School level')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='LLM temperature parameter')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='LLM top-p parameter')
    parser.add_argument('--max-tokens', type=int, default=1024,
                        help='LLM maximum token count')
    parser.add_argument('--max-concurrent', type=int, default=10,
                        help='Maximum concurrent requests')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    asyncio.run(generate_math_questions(
        llm_endpoint=args.endpoint,
        model_name=args.model,
        n_topics=args.topics,
        n_subtopics=args.subtopics,
        n_questions=args.questions,
        output_file=args.output,
        school_level=args.school_level,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_concurrent_requests=args.max_concurrent
    ))
