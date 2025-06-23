#!/usr/bin/env python3
from vllm import LLM, SamplingParams
import argparse
import sys
import os
import time

def main():
    parser = argparse.ArgumentParser(
        description="vLLM launcher for different parallelism modes (with optional overrides)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "tp", "dp", "pp_tp"],
        required=True,
        help="Run mode: single, tp, dp (currently not working), or pp_tp"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model name or path"
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=None,
        help="Override the pipeline parallel size"
    )
    args = parser.parse_args()

    # Fixed configs for each mode
    configs = {
        "single": {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "description": "Single GPU, Llama 3 8B Instruct"
        },
        "tp": {
            "model": "meta-llama/Llama-3.3-70B-Instruct",
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "description": "Tensor Parallel (4-way), Llama 3.3 70B Instruct"
        },
        "dp": {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 4,
            "description": "Data Parallel (4-way), Llama 3 8B Instruct"
        },
        "pp_tp": {
            "model": "meta-llama/Llama-3.3-70B-Instruct",
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 2,
            "data_parallel_size": 1,
            "description": "Pipeline(2) + Tensor(4) Llama 3.3 70B Instruct"
        }
    }

    if args.mode not in configs:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        sys.exit(1)

    cfg = configs[args.mode]

    # Apply overrides if provided
    model = args.model if args.model is not None else cfg["model"]
    pipeline_parallel_size = (
        args.pipeline_parallel_size
        if args.pipeline_parallel_size is not None
        else cfg["pipeline_parallel_size"]
    )

    print(f"Launching mode: {args.mode}")
    print(f"  Description: {cfg['description']}")
    print(f"  Model: {model}")
    print(f"  TP size: {cfg['tensor_parallel_size']}, PP size: {pipeline_parallel_size}, DP size: {cfg['data_parallel_size']}")

    # Launch LLM with (possibly overridden) parameters
    llm = LLM(
        model=model,
        tensor_parallel_size=cfg["tensor_parallel_size"],
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=cfg["data_parallel_size"],
        max_model_len=2048
    )
    sampling_params = SamplingParams(
        max_tokens=128,
        temperature=1.0,
        stop=["<|eot_id|>"]
    )

    # Example prompts
    system_prompt = (
        "You are a helpful, knowledgeable AI assistant. "
        "Answer as clearly and concisely as possible. "
    )

    prompts = [
        "What is photosynthesis?",
        "Who wrote 'Romeo and Juliet'?",
        "What is the capital of Japan?",
        "Define gravity.",
        "What is the largest planet in our solar system?",
        "Who painted the Mona Lisa?",
        "What does DNA stand for?",
        "How many continents are there?",
        "What is the boiling point of water?",
        "Who invented the telephone?",
        "What is the main ingredient in bread?",
        "What year did World War II end?",
        "What is the smallest prime number?",
        "Who was the first person on the moon?",
        "What is the chemical symbol for gold?",
        "What is the freezing point of water?",
        "Who is the author of 'Harry Potter'?",
        "What is the tallest mountain in the world?",
        "What is the square root of 64?",
        "What language is spoken in Brazil?",
        "What is the currency of the United Kingdom?",
        "Who discovered penicillin?",
        "What is the fastest land animal?",
        "What gas do plants absorb from the air?",
        "How many sides does a hexagon have?",
        "What is the largest ocean on Earth?",
        "Who is known as the 'Father of Computers'?",
        "What is the hardest natural substance?",
        "What is the main function of the lungs?",
        "What planet is known as the Red Planet?",
        "Who wrote 'The Odyssey'?",
        "What is the process by which water changes from liquid to gas?",
    ]


    template = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "{system_prompt}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        "{user_prompt}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    formatted_prompts = [template.format(system_prompt=system_prompt, user_prompt=p) for p in prompts]

    # Generate output
    start = time.time()
    outputs = llm.generate(formatted_prompts, sampling_params)
    end = time.time()
    
    # Count total generated tokens
    total_tokens = 0
    for output, prompt in zip(outputs, prompts):
        # print("Prompt:", prompt)
        # print("Output:", output.outputs[0].text.strip())
        for candidate in output.outputs:
            total_tokens += len(candidate.token_ids)

    elapsed = end - start
    toks_per_sec = total_tokens / elapsed if elapsed > 0 else 0

    print(f"Total generated tokens: {total_tokens}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Throughput: {toks_per_sec:.2f} tokens/sec")

if __name__ == "__main__":
    main()