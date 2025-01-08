import random
import string
import time
from typing import List

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

def generate_random_string(length: int = 20) -> str:
    """Generate a random string of fixed length"""
    return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))

def main():
    # Initialize model
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,  # Adjust based on number of GPUs
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=8192,  # Adjust based on available GPU memory
        device="cuda",
    )
    print("Created.")

    # Generate random prompts
    n_queries = 10000
    prompts = [generate_random_string() for _ in range(n_queries)]
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=50,
    )

    # Time the inference
    start_time = time.time()
    
    # Process in batches using tqdm
    outputs = []
    batch_size = 32  # Adjust based on GPU memory
    
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i + batch_size]
        outputs.extend(llm.generate(batch, sampling_params))
    
    end_time = time.time()
    
    # Print statistics
    total_time = end_time - start_time
    queries_per_second = n_queries / total_time
    
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Queries per second: {queries_per_second:.2f}")
    print(f"Average time per query: {(total_time/n_queries)*1000:.2f} ms")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()