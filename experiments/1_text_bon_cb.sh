#!/bin/bash
set -eou pipefail

input_file_path=./data/direct_request.jsonl

model="Llama-3-8B-Instruct-RR"

temperature=1.0

# note that overall N = num_concurrent_k * n_steps
num_concurrent_k=50 # in paper this is 60
n_steps=200 # in paper this is 120
request_ids=$(seq 0 158) # This is a set of easy ids for quick replication, for full set use $(seq 0 158)

# turn down to avoid rate limiting
gemini_num_threads=10

# baseline runs
baseline_temperature=1.0
n_samples=40 # in paper this is 5000

model_str=${model//\//-}

# run bon jailbreak for each specific id
for choose_specific_id in $request_ids; do

    output_dir=./exp/bon/text/${model_str}/${choose_specific_id}

    # if [ -f $output_dir/done_$n_steps ]; then
    #     echo "Skipping $output_dir because it is already done"
    #     continue
    # fi

    python -m bon.attacks.run_text_bon \
        --input_file_path $input_file_path \
        --output_dir $output_dir \
        --enable_cache False \
        --gemini_num_threads $gemini_num_threads \
        --openai_base_url http://localhost:8000/v1 \
        --lm_model $model \
        --lm_temperature $temperature \
        --choose_specific_id $choose_specific_id \
        --num_concurrent_k $num_concurrent_k \
        --n_steps $n_steps \
        --force_run True \
        --run_batch True
done