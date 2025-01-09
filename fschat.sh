python -m fastchat.serve.controller

python -m fastchat.serve.vllm_worker \
    --model-path GraySwanAI/Llama-3-8B-Instruct-RR \
    --device cuda:0

python -m fastchat.serve.vllm_worker \
    --model-path cais/HarmBench-Llama-2-13b-cls \
    --device cuda:1

python -m fastchat.serve.openai_api_server --host localhost --port 8000