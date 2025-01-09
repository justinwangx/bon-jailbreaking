python -m fastchat.serve.controller

CUDA_VISIBLE_DEVICES=0 python -m fastchat.serve.vllm_worker \
    --model-path GraySwanAI/Llama-3-8B-Instruct-RR \
    --device cuda \
    --controller-address http://localhost:21001 \
    --worker-address http://localhost:21002 \
    --port 21002

CUDA_VISIBLE_DEVICES=1 python -m fastchat.serve.vllm_worker \
    --model-path cais/HarmBench-Llama-2-13b-cls \
    --device cuda \
    --controller-address http://localhost:21001 \
    --worker-address http://localhost:21003 \
    --port 21003

python -m fastchat.serve.openai_api_server --host localhost --port 8000