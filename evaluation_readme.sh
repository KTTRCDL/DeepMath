# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ATTENTION_BACKEND=XFORMERS VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python3 uni_eval.py \
#     --base_model ~/data/zwhe99/DeepMath-Zero-7B \
#     --chat_template_name default \
#     --system_prompt_name simplerl \
#     --output_dir result/eval/MATH/4 \
#     --bf16 True \
#     --tensor_parallel_size 1 \
#     --data_id zwhe99/MATH \
#     --split math500 \
#     --max_model_len 32768 \
#     --temperature 0.6 \
#     --top_p 0.95 \
#     --n 16

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ATTENTION_BACKEND=XFORMERS VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python3 uni_eval.py \
    --base_model Qwen/Qwen2.5-7B \
    --chat_template_name default \
    --system_prompt_name simplerl \
    --output_dir result/eval/MATH/Qwen2.5-7B_16 \
    --bf16 True \
    --tensor_parallel_size 1 \
    --data_id zwhe99/MATH \
    --split math500 \
    --max_model_len 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --n 16