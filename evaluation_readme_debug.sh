# eval "$(mamba shell hook --shell bash)"
# mamba activate deepmath && VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ATTENTION_BACKEND=XFORMERS VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python3 uni_eval.py \
#     --base_model ~/data/zwhe99/DeepMath-Zero-7B \
#     --chat_template_name default \
#     --system_prompt_name simplerl \
#     --output_dir result/eval/aime90/1 \
#     --bf16 True \
#     --tensor_parallel_size 1 \
#     --data_id zwhe99/aime90 \
#     --split 2024 \
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

# DATASET_INFO = {
#     "zwhe99/MATH": {
#         "default_split": "math500",
#         "problem_key": "problem",
#         "answer_key": "expected_answer",
#         "category_keys": ["level", "type"]
#     },
#     "zwhe99/aime90": {
#         "default_split": "2024",
#         "problem_key": "problem",
#         "answer_key": "expected_answer",
#     },
#     "zwhe99/amc23": {
#         "default_split": "test",
#         "problem_key": "question",
#         "answer_key": "answer",
#     },
#     "zwhe99/simplerl-minerva-math": {
#         "default_split": "test",
#         "problem_key": "problem",
#         "answer_key": "answer",
#     },
#     "math-ai/aime25": {
#         "default_split": "test",
#         "problem_key": "problem",
#         "answer_key": "answer",
#     },
#     "zwhe99/simplerl-OlympiadBench": {
#         "default_split": "test",
#         "problem_key": "question",
#         "answer_key": "final_answer",
#     },
#     "zwhe99/gpqa_diamond_mc": {
#         "default_split": "test",
#         "problem_key": "problem",
#         "answer_key": "solution",
#         "category_keys": ["domain"]
#     },
#     "zwhe99/pm-en": {
#         "default_split": "test",
#         "problem_key": "question",
#         "answer_key": "answer",
#         "category_keys": ["level"]
#     }
# }