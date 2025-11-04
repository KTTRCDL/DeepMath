ray start  --head --port=6379  --node-ip-address=127.0.0.1 --num-gpus=2

set -e
set -u

SCRIPT_DIR=$(cd $(dirname $0); pwd)
WORK_DIR=$SCRIPT_DIR/../..
MODEL_DIR=$WORK_DIR/models
DATA_DIR=/hpc2hdd/home/xli026/data/DeepMath-103k
RUN_NAME=deepmath-1.5b
mkdir -p $MODEL_DIR/$RUN_NAME

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{
    "env_vars": {
        "PYTHONUNBUFFERED": "1",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_DISABLE": "0",
        "NCCL_COLLNET_ENABLE": "0",
        "VLLM_ATTENTION_BACKEND": "XFORMERS",
        "CUDA_VISIBLE_DEVICES": "0,1"
    },
    "pip": ["word2number", "timeout_decorator"]
    }' -- PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.rm_system_prompt=True \
    data.train_batch_size=2 \
    data.gen_batch_size=2 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=1000 \
    algorithm.filter_groups.metric=acc \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.27 \
    actor_rollout_ref.actor.max_grad_norm=5.0 \
    actor_rollout_ref.actor.use_token_level_loss=True \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1024 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=2048 \
    custom_reward_function.path=$WORK_DIR/utils/reward_utils/reward_func.py \
    custom_reward_function.name=reward_func \
    custom_reward_function.overlong_buffer.enable=True \
    custom_reward_function.overlong_buffer.len=1024 \
    custom_reward_function.overlong_buffer.penalty_factor=1.0 \
    trainer.project_name=deepmath \
    trainer.experiment_name=$RUN_NAME \
    trainer.run_id=$RUN_NAME \
    trainer.default_local_dir=$MODEL_DIR/$RUN_NAME \
    trainer.logger=['console'] \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.save_rollout=True \
    trainer.test_freq=-1 \
    trainer.total_epochs=2 \
    trainer.total_training_steps=1800 2>&1 | tee -a $MODEL_DIR/$RUN_NAME/train.log

# trainer.logger=['console','wandb'] \
# data.train_batch_size=128 \
# data.gen_batch_size=384 \
# data.max_prompt_length=2048 \
# data.max_response_length=24576 \
# actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
# actor_rollout_ref.actor.ppo_mini_batch_size=2 \