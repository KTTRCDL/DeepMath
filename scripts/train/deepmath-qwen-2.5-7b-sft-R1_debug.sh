#!/usr/bin/env bash

# SFT Qwen2.5-7B on DeepMath-103K (R1 solutions)

DATA_DIR=/hpc2hdd/home/xli026/data/DeepMath-103k
BASE_MODEL=Qwen/Qwen2.5-7B
SAVE_PATH=outputs/qwen2.5-7b-deepmath-sft-R1
NPROC_PER_NODE=1

# Note: Increase data.max_length if your sequences are long; adjust micro_batch_size_per_gpu accordingly.
# Common combos for 7B on 24–80GB GPUs:
#   - max_length=4096, micro_batch_size_per_gpu=1–2
#   - max_length=2048, micro_batch_size_per_gpu=2–4

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files=${DATA_DIR}/train_firstR1.parquet \
  data.val_files=${DATA_DIR}/train_firstR1.parquet \
  data.prompt_key=prompt \
  data.response_key=r1_solution_1 \
  data.max_length=2048 \
  data.truncation='left' \
  data.train_batch_size=1 \
  data.micro_batch_size_per_gpu=1 \
  model.partial_pretrain=${BASE_MODEL} \
  model.enable_gradient_checkpointing=True \
  trainer.default_local_dir=${SAVE_PATH} \
  trainer.project_name=deepmath-sft \
  trainer.experiment_name=qwen2.5-7b-deepmath-sft-R1 \
  trainer.logger=['console'] \
  trainer.total_epochs=1 \
  trainer.default_hdfs_dir=null \
  model.lora_rank=8 \
  model.lora_alpha=32 \
  model.target_modules=all-linear \
  $@

#   data.train_files=${DATA_DIR}/train.parquet \
#   data.val_files=${DATA_DIR}/test.parquet \

# Tips:
# - To train with LoRA, append: model.lora_rank=32 model.lora_alpha=16 model.target_modules=all-linear
# - To use a different base model, pass: BASE_MODEL=Qwen/Qwen2.5-7B $(basename "$0") ...
# - To train on only the first R1 solution, pre-generate a filtered parquet and override TRAIN_FILE/VAL_FILE.
