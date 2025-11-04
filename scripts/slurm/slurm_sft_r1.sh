#!/bin/bash
#SBATCH -p i64m1tga800u         # 指定GPU队列
#SBATCH -o log/train/deepmath-qwen-2.5-7b-sft-R1/output_%j.txt  # 指定作业标准输出文件，%j为作业号
#SBATCH -e log/train/deepmath-qwen-2.5-7b-sft-R1/err_%j.txt    # 指定作业标准错误输出文件
#SBATCH -n 5            # 指定CPU总核心数
#SBATCH --gres=gpu:1    # 指定GPU卡数
#SBATCH -D /hpc2hdd/home/xli026/project/DeepMath        # 指定作业执行路径为/apps

# 以下是作业要执行的命令
echo "Job started at $(date)"
# Start the evaluation script
module load cuda/12.4
eval "$(mamba shell hook --shell bash)"
mamba activate deepmath 
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

DATA_DIR=/hpc2hdd/home/xli026/data/DeepMath-103k
BASE_MODEL=Qwen/Qwen2.5-7B
SAVE_PATH=outputs/qwen2.5-7b-deepmath-sft-R1
NPROC_PER_NODE=1

# max length = 10240 + 2048 (prompt + response)
torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files=${DATA_DIR}/train.parquet \
  data.val_files=${DATA_DIR}/test.parquet \
  data.prompt_key=prompt \
  data.response_key=r1_solution_1 \
  data.response_key_val=solution \
  data.max_length=12288 \
  data.train_batch_size=32 \
  data.micro_batch_size_per_gpu=32 \
  model.partial_pretrain=${BASE_MODEL} \
  model.enable_gradient_checkpointing=True \
  trainer.default_local_dir=${SAVE_PATH} \
  trainer.project_name=deepmath-sft \
  trainer.experiment_name=qwen2.5-7b-deepmath-sft-R1-lora \
  trainer.logger=['console'] \
  trainer.total_epochs=1 \
  data.truncation='left' \
  trainer.default_hdfs_dir=null

echo "Job ended at $(date)"