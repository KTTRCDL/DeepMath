#!/bin/bash
#SBATCH -p i64m1tga800u         # 指定GPU队列
#SBATCH -o log/train/debug/deepmath-qwen-2.5-7b-sft-R1/output_%j.txt  # 指定作业标准输出文件，%j为作业号
#SBATCH -e log/train/debug/deepmath-qwen-2.5-7b-sft-R1/err_%j.txt    # 指定作业标准错误输出文件
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
# bash evaluation_readme.sh
bash scripts/train/deepmath-qwen-2.5-7b-sft-R1.sh
echo "Job ended at $(date)"