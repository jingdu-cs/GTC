#!/bin/bash

#PBS -l wd
#PBS -P hn98

#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=60GB

#PBS -l jobfs=100GB
#PBS -l walltime=48:00:00
#PBS -l storage=scratch/hn98+gdata/hn98

module load cuda/11.7.0
module load python3/3.8.5
source ~/.bashrc

# 激活 Conda 环境 - 确保这个命令在您的环境中有效
# 如果 condapbs_ex 是自定义命令，确保它已在 .bashrc 中定义
condapbs_ex timeseries || { echo "Failed to activate conda environment"; exit 1; }

# 验证 conda 环境
echo "Current conda environment:"
conda info --envs
which python
python --version

# 显示 GPU 信息
echo "GPU information:"
nvidia-smi 

# 运行 Python 脚本
echo "Starting Python script at $(date)"
python main.py --model "GTC" --dataset "sports"

# 输出结束时间
echo "Job finished at $(date)"
