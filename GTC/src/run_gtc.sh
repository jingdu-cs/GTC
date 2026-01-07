#!/bin/bash

module load cuda/11.7.0
module load python3/3.8.5
source ~/.bashrc

condapbs_ex timeseries || { echo "Failed to activate conda environment"; exit 1; }

echo "Current conda environment:"
conda info --envs
which python
python --version

echo "GPU information:"
nvidia-smi 

echo "Starting Python script at $(date)"
python main.py --model "GTC" --dataset "sports"

# 输出结束时间
echo "Job finished at $(date)"
