#!/bin/bash
#SBATCH --job-name=spliceai_auto
#SBATCH --partition=jklab-gpu-A6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=workdir/logs/%j.out
#SBATCH --error=workdir/logs/%j.err

source /home/dwkim/anaconda3/etc/profile.d/conda.sh
conda activate portal

python workdir/run_spliceai.py
