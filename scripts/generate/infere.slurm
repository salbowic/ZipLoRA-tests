#!/bin/bash

#SBATCH --job-name=sdxl_generate
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=60
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --output=./logs/generation_logs/generate_%j_%a.log

export HF_HOME="/net/tscratch/people/plgas2000/.cache/huggingface"

python $1
