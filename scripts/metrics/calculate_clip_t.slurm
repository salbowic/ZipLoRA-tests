#!/bin/bash

#SBATCH --job-name=calculate_clip_t
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=30
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --output=./logs/metrics_logs/clip_t_%j_%a.out

export MODEL_NAME="openai/clip-vit-base-patch32"
export GENERATED_DIR="rank_test_models/generated/style3_monster_toy_r256_r256_5_5_5"
export LOG_DIR="./logs/metrics_logs/style3_monster_toy_r256_r256_5_5_5"
export PROMPT_FILE="./prompts/sbu_toy_crt_prompts.txt"
export HF_HOME="/net/tscratch/people/plgas2000/.cache/huggingface"

python scripts/metrics/calculate_clip_t.py \
    --generated_dir=$GENERATED_DIR \
    --prompt_file=$PROMPT_FILE \
    --model_name=$MODEL_NAME \
    --log_dir=$LOG_DIR
