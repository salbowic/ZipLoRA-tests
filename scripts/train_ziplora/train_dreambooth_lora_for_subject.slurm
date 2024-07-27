#!/bin/bash

#SBATCH --job-name=train_loras_for_subject_lower_ppl
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=500
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --output=./logs/dreambooth_lora_sdxl/train_lora_for_subject_lower_ppl_%j_%a.out

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
export OUTPUT_DIR="./rank_test_models/subject_loras/db_lora_sdxl_sbu_toy_monster_toy_subject_r256_5000_$timestamp"
export INSTANCE_DIR="./dataset/dreambooth-main/dataset/monster_toy"
export PROMPT="a sbu toy"
export VALID_PROMPT="a sbu toy in a bucket"
export HF_HOME="/net/tscratch/people/plgas2000/.cache/huggingface"
export LOG_DIR="./logs/tensorboard_logs"

accelerate launch scripts/train_ziplora/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${PROMPT}" \
  --rank=256 \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=5e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=100 \
  --seed="1" \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --use_8bit_adam \