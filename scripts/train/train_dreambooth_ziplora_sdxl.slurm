#!/bin/bash

#SBATCH --job-name=train_ziplora
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=360
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --output=./logs/dreambooth_lora_sdxl/train_ziplora_%j_%a.out

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export HF_HOME="/net/tscratch/people/plgas2000/.cache/huggingface"

# for subject
export LORA_PATH="./models/dreambooth_lora_sdxl_dog3_subject_2024-05-27_02-00-55"
export INSTANCE_DIR="./dataset/dreambooth-main/dataset/dog3"
export PROMPT="a photo of sks dog"

# for style
export LORA_PATH2="./models/dreambooth_lora_sdxl_style1_cartoon_2024-05-27_02-25-27"
export INSTANCE_DIR2="./dataset/styles/style1-cartoon"
export PROMPT2="a cat of in crt style"

# general
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
export OUTPUT_DIR="./models/ziplora_dog3_style1_cartoon_$timestamp"
export VALID_PROMPT="a photo of sks dog in crt style"

export LOG_DIR="./logs/tensorboard_logs"

accelerate launch scripts/train/train_dreambooth_ziplora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --lora_name_or_path=$LORA_PATH \
  --instance_prompt="${PROMPT}" \
  --instance_data_dir=$INSTANCE_DIR \
  --lora_name_or_path_2=$LORA_PATH2 \
  --instance_prompt_2="${PROMPT2}" \
  --instance_data_dir_2=$INSTANCE_DIR2 \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=5e-5 \
  --similarity_lambda=0.01 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=10 \
  --mixed_precision="fp16" \
  --report_to="tensorboard" \
  --enable_xformers_memory_efficient_attention \
  --logging_dir=$LOG_DIR \
  --gradient_checkpointing \
  --use_8bit_adam \
