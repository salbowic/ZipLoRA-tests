#!/bin/bash

#SBATCH --job-name=test_train_dreambooth
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=360
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --output=./logs/dreambooth_sdxl/train_dreambooth_sdxl_%j_%a.out

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="./dataset/dreambooth-main/dataset/dog5"
export CLASS_DIR="./dataset/dreambooth_generated/dog"
export OUTPUT_DIR="./models/dreambooth_sdxl_dog5"
export HF_HOME="/net/tscratch/people/plgas2000/.cache/huggingface"
export LOG_DIR="./logs/tensorboard_logs"
export VAE_PATH="stabilityai/sdxl-vae"


accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision='no' \
  --dynamo_backend='no' \
  scripts/train/train_dreambooth_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --pretrained_model_name_or_path=$VAE_PATH \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a xyz5 dog" \
  --class_prompt="a dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --logging_dir=$LOG_DIR \
  --report_to="tensorboard"
