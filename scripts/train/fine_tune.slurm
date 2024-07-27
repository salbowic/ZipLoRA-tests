#!/bin/bash

#SBATCH --job-name=test_train_dreambooth
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=360
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --output=./logs/train_dreambooth_%j_%a.out

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./dreambooth_train_images"
export CLASS_DIR="./dreambooth_class_images"
export OUTPUT_DIR="./dreambooth_test_model"
export HF_HOME="/net/tscratch/people/plgfilipeckijan/.cache/huggingface"

accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision='no' \
  --dynamo_backend='no' \
  scripts/train/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="A picture of sks dog" \
  --class_prompt="A picture of a dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
