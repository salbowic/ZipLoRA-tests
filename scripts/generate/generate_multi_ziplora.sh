#!/bin/bash

WEIGHTS_PATH="rank_test_models/ziploras/ziplora_style2_r32_dog6_r32_555_sbu_dog_crt_style_2024-07-02_08-49-00"
OUTPUT_DIR="rank_test_models/generated/style2_dog6_r32_r32_5_5_5"
PROMPT_INPUT="prompts/sbu_dog_crt_prompts.txt"
NUM_GENERATIONS=1
NUM_INFERENCE_STEPS=30

sbatch scripts/generate/infere.slurm scripts/generate/generate_ziplora_multi.py \
    --weights_path="$WEIGHTS_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --prompt_input="$PROMPT_INPUT" \
    --num_generations=$NUM_GENERATIONS \
    --num_inference_steps=$NUM_INFERENCE_STEPS