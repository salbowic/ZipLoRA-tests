#! /bin/bash
sbatch scripts/generate/infere.slurm scripts/generate/generate_sdxl.py \
    --weights_path="models/dreambooth_lora_sdxl_dog6_2024-06-11_22-43-49/checkpoint-500/pytorch_lora_weights.safetensors" \
    --output_dir="generated/dog6" \
    --prompt_input="olis in a bucket" \
    --num_generations=5 \
    --num_inference_steps=50