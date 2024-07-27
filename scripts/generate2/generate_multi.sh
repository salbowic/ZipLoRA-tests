#! /bin/bash
sbatch scripts/generate/infere.slurm scripts/generate/generate_sdxl.py \
    --weights_path="models/dreambooth_lora_sdxl_rc_car_2024-05-27_18-41-49/pytorch_lora_weights.safetensors" \
    --output_dir="generated/rc_car" \
    --prompt_input="prompts/rc_car_prompts.txt" \
    --num_generations=1 \
    --num_inference_steps=30