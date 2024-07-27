import os
import torch
import argparse
from datetime import datetime
from diffusers import StableDiffusionXLPipeline
from ziplora_utils import insert_ziplora_to_unet



# Define your arguments directly
weights_path = "rank_test_models/ziploras/ziplora_style3_r32_monster_toy_r256_555_sbu_toy_crt_style_2024-07-03_02-38-17"
output_dir = "rank_test_models/generated/style3_monster_toy_r32_r256_5_5_5"
prompt_input = "prompts/sbu_toy_crt_prompts.txt"
num_generations = 1
num_inference_steps = 30

def load_prompts(prompt_input):
    if os.path.isfile(prompt_input):
        with open(prompt_input, 'r') as file:
            prompts = [line.strip() for line in file.readlines()]
    else:
        prompts = [prompt_input]
    return prompts

# Debugging: Print arguments to verify
print(f"Weights Path: {weights_path}")
print(f"Output Directory: {output_dir}")
print(f"Prompt Input: {prompt_input}")
print(f"Number of Generations: {num_generations}")
print(f"Number of Inference Steps: {num_inference_steps}")

pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"

prompt_list = load_prompts(prompt_input)

pipe = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path)
pipe.unet = insert_ziplora_to_unet(pipe.unet, weights_path)
pipe.to(device="cuda", dtype=torch.float16)

os.makedirs(output_dir, exist_ok=True)

time = datetime.now().strftime("%Y%m%d_%H%M%S")
for prompt in prompt_list:
    for i in range(num_generations):
        image = pipe(prompt=prompt, num_inference_steps=num_inference_steps).images[0]
        image_filename = os.path.join(output_dir, f"{prompt.replace(' ', '_')}_{i}_{time}.png")
        image.save(image_filename)
        print(f"Generated image saved to {image_filename}")
