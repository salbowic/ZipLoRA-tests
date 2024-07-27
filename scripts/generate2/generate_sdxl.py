import os
from diffusers import DiffusionPipeline
import torch
import argparse
from datetime import datetime

def load_prompts(prompt_input):
    # Check if the prompt_input is a path to a text file
    if os.path.isfile(prompt_input):
        with open(prompt_input, 'r') as file:
            prompts = [line.strip() for line in file.readlines()]
    else:
        # Treat prompt_input as a single prompt string
        prompts = [prompt_input]
    return prompts

parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--prompt_input", required=True)
parser.add_argument("--num_generations", type=int, required=True)
parser.add_argument("--num_inference_steps", type=int, required=True)

args = parser.parse_args()

# Set the paths
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
lora_weights_path = args.weights_path
output_dir = args.output_dir
prompt_input = args.prompt_input
num_generations = args.num_generations
num_inference_steps = args.num_inference_steps

# Load the prompts
prompt_list = load_prompts(prompt_input)

# Load the base pipeline and apply the LoRA weights
pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_weights_path)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

time = datetime.now().strftime("%Y%m%d_%H%M%S")
for prompt in prompt_list:
    for i in range(num_generations):
        # Perform inference with the base pipeline
        image = pipe(prompt=prompt, num_inference_steps=num_inference_steps).images[0]
        image.save(os.path.join(output_dir, f"{prompt.replace(' ', '_')}_{i}_{time}.png"))
