import torch
from diffusers import StableDiffusionXLPipeline
from ziplora_utils import insert_ziplora_to_unet

pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
ziplora_name_or_path = "./models/ziplora_style3_5000_r256_dog6_5000_r32_1000_2024-06-16_13-06-27/pytorch_lora_weights.safetensors"

# Define the prompt and the random generator
prompt = "a sbu dog in the crt style"

pipeline = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path)
pipeline.unet = insert_ziplora_to_unet(pipeline.unet, ziplora_name_or_path)
pipeline.to(device="cuda", dtype=torch.float16)
image = pipeline(prompt=prompt, num_inference_steps=25).images[0]
image.save("generated/test_ziplora_style3_r256_dog6_r32_5_5_1_v3.png")