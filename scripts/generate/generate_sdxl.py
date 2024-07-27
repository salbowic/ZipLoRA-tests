from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch

# Set the paths
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
lora_weights_path = "./models/db_lora_sdxl_style5_r256_ts2000_2024-06-15_19-32-34/pytorch_lora_weights.safetensors"
output_dir = "./generated"

# Load the base pipeline and apply the LoRA weights
pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_weights_path)

# Define the prompt and the random generator
prompt = "a city in the crt style"

# Perform inference with the base pipeline
image = pipe(prompt=prompt, num_inference_steps=25).images[0]
image.save("generated/test_dreambooth_style5_city_v1.png")
