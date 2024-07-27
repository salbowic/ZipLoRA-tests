import torch
from diffusers import StableDiffusionPipeline

model_id = "../../models/dreambooth_test_model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A picture of sks dog with red background"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("example_fine_tuned_image.jpg")
