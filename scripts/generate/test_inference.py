import torch
from diffusers import AutoPipelineForText2Image

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "A very stupid looking cat with a broom trying to catch a bird wearing a baseball cap"
image = pipeline_text2image(prompt=prompt).images[0]

image.save("sd_test.jpg")
