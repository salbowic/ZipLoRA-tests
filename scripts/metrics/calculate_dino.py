import argparse
import os
import logging
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

# Set up logging
def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'dino_metric.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Function to extract features
def extract_features(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.squeeze()

# Main function to calculate DINO metric
def calculate_dino_metric(original_dir, generated_dir, model_name, log_dir):
    setup_logging(log_dir)

    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)

    # List of image paths
    real_image_paths = [os.path.join(original_dir, img) for img in os.listdir(original_dir) if img.endswith(('jpg', 'png'))]
    generated_image_paths = [os.path.join(generated_dir, img) for img in os.listdir(generated_dir) if img.endswith(('jpg', 'png'))]

    # Extract features for real and generated images
    real_features = [extract_features(img_path, model, processor) for img_path in real_image_paths]
    generated_features = [extract_features(img_path, model, processor) for img_path in generated_image_paths]

    # Compute pairwise cosine similarities
    similarities = []
    for gen_feat in generated_features:
        for real_feat in real_features:
            similarity = F.cosine_similarity(gen_feat.unsqueeze(0), real_feat.unsqueeze(0))
            similarities.append(similarity.item())

    # Calculate the average similarity (DINO metric)
    dino_metric = sum(similarities) / len(similarities)
    logging.info(f'DINO Metric: {dino_metric}')
    print(f'DINO Metric: {dino_metric}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the DINO metric for generated images.")
    parser.add_argument('--original_dir', type=str, required=True, help='Path to the directory containing original images.')
    parser.add_argument('--generated_dir', type=str, required=True, help='Path to the directory containing generated images.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the pre-trained model from HuggingFace.')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save the log file.')

    args = parser.parse_args()

    calculate_dino_metric(
        original_dir=args.original_dir,
        generated_dir=args.generated_dir,
        model_name=args.model_name,
        log_dir=args.log_dir
    )
