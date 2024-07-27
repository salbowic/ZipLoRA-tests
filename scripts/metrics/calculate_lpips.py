import argparse
import os
import logging
from PIL import Image
import torchvision.transforms as transforms 
import torch
import lpips

# Set up logging
def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'lpips_metric.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Function to calculate LPIPS
def calculate_lpips_metric(original_dir, generated_dir, log_dir):
    setup_logging(log_dir)

    # Initialize LPIPS model
    loss_fn = lpips.LPIPS(net='alex')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to a common size
        transforms.ToTensor()
    ])

    # List of image paths
    real_image_paths = [os.path.join(original_dir, img) for img in os.listdir(original_dir) if img.endswith(('jpg', 'png'))]
    generated_image_paths = [os.path.join(generated_dir, img) for img in os.listdir(generated_dir) if img.endswith(('jpg', 'png'))]

    # Compute LPIPS for each pair
    lpips_scores = []
    for real_path in real_image_paths:
        real_img = Image.open(real_path).convert("RGB")
        for gen_path in generated_image_paths:
            gen_img = Image.open(gen_path).convert("RGB")
            lpips_score = loss_fn(transform(real_img), transform(gen_img))
            lpips_scores.append(lpips_score.item())

    # Calculate the average LPIPS score
    avg_lpips_score = sum(lpips_scores) / len(lpips_scores)
    logging.info(f'Average LPIPS Score: {avg_lpips_score}')
    print(f'Average LPIPS Score: {avg_lpips_score}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the LPIPS metric for generated images.")
    parser.add_argument('--original_dir', type=str, required=True, help='Path to the directory containing original images.')
    parser.add_argument('--generated_dir', type=str, required=True, help='Path to the directory containing generated images.')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save the log file.')

    args = parser.parse_args()

    calculate_lpips_metric(
        original_dir=args.original_dir,
        generated_dir=args.generated_dir,
        log_dir=args.log_dir
    )
