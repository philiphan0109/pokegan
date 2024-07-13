import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from model import Generator

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load the generator
def load_generator(generator_path):
    generator = Generator(100, 64).to(device)
    checkpoint = torch.load(generator_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['model_state_dict'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator.eval()
    return generator

def generate_and_save_images(generator, num_images=64, save_path="artifacts/images/"):
    with torch.no_grad():
        z_vector = torch.randn(num_images, 100, 1, 1, device=device)
        imgs_generated = generator(z_vector).detach().cpu()
    
    img_grid = torchvision.utils.make_grid(imgs_generated, nrow=8, normalize=True)
    img_grid_np = np.transpose(img_grid.numpy(), (1, 2, 0))
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(img_grid_np)
    
    # Extract the checkpoint number from the file name
    checkpoint_num = os.path.basename(generator_path).split('_')[1].split('.')[0]
    save_file = os.path.join(save_path, f"generated_images_{checkpoint_num}.png")
    plt.savefig(save_file, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Generated images saved to {save_file}")

if __name__ == "__main__":
    generator_path = "artifacts/checkpoints/generator_6159.pth"
    generator = load_generator(generator_path)
    
    generate_and_save_images(generator)