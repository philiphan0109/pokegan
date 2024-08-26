import os
import requests
from tqdm import tqdm
from datasets import load_dataset

# Load the dataset
ds = load_dataset("TheFusion21/PokemonCards")

# Directory to save the images
save_dir = "data/retry/"
os.makedirs(save_dir, exist_ok=True)

# Function to download an image
def download_image(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)

# Loop through the train split and download images
print("Downloading images from the train split...")
for item in tqdm(ds['train']):
    img_url = item['image_url']
    img_id = item['id']
    save_path = os.path.join(save_dir, f"{img_id}.png")
    download_image(img_url, save_path)

print("All images have been downloaded.")
