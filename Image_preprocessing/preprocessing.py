import os
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np

import numpy as np
import cv2
from PIL import Image
import torch.nn as nn


class GreyscaleContrast(nn.Module):
    """
    Convert the image to greyscale
    Apply histogram equalisation to increase contrast
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        # Convert PIL Image to numpy array
        img = np.array(img)

        # Convert image to greyscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization
        img = cv2.equalizeHist(img)

        # Convert numpy array back to PIL Image
        img = Image.fromarray(img)

        return img


# Path to the folder containing the original images
folder_path = "train_data"
# Path to the folder where resized images will be saved
output_folder_path = "train_data_resized"

# Define the transformation pipeline
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        GreyscaleContrast(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ColorJitter(brightness=0.01),
    ]
)

# Initialize counter
counter = 0

# Ensure the output folder exists
os.makedirs(output_folder_path, exist_ok=True)

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        # Open the image
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)

        # Convert the image to numpy array
        img_np = np.array(img)

        # Apply the transformations
        img_transformed = transform(Image.fromarray(img_np))

        # Save the transformed image with a suffix added to the filename
        new_filename = os.path.splitext(filename)[0] + ".jpg"
        output_img_path = os.path.join(output_folder_path, new_filename)
        img_transformed.save(output_img_path)

        # Increment counter
        counter += 1

        # Print progress
        print(f"Processed {counter} images.")

print("Resizing and transformations completed.")


import os
import shutil

# Path to the source folder containing the original .json files
source_folder_path = "train_data"
# Path to the destination folder where .json files will be moved
destination_folder_path = "train_data_resized"

# Ensure the destination folder exists
os.makedirs(destination_folder_path, exist_ok=True)

# Iterate through all files in the source folder
for filename in os.listdir(source_folder_path):
    if filename.endswith(".json"):
        # Construct paths for source and destination
        source_file_path = os.path.join(source_folder_path, filename)
        destination_file_path = os.path.join(destination_folder_path, filename)

        # Move the .json file to the destination folder
        shutil.move(source_file_path, destination_file_path)
        print(f"Moved {filename} to {destination_folder_path}")

print("All .json files moved.")
