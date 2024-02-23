import cv2
import torch
import numpy as np
from PIL import Image


class GreyscaleContrast(torch.nn.Module):
    """
    Convert the image to greyscale
    Apply histogram equalisation to increase contrast
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        # Convert PIL imaged to numpy array
        img = np.array(img)
        # Convert image to greyscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization to increase contrast
        img = cv2.equalizeHist(img)
        # Convert numpy array back to PIL image
        img = Image.fromarray(img)
        return img
