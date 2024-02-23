"""
Image augmentation to mitigate unbalanced dataset.
"""

import cv2
import numpy as np


def image_augmentation(name: int):
    # Load the image
    image = cv2.imread(f"images/{name}.jpg")

    # Add noise and save
    noise = np.random.normal(0, 1, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    cv2.imwrite(f"images/{name}_noisy.jpg", noisy)

    # Flip the image horizontally and save
    flipped = cv2.flip(image, 2)
    cv2.imwrite(f"images/{name}_flipped.jpg", flipped)

    # Apply more blur to the original image and save
    more_blurred = cv2.GaussianBlur(image, (0, 0), 10)
    cv2.imwrite(f"images/{name}_blurred.jpg", more_blurred)


image_augmentation(1000)
