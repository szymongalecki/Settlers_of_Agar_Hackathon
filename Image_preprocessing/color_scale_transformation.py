"""
Image preprocessing efforts to make bacteria colonies more visible.
"""

from PIL import Image
import cv2
import skimage.exposure


def greyscale(name: int):
    # Load the image
    image = Image.open(f"{name}.jpg")
    # Convert the image to grayscale
    grey_image = image.convert("L")
    # Save the grayscale image with a new name
    grey_image.save(f"{name}_grey.jpg")


def greyscale_increase_contrast(name: int):
    # Read the grayscale image
    grey_image = cv2.imread(f"{name}.jpg", cv2.IMREAD_GRAYSCALE)
    # Apply histogram equalization
    grey_equalized_image = cv2.equalizeHist(grey_image)
    # Save the image
    cv2.imwrite(f"{name}_grey_contrast.jpg", grey_equalized_image)


def rgb_to_svm(name: int):
    # Read the image - Notice that OpenCV reads the images as BRG instead of RGB
    img = cv2.imread(f"{name}.jpg")
    # Convert the BRG image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert the RGB image to HSV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Save the image with a new name
    cv2.imwrite(f"{name}_HSV.jpg", img)


def rgb_increase_contrast(name: int):
    # Load image with alpha channel
    img = cv2.imread(f"{name}.jpg")
    # Adjust just the input max value
    light = skimage.exposure.rescale_intensity(
        img, in_range=(0, 128), out_range=(0, 255)
    )
    cv2.imwrite(f"{name}_contrast_light.jpg", light)
    # Adjust both the input min and max values
    dark = skimage.exposure.rescale_intensity(
        img, in_range=(64, 192), out_range=(0, 255)
    )
    cv2.imwrite(f"{name}_contrast_dark.jpg", dark)


greyscale(1000)
greyscale_increase_contrast(1000)
rgb_increase_contrast(1000)
rgb_to_svm(1000)