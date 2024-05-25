import os
import cv2
from PIL import Image
import numpy as np


def load_dataset(root_dir, target_size=(256, 256)):
    images = []
    labels = []
    label_names = []

    print("Root Directory:", root_dir)
    print("Subdirectories and files:")

    for subdir, dirs, files in os.walk(root_dir):
        print("Currently scanning:", subdir)  # Print the current directory being scanned

        for file in files:
            if file.lower().endswith(".png"):  # Ensuring case insensitivity
                file_path = os.path.join(subdir, file)
                try:
                    # Open the image and convert to RGB
                    image = Image.open(file_path).convert('RGB')
                    if target_size:
                        # Resize using PIL's LANCZOS (high-quality downsampling)
                        image = image.resize(target_size, Image.LANCZOS)
                    # Convert image to numpy array
                    image_array = np.array(image)
                    # Apply preprocessing
                    processed_image = preprocess_image(image_array)
                    images.append(processed_image)
                    labels.append(os.path.basename(subdir))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        if subdir == root_dir:
            label_names.extend(dirs)

    images = np.array(images, dtype=np.uint8)  # Ensure images are stored as uint8
    labels = np.array(labels)

    return images, labels, label_names

def preprocess_image(image):
    """Apply grayscale, histogram equalization, and Gaussian blur."""
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)
    # Apply Gaussian filtering
    filtered_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
    return filtered_image



if __name__ == "__main__":
    # Usage
    root_dir = 'C:/Users/gdeok/Computer-Vision/early-vision/Large'  # Make sure this is the correct path
    images, labels, label_names = load_dataset(root_dir, target_size=(256, 256))

    print("Loaded", len(images), "images.")
    print("Label names:", label_names)
    print("First 10 labels:", labels[:10])
