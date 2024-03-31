import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from LBP_riu2 import LBP_riu2
from VAR import VAR
from feature_extraction import extract_combined_features, find_most_similar_cosine
from sklearn.datasets import fetch_openml  # To fetch the MNIST dataset
from tkinter.simpledialog import askinteger 


# Initialize the feature extractors
lbp_extractor = LBP_riu2(P=8, R=1)
var_extractor = VAR(P=8)

# Modify feature extraction to work with MNIST data format
def extract_combined_features(image, lbp_extractor, var_extractor, R=1):
    image = np.uint8(image * 255)  # Normalize

    # Compute LBP and VAR features
    lbp_features = lbp_extractor.compute_lbp(image).flatten()
    var_features = var_extractor.compute_var(image, R=R).flatten()

    # Combine features
    combined_features = np.concatenate((lbp_features, var_features))
    return combined_features

# Fetch MNIST data
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist_data = mnist.data / 255.0  # Normalize MNIST data

# Use a subset of MNIST as our dataset
dataset_images = mnist.data[:1000].reshape((-1, 28, 28))  # Reshape to 28x28 images
dataset_features = [extract_combined_features(img, lbp_extractor, var_extractor) for img in dataset_images]



# Reshape to 28x28 images
dataset_images = mnist_data[:1000].reshape((-1, 28, 28))

# Modify the display function to handle MNIST images
def display_mnist_image(image, panel):
    img = Image.fromarray(image)
    img = img.resize((200, 200), Image.NEAREST)  # Resize for better viewing
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img  # Keep a reference

# Modify the 'open_image' function to open MNIST images instead
def open_image():
    # Use a simple dialog to choose an index instead of a file path
    index = askinteger("Input", "Enter an index (0-999):")
    if index is not None and 0 <= index < 1000:
        global query_features
        query_image = dataset_images[index]
        query_features = dataset_features[index]
        display_mnist_image(query_image * 255, query_panel) # Multiply by 255 to display correctly

# Modify the 'find_similar' function to display MNIST images
def find_similar():
    if query_features is not None:
        # Pass the precomputed query_features, dataset_features, and image_paths to the function
        top_k_indices = find_most_similar_cosine(query_features, dataset_features, top_k=5)
        for index, panel in zip(top_k_indices, result_panels):
            img = dataset_images[index] * 255  # Multiply by 255 to display correctly
            img = Image.fromarray(img)
            img.thumbnail((100, 100))
            img = ImageTk.PhotoImage(img)
            panel.config(image=img)
            panel.image = img # Multiply by 255 to display correctly



# Create main window
root = tk.Tk()
root.title("Image Similarity GUI")

# Load query image button
load_button = tk.Button(root, text="Load Query Image", command=open_image)
load_button.pack()

# Panel to display the query image
query_panel = tk.Label(root)
query_panel.pack()

# Find similar images button
find_button = tk.Button(root, text="Find Similar Images", command=find_similar)
find_button.pack()

# Panels to display the result images
result_panels = [tk.Label(root) for _ in range(5)]
for panel in result_panels:
    panel.pack(side=tk.LEFT)

root.mainloop()
