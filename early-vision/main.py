import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from LBP_riu2 import LBP_riu2
from VAR import VAR
from feature_extraction import extract_combined_features, find_most_similar_cosine


# Function to open an image and extract features
def open_image():
    global query_features, img_path
    
    img_path = filedialog.askopenfilename()
    if img_path:
        query_features = extract_combined_features(img_path, lbp_extractor, var_extractor)
        load_query_image(img_path)

# Function to load and display the query image
def load_query_image(path):
    img = Image.open(path)
    img.thumbnail((200, 200))  # Resize for thumbnail
    img = ImageTk.PhotoImage(img)
    query_panel.config(image=img)
    query_panel.image = img  # Keep a reference so it's not garbage collected

# Function to find and display similar images
def find_similar():
    if query_features is not None:
        top_k_indices = find_most_similar_cosine(query_features, dataset_features, top_k=5)
        display_similar_images(top_k_indices)

# Function to display similar images
def display_similar_images(indices):
    for index, panel in zip(indices, result_panels):
        img_path = image_paths[index]
        img = Image.open(img_path)
        img.thumbnail((100, 100))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

# Initialize the feature extractors
lbp_extractor = LBP_riu2(P=8, R=1)
var_extractor = VAR(P=8)

# Dummy dataset features and paths (replace with your actual data)
dataset_features = np.load('dataset_features.npy')
image_paths = ['image1.jpg', 'image2.jpg', ...]  # replace with actual paths

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
