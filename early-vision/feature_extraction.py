import numpy as np
from skimage import io
from skimage.color import rgb2gray
from sklearn.metrics.pairwise import cosine_similarity
from LBP_riu2 import LBP_riu2  
from VAR import VAR            
import matplotlib.pyplot as plt

# Define function to extract combined LBP and VAR features from an image
def extract_combined_features(image_path, lbp_extractor, var_extractor, R=1):
    try:
        image = io.imread(image_path)
        if image.ndim == 3:
            image = rgb2gray(image)  # Convert to grayscale
        image = np.uint8(image * 255)  # Normalize

        # Compute LBP and VAR features
        lbp_features = lbp_extractor.compute_lbp(image).flatten()
        var_features = var_extractor.compute_var(image, R=R).flatten()

        # Combine features
        combined_features = np.concatenate((lbp_features, var_features))
        return combined_features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Define function to find the most similar images based on cosine similarity
def find_most_similar_cosine(query_features, dataset_features, top_k=5):
    if query_features is not None:
        # Compute cosine similarity
        similarities = cosine_similarity([query_features], dataset_features).flatten()
        # Get indices of top K similarities
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        return top_k_indices
"""
# Example workflow
if __name__ == "__main__":
    # Instantiate feature extractors
    lbp_extractor = LBP_riu2(P=8, R=1)
    var_extractor = VAR(P=8)

    # List of image paths
    image_paths = ['path_to_image1.png', 'path_to_image2.png']  # Update with actual paths

    # Extract features for the dataset
    dataset_features = np.array([extract_combined_features(path, lbp_extractor, var_extractor) for path in image_paths if extract_combined_features(path, lbp_extractor, var_extractor) is not None])

    # Path to your query image
    query_image_path = 'path_to_query_image.png'  # Update with the actual path

    # Find and print the most similar images
    similar_image_paths = find_most_similar_cosine(query_image_path, dataset_features, image_paths, top_k=5)
    print("Most similar images:")
    for path in similar_image_paths:
        print(path)
"""