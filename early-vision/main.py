# Normalize the histogram
import numpy as np
import matplotlib.pyplot as plt

# Assume the LBP_riu2 and VAR classes are defined correctly in their respective modules
from LBP_riu2 import LBP_riu2
from VAR import VAR

# Instantiate both classes
lbp_riu2 = LBP_riu2(P=8, R=1)
var = VAR(P=8)

# Load your image or create a synthetic one for testing
# image = skimage.io.imread('path_to_your_image.png') # Uncomment this line if you have an image to load
image = np.random.rand(256, 256) * 255  # Synthetic image for example purposes
image = image.astype('uint8')

# Compute LBP_riu2 and VAR features
lbp_image = lbp_riu2.compute_lbp(image)
var_image = var.compute_var(image, R=1)

# Compute the joint feature vector or joint histogram here as needed for your application
# The exact implementation would depend on how you plan to use these features

# Example of creating a joint histogram (2D histogram)
num_bins = 20  # Number of bins for histogram
joint_histogram, xedges, yedges = np.histogram2d(lbp_image.ravel(), var_image.ravel(), bins=num_bins)

# Normalize the histogram
joint_histogram_normalized = joint_histogram / np.sum(joint_histogram)

# Plot the joint histogram
plt.imshow(joint_histogram_normalized, interpolation='nearest', cmap='hot', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.title('Joint Histogram of LBP_riu2 and VAR')
plt.xlabel('VAR Values')
plt.ylabel('LBP_riu2 Values')
plt.colorbar()
plt.show()
joint_histogram_normalized = joint_histogram / np.sum(joint_histogram)

# Use a logarithmic color scale to enhance visibility of low-frequency bins
plt.imshow(np.log1p(joint_histogram_normalized), interpolation='nearest', cmap='hot')
plt.title('Log-Scaled Joint Histogram of LBP_RIU2 and VAR')
plt.xlabel('VAR Values')
plt.ylabel('LBP_RIU2 Values')
plt.colorbar()
plt.show()
