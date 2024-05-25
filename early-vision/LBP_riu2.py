import numpy as np
from skimage import feature

class LBP_riu2:
    def __init__(self, P=8, R=1):
        self.P = P
        self.R = R

    def _thresholded(self, center, pixels):
        return (pixels >= center).astype(int)

    def s(self, x):
        return 1 if x >= 0 else 0

    def _uniformity(self, pattern, center_value):
        # Apply the s(x) function to the differences to get the binary pattern
        binary_pattern = np.array([self.s(p - center_value) for p in pattern])
        # Compute the uniformity of the pattern, including the circular edge case
        transitions = np.abs(binary_pattern[-1] - binary_pattern[0])
        transitions += np.sum(np.abs(np.diff(binary_pattern)))
        return transitions


    def _get_pixel_neighbors(self, image, center_x, center_y):
        indices = np.arange(self.P)
        radius = np.arange(self.R, self.R+1)
        angles = indices * (2 * np.pi / self.P)
        x = radius * np.cos(angles) + center_x
        y = radius * np.sin(angles) + center_y
        x = np.clip(x, 0, image.shape[1] - 1).astype(int)
        y = np.clip(y, 0, image.shape[0] - 1).astype(int)
        return image[y, x]

    def compute_lbp(self, image):
        rows, cols = image.shape
        lbp_image = np.zeros((rows - 2 * self.R, cols - 2 * self.R), dtype=np.uint8)

        for i in range(self.R, rows - self.R):
            for j in range(self.R, cols - self.R):
                center = image[i, j]
                pixels = self._get_pixel_neighbors(image, j, i)
                binary_pattern = self._thresholded(center, pixels)
                uniformity = self._uniformity(binary_pattern)
                if uniformity <= 2:
                    rotated_pattern = self._rotate_to_min(binary_pattern)
                    lbp_value = np.sum(rotated_pattern * (1 << np.arange(self.P)))
                else:
                    lbp_value = self.P + 1
                lbp_image[i - self.R, j - self.R] = lbp_value
        return lbp_image

    def compute_histogram(self, lbp_image):
        n_bins = self.P + 2
        hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float")
        hist /= hist.sum()
        return hist

# Example usage
"""
lbp_rui2 = LBP_riu2(P=8, R=1)
image = np.random.rand(256, 256) * 255  # Example grayscale image
image = image.astype('uint8')

lbp_image = lbp_rui2.compute_lbp(image)
histogram = lbp_rui2.compute_histogram(lbp_image)
"""
