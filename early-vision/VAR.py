import numpy as np

class Var:
    def __init__(self, P=8):
        self.P = P

    def _get_pixel_neighbors(self, image, center_x, center_y, R):
        indices = np.arange(self.P)
        angles = indices * (2 * np.pi / self.P)
        x = R * np.cos(angles) + center_x
        y = R * np.sin(angles) + center_y
        x = np.clip(x, 0, image.shape[1] - 1).astype(int)
        y = np.clip(y, 0, image.shape[0] - 1).astype(int)
        return image[y, x]

    def compute_var(self, image, R):
        var_image = np.zeros_like(image, dtype=np.float32)
        for i in range(R, image.shape[0] - R):
            for j in range(R, image.shape[1] - R):
                neighbors = self._get_pixel_neighbors(image, j, i, R)
                mu = np.mean(neighbors)
                var_image[i, j] = np.sum((neighbors - mu) ** 2) / self.P
        return var_image

# Example usage
"""
var_obj = VAR(P=8)
image = np.random.rand(256, 256) * 255  # Example grayscale image
image = image.astype('uint8')
R = 1  # Example radius
var_image = var_obj.compute_var(image, R)
"""