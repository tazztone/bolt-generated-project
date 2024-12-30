		import numpy as np
    from PIL import Image

    def calculate_laplacian_variance(image: Image.Image) -> float:
        \"\"\"Calculates the Laplacian variance of an image.

        Args:
            image: The input image.

        Returns:
            The Laplacian variance of the image.
        \"\"\"
        # Convert PIL image to NumPy array
        img_array = np.array(image)

        # Ensure the image is in grayscale
        if img_array.ndim == 3:
            img_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])

        # Compute the Laplacian
        laplacian = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])

        # Convolve the image with the Laplacian
        output = np.zeros_like(img_array)
        for i in range(1, img_array.shape[0] - 1):
            for j in range(1, img_array.shape[1] - 1):
                output[i, j] = np.sum(img_array[i - 1:i + 2, j - 1:j + 2] * laplacian)

        # Compute the variance of the Laplacian
        variance = np.var(output)

        return variance

    class LaplacianVariance:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "image": ("IMAGE",),
                }
            }

        RETURN_TYPES = ("FLOAT",)
        FUNCTION = "compute"
        CATEGORY = "marduk191/image"

        def compute(self, image):
            # Convert the tensor to PIL Image
            pil_image = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            
            variance = calculate_laplacian_variance(pil_image)
            return (variance,)

    NODE_CLASS_MAPPINGS = {
        "LaplacianVariance": LaplacianVariance,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "LaplacianVariance": "Laplacian Variance",
    }
