import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compare_images(color_image_path, grayscale_image_path):
    """
    Compare a colored image and a grayscale image by resizing if necessary and computing various metrics.
    
    Args:
        color_image_path (str): Path to the colored image.
        grayscale_image_path (str): Path to the grayscale image.
    
    Returns:
        dict: A dictionary containing comparison metrics.
    """
    # Load the colored image
    color_image = cv2.imread(color_image_path)
    if color_image is None:
        raise FileNotFoundError(f"Could not load image at {color_image_path}")

    # Load the grayscale image
    grayscale_image = cv2.imread(grayscale_image_path, cv2.IMREAD_GRAYSCALE)
    if grayscale_image is None:
        raise FileNotFoundError(f"Could not load image at {grayscale_image_path}")

    # Resize images to the same size if necessary
    color_height, color_width = color_image.shape[:2]
    gray_height, gray_width = grayscale_image.shape[:2]

    if (color_height != gray_height) or (color_width != gray_width):
        print("Resizing grayscale image to match the colored image dimensions.")
        grayscale_image = cv2.resize(grayscale_image, (color_width, color_height), interpolation=cv2.INTER_AREA)

    grayscale_image = 255 - grayscale_image

    
    # Convert the colored image to grayscale
    color_image_gray = cv2.cvtColor(color_image, cv2.COLOR_RGBA2GRAY)

    # Compute Mean Squared Error (MSE)
    mse_value = np.mean((color_image_gray - grayscale_image) ** 2)

    # Compute Peak Signal-to-Noise Ratio (PSNR)
    psnr_value = cv2.PSNR(color_image_gray, grayscale_image)

    # Compute Structural Similarity Index (SSIM)
    ssim_value, _ = ssim(color_image_gray, grayscale_image, full=True)

    # Compute Mean Absolute Error (MAE)
    mae_value = np.mean(np.abs(color_image_gray - grayscale_image))

    # Compute Histogram Comparisons
    hist_color = cv2.calcHist([color_image_gray], [0], None, [256], [0, 256])
    hist_gray = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])

    # Normalize histograms
    cv2.normalize(hist_color, hist_color)
    cv2.normalize(hist_gray, hist_gray)

    # Histogram Correlation
    hist_corr = cv2.compareHist(hist_color, hist_gray, cv2.HISTCMP_CORREL)

    # Histogram Chi-Squared
    hist_chi_sq = cv2.compareHist(hist_color, hist_gray, cv2.HISTCMP_CHISQR)

    # Histogram Intersection
    hist_intersect = cv2.compareHist(hist_color, hist_gray, cv2.HISTCMP_INTERSECT)

    # Histogram Bhattacharyya Distance
    hist_bhatta = cv2.compareHist(hist_color, hist_gray, cv2.HISTCMP_BHATTACHARYYA)

    # Compile all metrics into a dictionary
    metrics = {
        'Mean Squared Error (MSE)': mse_value,
        'Peak Signal-to-Noise Ratio (PSNR)': psnr_value,
        'Structural Similarity Index (SSIM)': ssim_value,
        'Mean Absolute Error (MAE)': mae_value,
        'Histogram Correlation': hist_corr,
        'Histogram Chi-Squared': hist_chi_sq,
        'Histogram Intersection': hist_intersect,
        'Histogram Bhattacharyya Distance': hist_bhatta
    }

    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    return metrics

# Example usage
if __name__ == "__main__":
    # Replace with your image paths
    color_image_path = 'comparepp/0000000000.jpeg'
    grayscale_image_path = 'comparepp/0000000000mypp_depth.png'
    
    compare_images(color_image_path, grayscale_image_path)
