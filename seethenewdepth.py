import cv2
import matplotlib.pyplot as plt

# Load the image in grayscale mode
image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Plot the histogram
    plt.hist(image.ravel(), bins=256, range=[0, 256])
    plt.title('Histogram of the Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()