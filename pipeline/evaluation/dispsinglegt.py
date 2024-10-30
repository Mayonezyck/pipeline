import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
# Load the depth map
depth_map = cv2.imread('/home/yicheng/Github/pipeline/output/depth-anything:v2_20241025140032_output/0000000000.png', cv2.IMREAD_UNCHANGED)
# Load the image
# image = cv2.imread('0000000000.png')

# # Convert the image from BGR to RGB
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Display the image
# plt.figure()
# plt.imshow(image_rgb)
# plt.title('Image')
# plt.axis('off')
# plt.show()
# plt.imshow(depth_map, cmap='gray')
# plt.colorbar()
# plt.show()

# Create a figure to display the images side by side
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# # Display the RGB image
# axes[0].imshow(image_rgb)
# axes[0].set_title('Image')
# axes[0].axis('off')

# Display the depth map
axes[1].imshow(255 - depth_map, cmap='gray')
axes[1].set_title('Depth Map')
axes[1].axis('off')

# Show the figure
plt.show()