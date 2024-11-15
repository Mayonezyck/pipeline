import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
# Load the depth map
depth_map = cv2.imread('/home/yicheng/Github/pipeline/data/Hamlyn/rectified17/depth01_10/0000000000.png', cv2.IMREAD_UNCHANGED)
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
# Convert the depth map from RGBA to grayscale
print(depth_map.shape)
# if depth_map.shape[2] == 4:  # Check if the depth map has an alpha channel
#     depth_map_gray = cv2.cvtColor(depth_map, cv2.COLOR_RGBA2GRAY)
#     print("Depth map has an alpha channel")
# else:
#     depth_map_gray = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)
# depth_map = depth_map_gray
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# # Display the RGB image
# axes[0].imshow(image_rgb)
# axes[0].set_title('Image')
# axes[0].axis('off')

# Display the depth map
axes[1].imshow(depth_map, cmap='rainbow')
axes[1].set_title('Depth Map')
axes[1].axis('off')

# Show the figure
plt.show()