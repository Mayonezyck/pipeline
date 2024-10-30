import numpy as np
import cv2

import matplotlib.pyplot as plt

# Load the depth image (first image)
depth_image_path = '0000000000.png'
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

# Load the prediction image (second image)
prediction_image_path = '0000000000_disp.npy'
prediction_image = np.load(prediction_image_path)

print(f"Depth Image Shape: {depth_image.shape}")
print(f"Prediction Image Shape: {prediction_image.shape}")
depth_image = np.squeeze(depth_image)

#prediction_image = 1/prediction_image #if the result is actually disparity
#Apply mask to the prediction image
# Convert disparity to depth using the baseline and focal length
baseline = 0.5196155
focal_length = 417.9036255  # pixels 

prediction_image = baseline * focal_length / (prediction_image + 1e-6)  # Adding a small value to avoid division by zero

prediction_image = np.squeeze(prediction_image)

# Ensure the prediction image has the same shape as the depth image
prediction_image = cv2.resize(prediction_image, (depth_image.shape[1], depth_image.shape[0]))

mask = depth_image > 0
# Normalize the prediction image
# prediction_image = (prediction_image - np.min(prediction_image)) / (np.max(prediction_image) - np.min(prediction_image))
# Normalize the masked prediction image
#prediction_image = (prediction_image - np.min(prediction_image[mask])) / (np.max(prediction_image[mask]) - np.min(prediction_image[mask]))
print(f"Prediction Image Range: min={prediction_image.min()}, max={prediction_image.max()}")
scale = np.median(depth_image[mask]) / np.median(prediction_image[mask])
print(f"Median of Depth Image: {np.median(depth_image[mask])}")
print(f"Median of Prediction Image: {np.median(prediction_image[mask])}")
print(f"Scale: {scale}")
pred_image_scaled = prediction_image * scale


masked_prediction_image = np.zeros_like(pred_image_scaled)
masked_prediction_image[mask] = pred_image_scaled[mask]
print(f"Depth Image Range: min={depth_image.min()}, max={depth_image.max()}")
print(f"Masked Prediction Image Range: min={masked_prediction_image.min()}, max={masked_prediction_image.max()}")
# Plot the images side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(depth_image, cmap='gray')
axes[0].set_title('Depth Image')
axes[0].axis('off')

axes[1].imshow(masked_prediction_image, cmap='gray')
axes[1].set_title('Prediction Image')
axes[1].axis('off')

plt.show()