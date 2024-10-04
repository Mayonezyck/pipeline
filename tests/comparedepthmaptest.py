import cv2
import torch

import numpy as np
import matplotlib as plt

imname = '0000000000'
l = cv2.imread(f'/home/yicheng/Github/pipeline/data/Hamlyn/rectified01/image01/{imname}.jpg',cv2.IMREAD_UNCHANGED)
r = cv2.imread(f'/home/yicheng/Github/pipeline/data/Hamlyn/rectified01/image02/{imname}.jpg',cv2.IMREAD_UNCHANGED)

dl = cv2.imread(f'/home/yicheng/Github/pipeline/data/Hamlyn/rectified01/depth01/{imname}.png',cv2.IMREAD_UNCHANGED)
dr = cv2.imread(f'/home/yicheng/Github/pipeline/data/Hamlyn/rectified01/depth02/{imname}.png',cv2.IMREAD_UNCHANGED)
print(f'{np.min(dl)}')
print(f'{np.min(dr)}')
diff = np.abs(dl - dr)

print(np.mean(diff))

dl_f = dl.astype(np.float32)
dr_f = dr.astype(np.float32)

min_disparity = 0.1

f = 383.1901395
B = 5.382236

absolute_depth_left = np.where(dl_f > min_disparity, (f * B) / dl_f, 0)
absolute_depth_right = np.where(dr_f > min_disparity, (f * B) / dr_f, 0)



# dl_f = dl.astype(np.float32)
# dr_f = dr.astype(np.float32)
# diff_f = np.abs(dl_f - dr_f)
# print(f"Left Depth Map - Min: {np.min(dl_f)}, Max: {np.max(dl_f)}")
# print(f"Right Depth Map - Min: {np.min(dr_f)}, Max: {np.max(dr_f)}")

import matplotlib.pyplot as plt

# print(np.mean(diff_f))
min_disparity = 0.1

# dl_f[dl_f <= min_disparity] = np.nan  # Set to NaN to exclude in the final result
# dr_f[dr_f <= min_disparity] = np.nan  # Set to NaN

# # Convert disparity map from left camera to absolute depth
# absolute_depth_left = (f * B) / dl_f

# # Convert disparity map from right camera to absolute depth
# absolute_depth_right = (f * B) / dr_f
absolute_depth_left = np.where(dl_f > min_disparity, (f * B) / dl_f, 0)
absolute_depth_right = np.where(dr_f > min_disparity, (f * B) / dr_f, 0)

print(f"Left Absolute Depth Map - Min: {np.min(absolute_depth_left)}, Max: {np.max(absolute_depth_left)}")
print(f"Right Absolute Depth Map - Min: {np.min(absolute_depth_right)}, Max: {np.max(absolute_depth_right)}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

height,width = 480, 640
X, Y = np.meshgrid(np.arange(width), np.arange(height))
Z = absolute_depth_left
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
scatter = ax.scatter(X_flat, Y_flat, Z_flat, c=Z_flat, cmap='viridis')

# Add color bar for reference
fig.colorbar(scatter)

# Add labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Depth (Z)')

# Show the plot
plt.show()

plt.subplot(2, 2, 1)
plt.imshow(dl_f, cmap='gray')
plt.title('Original Depth Left Camera')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(dr_f, cmap='gray')
plt.title('Original Depth Right Camera')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(absolute_depth_left, cmap='gray')
plt.title('Absolute Depth Left Camera')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(absolute_depth_right, cmap='gray')
plt.title('Absolute Depth Right Camera')
plt.colorbar()

plt.show()


f = 383.1901395
B = 5.382236
dif_max = []
for i in range(10,99):

    dl = cv2.imread(f'/home/yicheng/Github/pipeline/data/Hamlyn/rectified01/depth01/00000000{str(i)}.png',cv2.IMREAD_UNCHANGED)
    dr = cv2.imread(f'/home/yicheng/Github/pipeline/data/Hamlyn/rectified01/depth02/00000000{str(i)}.png',cv2.IMREAD_UNCHANGED)
    dl_f = dl.astype(np.float32)
    dr_f = dr.astype(np.float32)
    diff_f = np.abs(dl_f - dr_f)
    print(f"Left Depth Map - Min: {np.min(dl_f)}, Max: {np.max(dl_f)}")
    print(f"Right Depth Map - Min: {np.min(dr_f)}, Max: {np.max(dr_f)}")
    min_disparity = 0.1
    dl_f[dl_f <= min_disparity] = np.nan  # Set to NaN to exclude in the final result
    dr_f[dr_f <= min_disparity] = np.nan  # Set to NaN

    # Convert disparity map from left camera to absolute depth
    absolute_depth_left = (f * B) / dl_f

    # Convert disparity map from right camera to absolute depth
    absolute_depth_right = (f * B) / dr_f
    print(f"Left Absolute Depth Map - Min: {np.nanmin(absolute_depth_left)}, Max: {np.nanmax(absolute_depth_left)}")
    print(f"Right Absolute Depth Map - Min: {np.nanmin(absolute_depth_right)}, Max: {np.nanmax(absolute_depth_right)}")

    dif_max.append(np.nanmax(absolute_depth_left) - np.nanmax(absolute_depth_right))

import matplotlib.pyplot as plt
plt.plot(dif_max)
plt.show()
