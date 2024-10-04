import cv2
import torch
import open3d as o3d
import numpy as np
from pytorch3d.ops import iterative_closest_point
from pytorch3d.structures import Pointclouds
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()


def depth_map_to_point_cloud(depth_map, focal_length):
    """
    Convert a depth map into a point cloud in 3D space.
    Assumes the depth map is aligned with a camera coordinate system.
    
    depth_map: 2D numpy array (H, W) of absolute depth values
    focal_length: The focal length of the camera used to capture the depth map

    Returns:
        point_cloud: A (N, 3) numpy array of 3D points
    """
    h, w = depth_map.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    
    # Calculate the corresponding 3D coordinates
    X = (i - w / 2) * depth_map / focal_length
    Y = (j - h / 2) * depth_map / focal_length
    Z = depth_map
    
    # Flatten and stack into a point cloud
    point_cloud = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    return point_cloud

# Function to convert depth map to point cloud
def depth_to_pointcloud_torch(depth, intrinsics):
    # Assuming intrinsics is a 3x3 camera matrix
    h, w = depth.shape
    i, j = torch.meshgrid(torch.arange(h), torch.arange(w))
    z = depth.flatten()
    # Reproject to 3D space using camera intrinsics
    x = (j.flatten() - intrinsics[0][2]) * z / intrinsics[0][0]
    y = (i.flatten() - intrinsics[1][2]) * z / intrinsics[1][1]
    return torch.stack([x, y, z], dim=-1)

def remove_small_points(point_cloud, min_threshold=1e-3):
    """
    Removes points from the point cloud that have very small (close to zero) values in the z-axis (depth).
    
    Args:
    - point_cloud (numpy.ndarray): Input point cloud of shape (N, 3), where each row is [x, y, z].
    - min_threshold (float): The minimum allowed value for z (depth). Points with z <= min_threshold will be removed.

    Returns:
    - filtered_point_cloud (numpy.ndarray): The filtered point cloud with no small z-values.
    """
    # Apply the condition to filter points with small z values
    mask = point_cloud[:, 2] > min_threshold  # Keep points where z > min_threshold
    filtered_point_cloud = point_cloud[mask]
    
    return filtered_point_cloud
def visualize_pointcloud(points):
    """
    Visualize a point cloud using Open3D.
    
    Args:
        points: A torch tensor of shape (N, 3) representing the point cloud.
    """
    # Ensure the tensor is on CPU and convert to NumPy
    points_np = points.cpu().numpy()
    print("NumPy array shape:", points_np.shape)  # Debug print
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Assign points to the Open3D point cloud
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

worldmap = Pointclouds(points=[])
worldmap = worldmap.to(device)
intrinsics = [[383.1901395, 0, 155.9659519195557, 0], [0, 383.1901395, 124.3335933685303, 0],[0, 0, 1, 0]]

f = 383.1901395
B = 5.382236

for i in range(0,22):
    imname = str(i).zfill(10)
    # Load depth map
    depth_map = cv2.imread(f"/home/yicheng/Github/pipeline/data/Hamlyn/rectified01/depth01/{imname}.png", cv2.IMREAD_UNCHANGED)
    depth_map = depth_map.astype(np.float32)
    current_pc = depth_map_to_point_cloud(depth_map, f)
    min_threshold = 1e-2  # Set a threshold for the minimum z-value (depth)

    current_pc_filtered = remove_small_points(current_pc, min_threshold)
    current_pc_filtered_tensor = torch.tensor(current_pc_filtered, dtype=torch.float32)

    current_pc_filtered_tensor_unsqueezed = current_pc_filtered_tensor.unsqueeze(0)

    current_pc_filtered_tensor_unsqueezed = current_pc_filtered_tensor_unsqueezed.to(device)
    if i == 0:
        worldmap = current_pc_filtered_tensor_unsqueezed
    else:
        
        print(worldmap.shape)  # Should be (N, 3)
        #print(transformed_pointcloud.shape)  # Should also be (M, 3)

        icp_result = iterative_closest_point(current_pc_filtered_tensor_unsqueezed, worldmap, max_iterations=200)
        
        print(f'Converged or not: {icp_result.converged}')
        print(f'RMSE for {i} = {icp_result.rmse}')
        #transform = Transform3d().rotate(icp_result.RTs.R).translate(icp_result.RTs.T)
        #transform = transform.to(device)
        # Apply the transformation to the point cloud
        #current_pc_filtered_tensor_unsqueezed = current_pc_filtered_tensor_unsqueezed.to(device)
        #transformed_pointcloud = transform.transform_points(current_pc_filtered_tensor_unsqueezed)
        #print(worldmap.shape)  # Should be (N, 3)
        #print(transformed_pointcloud.shape)  # Should also be (M, 3)
        if icp_result.converged:
            worldmap = torch.cat([worldmap.squeeze(0), icp_result.Xt.squeeze(0)], dim=0)
            worldmap = worldmap.unsqueeze(0)
        # Convert to a full 4x4 transformation matrix
        R = icp_result.RTs.R.squeeze()  # Rotation matrix
        T = icp_result.RTs.T.squeeze()  # Translation vector
        transform_matrix = torch.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = T


        # Print the transformation matrix
        print("Transformation Matrix (ICP Result):")
        print(transform_matrix)
        if i%10 == 0:
            #visualize_pointcloud(worldmap.squeeze(0))
            # Assuming worldmap is the final point cloud tensor with shape [N, 3]
            # Convert PyTorch tensor to a NumPy array
            worldmap_np = worldmap.squeeze(0).cpu().numpy()

            # Create an Open3D point cloud object
            pcd = o3d.geometry.PointCloud()

            # Assign points to the Open3D point cloud
            pcd.points = o3d.utility.Vector3dVector(worldmap_np)

            # Save the point cloud to a PLY file
            o3d.io.write_point_cloud(f"output_pointcloud000{str(i)}.ply", pcd)

            print("Point cloud saved to output_pointcloud.ply")
# This following chunk is for verifing if the ICP will return a consistnet transformation for stereo pairs.
'''
imname = '0000000000'
for i in range(101):
    imname = str(i).zfill(10)
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

    # Mask to ensure no division by zero or small disparities
    mask_left = dl_f > min_disparity
    mask_right = dr_f > min_disparity

    # Initialize absolute depth with zeros
    absolute_depth_left = np.zeros_like(dl_f)
    absolute_depth_right = np.zeros_like(dr_f)

    # Apply the depth formula only where the disparity is valid
    absolute_depth_left[mask_left] = (f * B) / dl_f[mask_left]
    absolute_depth_right[mask_right] = (f * B) / dr_f[mask_right]

    point_cloud_left = depth_map_to_point_cloud(dl_f, f)
    point_cloud_right = depth_map_to_point_cloud(dr_f, f)
    
    min_threshold = 1  # Set a threshold for the minimum z-value (depth)

    point_cloud_left_filtered = remove_small_points(point_cloud_left, min_threshold)
    point_cloud_right_filtered = remove_small_points(point_cloud_right, min_threshold)
    # Convert to torch tensors
    points_left = torch.tensor(point_cloud_left_filtered, dtype=torch.float32)
    points_right = torch.tensor(point_cloud_right_filtered, dtype=torch.float32)

    # Reshape for PyTorch3D (batch size of 1, N points, 3 coordinates)
    points_left = points_left.unsqueeze(0)  # Shape: (1, N, 3)
    points_right = points_right.unsqueeze(0)  # Shape: (1, N, 3)

    points_left = points_left.to(device)
    points_right = points_right.to(device)

    # Perform ICP alignment using PyTorch3D
    icp_result = iterative_closest_point(points_left, points_right, max_iterations=100)

    # Extract the transformation matrix
    R = icp_result.RTs.R.squeeze()  # Rotation matrix
    T = icp_result.RTs.T.squeeze()  # Translation vector

    # Convert to a full 4x4 transformation matrix
    transform_matrix = torch.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = T

    # Print the transformation matrix
    print("Transformation Matrix (ICP Result):")
    print(transform_matrix)
'''


'''
dl_f = dl.astype(np.float32)
dr_f = dr.astype(np.float32)
diff_f = np.abs(dl_f - dr_f)
print(f"Left Depth Map - Min: {np.min(dl_f)}, Max: {np.max(dl_f)}")
print(f"Right Depth Map - Min: {np.min(dr_f)}, Max: {np.max(dr_f)}")

import matplotlib.pyplot as plt

print(np.mean(diff_f))
min_disparity = 0.1

dl_f[dl_f <= min_disparity] = np.nan  # Set to NaN to exclude in the final result
dr_f[dr_f <= min_disparity] = np.nan  # Set to NaN

# Convert disparity map from left camera to absolute depth
absolute_depth_left = (f * B) / dl_f

# Convert disparity map from right camera to absolute depth
absolute_depth_right = (f * B) / dr_f
# absolute_depth_left = np.where(dl_f > min_disparity, (f * B) / dl_f, 0)
# absolute_depth_right = np.where(dr_f > min_disparity, (f * B) / dr_f, 0)

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

# plt.subplot(2, 2, 1)
# plt.imshow(dl_f, cmap='gray')
# plt.title('Original Depth Left Camera')
# plt.colorbar()

# plt.subplot(2, 2, 2)
# plt.imshow(dr_f, cmap='gray')
# plt.title('Original Depth Right Camera')
# plt.colorbar()

# plt.subplot(2, 2, 3)
# plt.imshow(absolute_depth_left, cmap='gray')
# plt.title('Absolute Depth Left Camera')
# plt.colorbar()

# plt.subplot(2, 2, 4)
# plt.imshow(absolute_depth_right, cmap='gray')
# plt.title('Absolute Depth Right Camera')
# plt.colorbar()

# plt.show()


# f = 383.1901395
# B = 5.382236
# dif_max = []
# for i in range(26,60):

#     dl = cv2.imread(f'/home/yicheng/Github/pipeline/data/Hamlyn/rectified01/depth01/00000000{str(i)}.png',cv2.IMREAD_UNCHANGED)
#     dr = cv2.imread(f'/home/yicheng/Github/pipeline/data/Hamlyn/rectified01/depth02/00000000{str(i)}.png',cv2.IMREAD_UNCHANGED)
#     dl_f = dl.astype(np.float32)
#     dr_f = dr.astype(np.float32)
#     diff_f = np.abs(dl_f - dr_f)
#     print(f"Left Depth Map - Min: {np.min(dl_f)}, Max: {np.max(dl_f)}")
#     print(f"Right Depth Map - Min: {np.min(dr_f)}, Max: {np.max(dr_f)}")
#     min_disparity = 0.1
#     # dl_f[dl_f <= min_disparity] = np.nan  # Set to NaN to exclude in the final result
#     # dr_f[dr_f <= min_disparity] = np.nan  # Set to NaN

#     # # Convert disparity map from left camera to absolute depth
#     # absolute_depth_left = (f * B) / dl_f

#     # # Convert disparity map from right camera to absolute depth
#     # absolute_depth_right = (f * B) / dr_f
#     # print(f"Left Absolute Depth Map - Min: {np.nanmin(absolute_depth_left)}, Max: {np.nanmax(absolute_depth_left)}")
#     # print(f"Right Absolute Depth Map - Min: {np.nanmin(absolute_depth_right)}, Max: {np.nanmax(absolute_depth_right)}")

#     dif_max.append(np.max(dl_f) - np.max(dr_f))

# import matplotlib.pyplot as plt
# plt.plot(dif_max)
# plt.show()
'''