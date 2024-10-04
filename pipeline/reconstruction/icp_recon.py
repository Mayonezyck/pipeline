import cv2
import torch
import os
import open3d as o3d
import numpy as np
from pytorch3d.ops import iterative_closest_point
from pytorch3d.structures import Pointclouds
from datetime import datetime

def recon(config):
    mute = config['IF_MUTE']
    in_path = config['DEPTH_FOLDER']
    firstTime = True
    device = torch.device("cuda:0" if torch.cuda.is_available() and config['USE_GPU'] else "cpu")
    if not mute:
        print(f'Device chosen: {device}')
    worldmap = Pointclouds(points=[])
    worldmap = worldmap.to(device)
    file_list = sorted([f for f in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, f))])
    for filename in file_list:
        file_path = os.path.join(in_path, filename)
        if not mute:
            print(f"Processing file: {filename}")
        depth_map = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)    
        depth_map = depth_map.astype(np.float32)
        #print(depth_map.shape)
        if len(depth_map.shape) != 2:
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
        #print(depth_map.shape)
        current_pc = depth_map_to_point_cloud(depth_map, config['fx']) #assume same of fx fy
        min_threshold = config['SMALL_POINT_THRESHOLD'] # Set a threshold for the minimum z-value (depth)
        current_pc_filtered = remove_small_points(current_pc, min_threshold)
        current_pc_filtered_tensor = torch.tensor(current_pc_filtered, dtype=torch.float32)

        current_pc_filtered_tensor_unsqueezed = current_pc_filtered_tensor.unsqueeze(0)

        current_pc_filtered_tensor_unsqueezed = current_pc_filtered_tensor_unsqueezed.to(device)
        if firstTime:
            worldmap = current_pc_filtered_tensor_unsqueezed
            if not mute:
                print('World map initialized.')
            firstTime = False
        else:
            if not mute:
                print(f'Current worldmap Shape{worldmap.shape}')
            icp_result = iterative_closest_point(current_pc_filtered_tensor_unsqueezed, worldmap, max_iterations=config['ICP_MAX_ITERATION'])
            if not mute:
                print(f'Converged or not: {icp_result.converged}')
                print(f'RMSE for {filename} = {icp_result.rmse}')
            if icp_result.converged:
                worldmap = torch.cat([worldmap.squeeze(0), icp_result.Xt.squeeze(0)], dim=0)
                worldmap = worldmap.unsqueeze(0)
                output_ply(worldmap)
            R = icp_result.RTs.R.squeeze()  # Rotation matrix
            T = icp_result.RTs.T.squeeze()  # Translation vector
            transform_matrix = torch.eye(4)
            transform_matrix[:3, :3] = R
            transform_matrix[:3, 3] = T
            # Print the transformation matrix
            if not mute:
                print("Transformation Matrix (ICP Result):")
                print(transform_matrix)
            

   
def output_ply(pointcloud):
    pointcloud_np = pointcloud.squeeze(0).cpu().numpy()
     # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Assign points to the Open3D point cloud
    pcd.points = o3d.utility.Vector3dVector(pointcloud_np)
    # Save the point cloud to a PLY file
    
    now = datetime.now()
    current_time = now.strftime("%Y%m%d%H%M%S")
    ply_name = f"output/ICP_{current_time}.ply"
    o3d.io.write_point_cloud(ply_name, pcd)
    print("Point cloud saved to output folder")
    return ply_name


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