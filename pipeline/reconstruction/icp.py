import torch, os
from pytorch3d.structures import Pointclouds
import matplotlib.pyplot as plt

import numpy as np
import cv2
import open3d as o3d
from scipy.spatial import cKDTree
from datetime import datetime
def register_depth_map_to_pointcloud_new(depth_map_path, device, need_reverse, focal_length, texture_folder, universal_mask=None):
    """
    Registers a depth map to a point cloud using PyTorch3D's Pointclouds structure,
    and applies color from a corresponding texture image.

    Args:
        depth_map_path (str): Path to the depth map image file.
        device (torch.device): The device to use ('cpu' or 'cuda').
        need_reverse (bool): Whether to reverse the depth values.
        focal_length (float): The focal length of the camera.
        texture_folder (str): Path to the folder containing texture images.
        universal_mask (np.ndarray): A binary mask indicating valid pixels.

    Returns:
        Pointclouds: A PyTorch3D Pointclouds object containing the point cloud data with colors.
    """
    # Read the depth map using OpenCV in unchanged mode to preserve depth information
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        raise ValueError(f"Could not read depth map from path: {depth_map_path}")

    if need_reverse:
        depth_map = 255 - depth_map

    # Convert depth map from RGBA or RGB to grayscale if necessary
    if len(depth_map.shape) == 3:
        if depth_map.shape[2] == 4:
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGBA2GRAY)
        elif depth_map.shape[2] == 3:
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)

    # Ensure depth map is single-channel
    if len(depth_map.shape) > 2:
        depth_map = depth_map[:, :, 0]

    # Get the dimensions of the depth map
    height, width = depth_map.shape

    # Construct the corresponding texture image path
    depth_map_filename = os.path.basename(depth_map_path)
    depth_map_name, _ = os.path.splitext(depth_map_filename)
    texture_filename = depth_map_name.split('_')[0] + '.jpg'
    texture_path = os.path.join(texture_folder, texture_filename)
    # Read the texture image
    texture_image = cv2.imread(texture_path)
    if texture_image is None:
        raise ValueError(f"Could not read texture image from path: {texture_path}")

    # Ensure the texture image matches the depth map size
    if texture_image.shape[0] != height or texture_image.shape[1] != width:
        texture_image = cv2.resize(texture_image, (width, height), interpolation=cv2.INTER_LINEAR)

    # Convert texture image from BGR to RGB
    texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

    # Flatten the texture image and depth map
    texture_image = texture_image.reshape(-1, 3)
    depth = depth_map.flatten()

    # Create a grid of (x, y) coordinates corresponding to each pixel in the depth map
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    xx = xx.flatten()
    yy = yy.flatten()

    # Create a valid mask based on depth > 0
    #valid = depth > 0 
    valid = np.logical_and(depth > 0, depth < 255)
    print(valid.shape)
    print(universal_mask.shape)
    # If universal_mask is provided, apply it
    if universal_mask is not None:
        universal_mask_flat = universal_mask.flatten()
        valid = np.logical_and(valid, universal_mask_flat)

    # Filter the data with the valid mask
    xx = xx[valid]
    yy = yy[valid]
    depth = depth[valid]
    colors = texture_image[valid]

    # Define camera intrinsics (assuming a pinhole camera model)
    fx = fy = focal_length  # Focal lengths in pixel units
    cx = width / 2.0  # Principal point x-coordinate
    cy = height / 2.0  # Principal point y-coordinate

    # Convert pixel coordinates and depth to 3D space coordinates
    x = (xx - cx) * depth / fx
    y = (yy - cy) * depth / fy
    z = depth

    # Stack the coordinates to create an (N, 3) array of 3D points
    points = np.stack((x, y, z), axis=-1)

    # Normalize colors to [0, 1]
    colors = colors.astype(np.float32) / 255.0

    # Convert the NumPy arrays to PyTorch tensors and move them to the specified device
    points = torch.from_numpy(points).float().to(device)
    colors = torch.from_numpy(colors).float().to(device)
    print(f"Points shape: {points.shape}")
    print(f"Colors shape: {colors.shape}")

    # Create a PyTorch3D Pointclouds object with colors
    point_cloud = Pointclouds(points=[points], features=[colors])
    # Print the value of the first point of the point cloud
    print(f"First point: {points[0]}")
    return point_cloud

def icp(A, B, max_iterations=20, tolerance=1e-5):
    """
    Perform Iterative Closest Point (ICP) algorithm to align point cloud B to point cloud A.

    Args:
        A (Pointclouds): The target point cloud.
        B (Pointclouds): The source point cloud to be aligned.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.

    Returns:
        Pointclouds: The aligned source point cloud with features (colors).
    """
    # Convert point clouds to numpy arrays
    A_points = A.points_padded().squeeze().cpu().numpy()
    B_points = B.points_padded().squeeze().cpu().numpy()

    A_colors = A.features_padded().squeeze().cpu().numpy()
    B_colors = B.features_padded().squeeze().cpu().numpy()

    # Initialize variables
    prev_error = None
    T = np.eye(4)

    for i in range(max_iterations):
        print(f"Iteration {i+1}/{max_iterations}")

        # Build KD-Tree for target point cloud
        tree = cKDTree(A_points)

        # Find the nearest neighbors for each point in B
        distances, indices = tree.query(B_points)
        closest_points = A_points[indices]

        # Compute the centroids of the point clouds
        centroid_A = np.mean(closest_points, axis=0)
        centroid_B = np.mean(B_points, axis=0)
        print(f"Centroid A: {centroid_A}, Centroid B: {centroid_B}")

        # Center the point clouds
        AA = closest_points - centroid_A
        BB = B_points - centroid_B

        # Compute the covariance matrix
        H = BB.T @ AA
        print(f"Covariance matrix H:\n{H}")

        # Compute the Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        print(f"Rotation matrix R:\n{R}")

        # Ensure a proper rotation (det(R) should be 1)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        print(f"Adjusted rotation matrix R:\n{R}")

        t = centroid_A - R @ centroid_B
        print(f"Translation vector t: {t}")

        # Update the transformation matrix
        T_new = np.eye(4)
        T_new[:3, :3] = R
        T_new[:3, 3] = t
        T = T_new @ T
        print(f"Transformation matrix T:\n{T}")

        # Apply the transformation to B_points
        B_points = (R @ B_points.T).T + t

        # Check for convergence
        mean_error = np.mean(distances)
        print(f"Mean error: {mean_error}")

        if prev_error is not None and abs(prev_error - mean_error) < tolerance:
            print("Convergence reached.")
            break
        prev_error = mean_error
        # Visualize the point clouds A and transformed B at this iteration
        A_pcd = o3d.geometry.PointCloud()
        A_pcd.points = o3d.utility.Vector3dVector(A_points)
        A_pcd.paint_uniform_color([1, 0, 0])  # Red for point cloud A

        B_pcd = o3d.geometry.PointCloud()
        B_pcd.points = o3d.utility.Vector3dVector(B_points)
        B_pcd.paint_uniform_color([0, 1, 0])  # Green for point cloud B

        o3d.visualization.draw_geometries([A_pcd, B_pcd])

    # Convert back to tensors and create aligned point cloud with features
    B_points_tensor = torch.from_numpy(B_points).float().to(B.device)
    B_colors_tensor = torch.from_numpy(B_colors).float().to(B.device)
    aligned_B = Pointclouds(points=[B_points_tensor], features=[B_colors_tensor])

    return aligned_B

def icp_ransac(A, B, max_iterations=20, tolerance=1e-5, ransac_iterations=100, inlier_threshold=0.05):
    """
    Perform Iterative Closest Point (ICP) algorithm with RANSAC to align point cloud B to point cloud A.

    Args:
        A (Pointclouds): The target point cloud.
        B (Pointclouds): The source point cloud to be aligned.
        max_iterations (int): Maximum number of ICP iterations.
        tolerance (float): Convergence tolerance.
        ransac_iterations (int): Number of RANSAC iterations within each ICP iteration.
        inlier_threshold (float): Distance threshold to consider a point as an inlier.

    Returns:
        Pointclouds: The aligned source point cloud with features (colors).
    """
    # Convert point clouds to numpy arrays
    A_points = A.points_padded().squeeze().cpu().numpy()
    B_points = B.points_padded().squeeze().cpu().numpy()

    A_colors = A.features_padded().squeeze().cpu().numpy()
    B_colors = B.features_padded().squeeze().cpu().numpy()

    # Initialize variables
    prev_error = None
    T = np.eye(4)

    for i in range(max_iterations):
        print(f"Iteration {i+1}/{max_iterations}")

        # Build KD-Tree for target point cloud
        tree = cKDTree(A_points)

        # Find the nearest neighbors for each point in B
        distances, indices = tree.query(B_points)
        closest_points = A_points[indices]

        # Initialize RANSAC variables
        best_inliers = []
        best_R = None
        best_t = None
        max_inliers = 0

        for r in range(ransac_iterations):
            # Randomly select a subset of correspondences
            idx = np.random.choice(len(B_points), size=3, replace=False)
            B_sample = B_points[idx]
            A_sample = closest_points[idx]

            # Compute the centroids of the samples
            centroid_A = np.mean(A_sample, axis=0)
            centroid_B = np.mean(B_sample, axis=0)

            # Center the samples
            AA = A_sample - centroid_A
            BB = B_sample - centroid_B

            # Compute the covariance matrix
            H = BB.T @ AA

            # Compute the Singular Value Decomposition (SVD)
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            # Ensure a proper rotation
            if np.linalg.det(R) < 0:
                Vt[2, :] *= -1
                R = Vt.T @ U.T

            t = centroid_A - R @ centroid_B

            # Apply transformation to all B_points
            B_transformed = (R @ B_points.T).T + t

            # Compute residuals (distances to closest points in A)
            residuals = np.linalg.norm(B_transformed - closest_points, axis=1)

            # Identify inliers
            inliers = np.where(residuals < inlier_threshold)[0]
            num_inliers = len(inliers)

            # Update the best model if current one is better
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_inliers = inliers
                best_R = R
                best_t = t

        print(f"Number of inliers: {max_inliers}")

        # If no inliers were found, break the loop
        if max_inliers == 0:
            print("No inliers found. Exiting RANSAC.")
            break

        # Recompute transformation using all inliers
        B_inliers = B_points[best_inliers]
        A_inliers = closest_points[best_inliers]

        # Compute centroids of inliers
        centroid_A = np.mean(A_inliers, axis=0)
        centroid_B = np.mean(B_inliers, axis=0)

        # Center the inliers
        AA = A_inliers - centroid_A
        BB = B_inliers - centroid_B

        # Compute covariance matrix
        H = BB.T @ AA

        # Compute SVD
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure a proper rotation
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = centroid_A - R @ centroid_B

        # Update the transformation matrix
        T_new = np.eye(4)
        T_new[:3, :3] = R
        T_new[:3, 3] = t
        T = T_new @ T
        print(f"Updated transformation matrix T:\n{T}")

        # Apply the transformation to B_points
        B_points = (R @ B_points.T).T + t

        # Compute mean error using inliers
        mean_error = np.mean(distances[best_inliers])
        print(f"Mean error: {mean_error}")

        # Check for convergence
        if prev_error is not None and abs(prev_error - mean_error) < tolerance:
            print("Convergence reached.")
            break
        prev_error = mean_error

        # Optional: Visualize the point clouds A and transformed B at this iteration
        # Uncomment the following lines if you wish to visualize each iteration
        
        A_pcd = o3d.geometry.PointCloud()
        A_pcd.points = o3d.utility.Vector3dVector(A_points)
        A_pcd.paint_uniform_color([1, 0, 0])  # Red for point cloud A

        B_pcd = o3d.geometry.PointCloud()
        B_pcd.points = o3d.utility.Vector3dVector(B_points)
        B_pcd.paint_uniform_color([0, 1, 0])  # Green for point cloud B

        o3d.visualization.draw_geometries([A_pcd, B_pcd])
        

    # Convert back to tensors and create aligned point cloud with features
    B_points_tensor = torch.from_numpy(B_points).float().to(B.device)
    B_colors_tensor = torch.from_numpy(B_colors).float().to(B.device)
    aligned_B = Pointclouds(points=[B_points_tensor], features=[B_colors_tensor])

    return aligned_B

def icp_with_overlap_filtering(
    A, B, max_iterations=40, tolerance=1e-5, initial_distance_threshold=10, final_distance_threshold=0.5, visualize = False
):
    """
    Perform ICP algorithm to align point cloud B to point cloud A with filtering of non-overlapping regions.

    Args:
        A (Pointclouds): The reference (first frame) point cloud.
        B (Pointclouds): The source point cloud to be aligned.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.
        initial_distance_threshold (float): Initial threshold to filter out non-overlapping regions.
        final_distance_threshold (float): Final threshold value for the last iteration.

    Returns:
        Pointclouds: The aligned source point cloud with features (colors).
    """
    # Convert point clouds to numpy arrays
    A_points = A.points_padded().squeeze().cpu().numpy()
    B_points = B.points_padded().squeeze().cpu().numpy()

    A_colors = A.features_padded().squeeze().cpu().numpy()
    B_colors = B.features_padded().squeeze().cpu().numpy()

    # Initialize variables
    prev_error = None
    T = np.eye(4)
    errors_over_time = []

    # Initialize distance threshold
    distance_threshold = initial_distance_threshold

    for i in range(max_iterations):
        print(f"Iteration {i+1}/{max_iterations}")

        # Build KD-Tree for target point cloud
        tree = cKDTree(A_points)

        # Find the nearest neighbors for each point in B
        distances, indices = tree.query(B_points)
        closest_points = A_points[indices]

        # Adjust the distance threshold dynamically
        mean_distance = np.mean(distances)
        median_distance = np.median(distances)
        percentile_90 = np.percentile(distances, 90)

        # Option 1: Based on mean distance
        # distance_threshold = mean_distance * 1.5

        # Option 2: Based on median distance
        # distance_threshold = median_distance * 2.0

        # Option 3: Based on a specific percentile
        distance_threshold = percentile_90

        # Optionally, decrease the threshold over iterations
        alpha = i / (max_iterations - 1)
        distance_threshold = (1 - alpha) * initial_distance_threshold + alpha * final_distance_threshold

        print(f"Adaptive distance threshold: {distance_threshold}")

        # Filter out correspondences with distances greater than the threshold
        mask = distances < distance_threshold
        filtered_B_points = B_points[mask]
        filtered_closest_points = closest_points[mask]
        filtered_distances = distances[mask]

        print(f"Number of correspondences after filtering: {len(filtered_B_points)}")

        # If not enough correspondences, break
        if len(filtered_B_points) < 3:
            print("Not enough correspondences after filtering. Exiting ICP.")
            break

        # Compute the centroids of the point clouds
        centroid_A = np.mean(filtered_closest_points, axis=0)
        centroid_B = np.mean(filtered_B_points, axis=0)
        print(f"Centroid A: {centroid_A}, Centroid B: {centroid_B}")

        # Center the point clouds
        AA = filtered_closest_points - centroid_A
        BB = filtered_B_points - centroid_B

        # Compute the covariance matrix
        H = BB.T @ AA
        print(f"Covariance matrix H:\n{H}")

        # Compute the Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        print(f"Rotation matrix R:\n{R}")

        # Ensure a proper rotation (det(R) should be 1)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        print(f"Adjusted rotation matrix R:\n{R}")

        t = centroid_A - R @ centroid_B
        print(f"Translation vector t: {t}")

        # Update the transformation matrix
        T_new = np.eye(4)
        T_new[:3, :3] = R
        T_new[:3, 3] = t
        T = T_new @ T
        print(f"Transformation matrix T:\n{T}")

        # Apply the transformation to B_points
        B_points = (R @ B_points.T).T + t

        # Collect mean error for visualization
        mean_error = np.mean(filtered_distances)
        errors_over_time.append(mean_error)
        print(f"Mean error: {mean_error}")

        # Check for convergence
        if prev_error is not None and abs(prev_error - mean_error) < tolerance:
            print("Convergence reached.")
            break
        prev_error = mean_error

        # Visualize the point clouds with the specified colors
        if visualize:
            visualize_alignment(
                A_points, B_points, mask, iteration=i, save_visualization=False
            )
        # visualize_alignment(
        #    A_points, B_points, mask, iteration=i, save_visualization=False
        # )

    # Convert back to tensors and create aligned point cloud with features
    B_points_tensor = torch.from_numpy(B_points).float().to(B.device)
    B_colors_tensor = torch.from_numpy(B_colors).float().to(B.device)
    aligned_B = Pointclouds(points=[B_points_tensor], features=[B_colors_tensor])

    # Save the error over iterations plot
    save_error_plot(errors_over_time)

    return aligned_B

def icp_open3d(A, B, max_iterations=20, tolerance=1e-5):
    """
    Perform Iterative Closest Point (ICP) algorithm using Open3D to align point cloud B to point cloud A.

    Args:
        A (Pointclouds): The target point cloud.
        B (Pointclouds): The source point cloud to be aligned.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.

    Returns:
        Pointclouds: The aligned source point cloud with features (colors).
    """
    # Convert point clouds to numpy arrays
    A_points = A.points_padded().squeeze().cpu().numpy()
    B_points = B.points_padded().squeeze().cpu().numpy()

    A_colors = A.features_padded().squeeze().cpu().numpy()
    B_colors = B.features_padded().squeeze().cpu().numpy()

    # Create Open3D point clouds
    A_pcd = o3d.geometry.PointCloud()
    A_pcd.points = o3d.utility.Vector3dVector(A_points)
    A_pcd.colors = o3d.utility.Vector3dVector(A_colors)

    B_pcd = o3d.geometry.PointCloud()
    B_pcd.points = o3d.utility.Vector3dVector(B_points)
    B_pcd.colors = o3d.utility.Vector3dVector(B_colors)

    # Perform ICP using Open3D
    threshold = 10  # Distance threshold for ICP
    transformation = np.eye(4)
    for i in range(max_iterations):
        result_icp = o3d.pipelines.registration.registration_icp(
            B_pcd, A_pcd, threshold, transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)
        )
        transformation = result_icp.transformation
        print(f"Iteration {i+1}/{max_iterations}, Transformation:\n{transformation}")
        mean_error = np.mean(result_icp.inlier_rmse)
        print(f"Iteration {i+1}/{max_iterations}, Mean error: {mean_error}")

        if mean_error < tolerance:
            print("Convergence reached.")
            break

    # Apply the final transformation to B_points
    B_points_transformed = np.asarray(B_pcd.points)
    B_points_tensor = torch.from_numpy(B_points_transformed).float().to(B.device)
    B_colors_tensor = torch.from_numpy(B_colors).float().to(B.device)
    aligned_B = Pointclouds(points=[B_points_tensor], features=[B_colors_tensor])

    return aligned_B


def visualize_alignment(A_points, B_points, mask, iteration, save_visualization=False):
    """
    Visualize the point clouds with specified colors.

    Args:
        A_points (numpy.ndarray): The reference point cloud points.
        B_points (numpy.ndarray): The transformed source point cloud points.
        mask (numpy.ndarray): Boolean array indicating points used in alignment.
        iteration (int): The current iteration number.
        save_visualization (bool): Whether to save the visualization to a file.
    """
    # Create Open3D point clouds
    A_pcd = o3d.geometry.PointCloud()
    A_pcd.points = o3d.utility.Vector3dVector(A_points)
    A_pcd.paint_uniform_color([1, 0, 0])  # Red for point cloud A

    B_pcd_used = o3d.geometry.PointCloud()
    B_pcd_used.points = o3d.utility.Vector3dVector(B_points[mask])
    B_pcd_used.paint_uniform_color([0.5, 0, 0.5])  # Purple for used points in B

    B_pcd_unused = o3d.geometry.PointCloud()
    B_pcd_unused.points = o3d.utility.Vector3dVector(B_points[~mask])
    B_pcd_unused.paint_uniform_color([0, 1, 0])  # Green for unused points in B

    # Combine all point clouds
    combined_pcd = A_pcd + B_pcd_used + B_pcd_unused

    if save_visualization:
        # Use Open3D's offscreen renderer to save the visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(A_pcd)
        vis.add_geometry(B_pcd_used)
        vis.add_geometry(B_pcd_unused)
        vis.poll_events()
        vis.update_renderer()
        # Capture the screen
        image = vis.capture_screen_float_buffer()
        plt.imsave(f'alignment_iteration_{iteration}.png', np.asarray(image), dpi=300)
        vis.destroy_window()
    else:
        # Show the visualization (if running in an environment that supports it)
        o3d.visualization.draw_geometries([A_pcd, B_pcd_used, B_pcd_unused])

def save_error_plot(errors_over_time, filename="mean_error_over_iterations.png"):
    """
    Save the mean error over iterations plot.

    Args:
        errors_over_time (list): List of mean errors over iterations.
        filename (str): Filename to save the plot.
    """
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.subplots()
    ax.plot(range(len(errors_over_time)), errors_over_time, marker='o')
    ax.set_title('Mean Error Over Iterations')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Error')
    ax.grid(True)

    # Save the figure to a file
    fig.savefig(filename, dpi=300)

def visualize_pointcloud(pointcloud):
    points = pointcloud.points_padded().squeeze().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    plt.show()

def visualize_pointcloud_open3d(pointcloud):
    points = pointcloud.points_padded().squeeze().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def visualize_pointcloud_with_texture(point_cloud):
    """
    Visualizes a point cloud with texture using Open3D and adds a coordinate frame at the origin.

    Args:
        point_cloud (Pointclouds): A PyTorch3D Pointclouds object containing the point cloud data with colors.
    """
    # Extract points and colors from the PyTorch3D Pointclouds object
    points = point_cloud.points_padded()[0]  # Shape: (N, 3)
    colors = point_cloud.features_padded()[0]  # Shape: (N, 3)

    # Move data to CPU and convert to NumPy arrays
    points_np = points.cpu().numpy()
    colors_np = colors.cpu().numpy()

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    
    # Assign the point coordinates
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # Assign the colors to the point cloud
    # Ensure that colors are in the range [0, 1]
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    # Create a coordinate frame at the origin
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

    # Visualize the point cloud with the coordinate frame
    o3d.visualization.draw_geometries([pcd, coordinate_frame])

def generate_random_pointcloud(num_points, device):
    """
    Generates a random point cloud with the specified number of points.

    Args:
        num_points (int): The number of points to generate.
        device (torch.device): The device to use ('cpu' or 'cuda').

    Returns:
        Pointclouds: A PyTorch3D Pointclouds object containing the random point cloud.
    """
    # Generate random 3D coordinates in the range [-1, 1]
    points = torch.rand((num_points, 3), device=device) * 2 - 1

    # Optionally, generate random RGB colors for each point
    colors = torch.rand((num_points, 3), device=device)

    # Create a PyTorch3D Pointclouds object
    point_cloud = Pointclouds(points=[points], features=[colors])

    return point_cloud


def output_ply(pointcloud):
    # Extract point positions
    points_tensor = pointcloud.points_list()[0]
    pointcloud_np = points_tensor.cpu().numpy()

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud_np)

    # Check if the point cloud has color features
    if pointcloud.features_list():
        colors_tensor = pointcloud.features_list()[0]
        colors_np = colors_tensor.cpu().numpy()

        # Normalize colors if they are not in [0, 1]
        if colors_np.max() > 1.0:
            colors_np = colors_np / 255.0

        # Assign colors to the Open3D point cloud
        pcd.colors = o3d.utility.Vector3dVector(colors_np)

    # Save the point cloud to a PLY file
    now = datetime.now()
    current_time = now.strftime("%Y%m%d%H%M%S")
    ply_name = f"output/ICP_{current_time}.ply"
    o3d.io.write_point_cloud(ply_name, pcd)
    print("Point cloud saved to output folder")
    return ply_name



def merge_point_clouds(A, B, voxel_size=0.1):
    """
    Merge two aligned point clouds by combining overlapping points using a voxel grid.

    Args:
        A (Pointclouds): The first point cloud.
        B (Pointclouds): The second point cloud, aligned to A.
        voxel_size (float): The size of the voxel grid.

    Returns:
        Pointclouds: The merged point cloud.
    """
    # Concatenate the points and features from both point clouds
    points = torch.cat([A.points_padded(), B.points_padded()], dim=1)  # Shape: (1, N+M, 3)
    features = torch.cat([A.features_padded(), B.features_padded()], dim=1)  # Shape: (1, N+M, C)

    # Remove batch dimension
    points = points.squeeze(0)  # Shape: (N+M, 3)
    features = features.squeeze(0)  # Shape: (N+M, C)

    # Compute voxel indices for each point
    coords = torch.floor(points / voxel_size).int()

    # Combine points within the same voxel
    unique_coords, inverse_indices = torch.unique(coords, return_inverse=True, dim=0)

    # Initialize lists to hold merged points and features
    merged_points = []
    merged_features = []

    # Iterate over unique voxel coordinates
    for i in range(unique_coords.size(0)):
        # Find points that belong to the current voxel
        mask = (inverse_indices == i)
        voxel_points = points[mask]
        voxel_features = features[mask]

        # Compute the mean position and feature within the voxel
        mean_point = voxel_points.mean(dim=0)
        mean_feature = voxel_features.mean(dim=0)

        merged_points.append(mean_point)
        merged_features.append(mean_feature)

    # Stack merged points and features
    merged_points = torch.stack(merged_points).unsqueeze(0)  # Shape: (1, K, 3)
    merged_features = torch.stack(merged_features).unsqueeze(0)  # Shape: (1, K, C)

    # Create a new Pointclouds object
    merged_pointcloud = Pointclouds(points=merged_points, features=merged_features)

    return merged_pointcloud

def merge_point_clouds_kdtree(A, B, threshold_close=0.1, threshold_far=0.5, threshold_erroneous=5.0, keepallpoints=False):
    """
    Merge two aligned point clouds by removing duplicates, misaligned points, and erroneous points using a KD-Tree.

    Args:
        A (Pointclouds): The first point cloud.
        B (Pointclouds): The second point cloud, aligned to A.
        threshold_close (float): Distance threshold for duplicates.
        threshold_far (float): Distance threshold for new features.
        threshold_erroneous (float): Distance threshold beyond which points are considered erroneous.

    Returns:
        Pointclouds: The merged point cloud.
    """
    # Extract points and features from point clouds A and B
    A_points = A.points_padded().squeeze(0).cpu().numpy()
    A_features = A.features_padded().squeeze(0).cpu().numpy()

    B_points = B.points_padded().squeeze(0).cpu().numpy()
    B_features = B.features_padded().squeeze(0).cpu().numpy()

    # Build KD-Tree for point cloud A
    tree = cKDTree(A_points)

    # Find nearest neighbors in A for each point in B
    distances, indices = tree.query(B_points)

    # Create masks based on the thresholds
    #mask_duplicates = distances < threshold_close
    #mask_misaligned = (distances >= threshold_close) & (distances <= threshold_far)
    mask_new_features = (distances > threshold_far) & (distances <= threshold_erroneous)
    #mask_erroneous = distances > threshold_erroneous

    # Discard points from B that are duplicates, misaligned, or erroneous
    # Keep points from B that are new features

    print(f"Number of points in B: {B_points.shape[0]}")
    B_points_to_keep = B_points[mask_new_features]
    B_features_to_keep = B_features[mask_new_features]
    print(f"Number of points to keep from B: {B_points_to_keep.shape[0]}")
    # Combine A's points with the selected B points
    if not keepallpoints:
        merged_points = np.vstack((A_points, B_points_to_keep))
        merged_features = np.vstack((A_features, B_features_to_keep))
    else:
        merged_points = np.vstack((A_points, B_points))
        merged_features = np.vstack((A_features, B_features))

    # Convert back to tensors
    merged_points_tensor = torch.from_numpy(merged_points).float().unsqueeze(0).to(A.device)
    merged_features_tensor = torch.from_numpy(merged_features).float().unsqueeze(0).to(A.device)

    # Create the merged point cloud
    merged_pointcloud = Pointclouds(points=merged_points_tensor, features=merged_features_tensor)

    return merged_pointcloud


def generate_universal_mask(data_paths, threshold=10):
    """
    Generates a universal mask by finding pixels that consistently correspond to black borders
    across all input images.

    Args:
        data_paths (list of str): List of paths to the original input images.
        threshold (int): Pixel intensity value below which a pixel is considered black.

    Returns:
        np.ndarray: A binary mask where True indicates valid pixels, and False indicates black borders.
    """
    # Read the first image to get the dimensions
    first_image = cv2.imread(data_paths[0], cv2.IMREAD_GRAYSCALE)
    if first_image is None:
        raise ValueError(f"Could not read image from path: {data_paths[0]}")
    height, width = first_image.shape

    # Initialize a count mask
    count_mask = np.zeros((height, width), dtype=np.int32)
    num_images = len(data_paths)

    for image_path in data_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image from path: {image_path}")

        # Ensure the image matches the expected size
        if image.shape != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        # Create a binary mask where pixel intensity <= threshold (considered black)
        black_mask = image <= threshold

        # Increment the count mask at positions where the pixel is black
        count_mask[black_mask] += 1

    # Create the universal mask
    # Pixels that are black in all images are marked as False (invalid)
    universal_mask = count_mask < num_images

    return universal_mask

def analyze_point_cloud_distances(A, B, threshold_close=0.1, threshold_far=0.5, threshold_erroneous=5.0):
    """
    Analyze distances between two point clouds by finding pairs of points and computing
    minimum, mean, median, and maximum distances.

    Args:
        A (Pointclouds): The first point cloud.
        B (Pointclouds): The second point cloud.
        threshold_close (float): Distance threshold for duplicates.
        threshold_far (float): Distance threshold for new features.
        threshold_erroneous (float): Distance threshold beyond which points are considered erroneous.

    Returns:
        dict: A dictionary containing the minimum, mean, median, and maximum distances.
    """
    # Extract points from point clouds A and B
    A_points = A.points_padded().squeeze(0).cpu().numpy()
    B_points = B.points_padded().squeeze(0).cpu().numpy()

    # Build KD-Tree for point cloud A
    tree = cKDTree(A_points)

    # Find nearest neighbors in A for each point in B
    distances, indices = tree.query(B_points)

    # Filter distances based on thresholds
    #valid_distances = distances[(distances > threshold_close) & (distances <= threshold_erroneous)]
    valid_distances = distances 
    # Compute statistics
    min_distance = np.min(valid_distances)
    mean_distance = np.mean(valid_distances)
    median_distance = np.median(valid_distances)
    max_distance = np.max(valid_distances)
    # Compute the range of distances where 90% of the points fall in
    lower_percentile = np.percentile(valid_distances, 5)
    upper_percentile = np.percentile(valid_distances, 95)
    distance_range_90_percent = (lower_percentile, upper_percentile)
    std_distance = np.std(valid_distances)
    mean_plus_2std = mean_distance + 2 * std_distance
    mean_plus_3std = mean_distance + 3 * std_distance
    std23thresholds = (mean_plus_2std, mean_plus_3std)

    return {
        "min_distance": min_distance,
        "mean_distance": mean_distance,
        "median_distance": median_distance,
        "max_distance": max_distance,
        "distance_range_90_percent": distance_range_90_percent,
        "std_distance": std_distance,
        "std23thresholds": std23thresholds
    }

def recon(config):
    mute = config['IF_MUTE']
    in_path = config['DEPTH_FOLDER']
    data_path = config['DATA_PATH']
    print(in_path)
    firstTime = True
    device = torch.device("cuda:0" if torch.cuda.is_available() and config['USE_GPU'] else "cpu")
    if not mute:
        print(f'Device chosen: {device}')
    worldmap = Pointclouds(points=[])
    worldmap = worldmap.to(device)
    
    data_paths = []
    data_path_list = sorted([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))])
    for filename in data_path_list:
        data_paths.append(os.path.join(data_path, filename))
    universal_mask = generate_universal_mask(data_paths, threshold=10)
    # universal_mask[:] = 1 #Toggle to disable masking
    # Visualize the universal mask
    plt.imshow(universal_mask, cmap='gray')
    plt.title('Universal Mask')
    plt.show()
    file_list = sorted([f for f in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, f))])
    file_paths = []
    counter = 0
    global_distances = []
    for filename in file_list:
        file_paths.append(os.path.join(in_path, filename))
    for file_path in file_paths:
        print(f'Processing file: {file_path}')
        if firstTime:
            firstTime = False
            worldmap = register_depth_map_to_pointcloud_new(file_path, device, need_reverse=config['NEED_REVERSE'], focal_length=config['fx'],texture_folder=config['DATA_PATH'],universal_mask=universal_mask)
        else:
            new_pointcloud = register_depth_map_to_pointcloud_new(file_path, device, need_reverse=config['NEED_REVERSE'], focal_length=config['fx'],texture_folder=config['DATA_PATH'],universal_mask=universal_mask)   
            # Step 1: Align B to A using ICP
            
            if counter >=28:
                aligned_B = icp_with_overlap_filtering(worldmap, new_pointcloud, visualize=True)
            #worldmap = merge_point_clouds(worldmap, aligned_B)
            else:
                aligned_B = icp_with_overlap_filtering(worldmap, new_pointcloud)
            analyze_point_cloud_distances(worldmap, aligned_B)
            distances = analyze_point_cloud_distances(worldmap, aligned_B)
            global_distances.append(distances)
            #worldmap = merge_point_clouds_kdtree(worldmap, aligned_B,threshold_far=distances['std23thresholds'][0], threshold_erroneous=distances['std23thresholds'][1])
            worldmap = merge_point_clouds_kdtree(worldmap, aligned_B, keepallpoints=True)
            #worldmap = clean_cloud_gpu(worldmap)
            # # Step 2: Extract points and features from A and aligned_B
            # A_points = worldmap.points_padded()       # Shape: (1, N, 3)
            # A_features = worldmap.features_padded()   # Shape: (1, N, C)

            # B_points = aligned_B.points_padded()     # Shape: (1, M, 3)
            # B_features = aligned_B.features_padded() # Shape: (1, M, C)
            if counter%10 == 0:
                print(f'Counter: {counter}')
                visualize_pointcloud_with_texture(worldmap)
                output_ply(worldmap)
            counter += 1
            
            # # Step 3: Concatenate the points and features 
            # merged_points = torch.cat([A_points, B_points], dim=1)       # Shape: (1, N+M, 3)
            # merged_features = torch.cat([A_features, B_features], dim=1) # Shape: (1, N+M, C)

            # # Step 4: Create the merged point cloud
            # worldmap = Pointclouds(points=merged_points, features=merged_features)

            #worldmap = icp(worldmap, new_pointcloud)
    visualize_pointcloud_with_texture(worldmap)
    print(global_distances)
    return output_ply(worldmap)
    # return output_ply(worldmap)
    # visualize_pointcloud_open3d(worldmap)
    # rp = generate_random_pointcloud(1000, device)
    # visualize_pointcloud_open3d(rp)



#recon({'IF_MUTE': False, 'DATA_PATH' : 'data/Hamlyn/rectified01/image01', 'DEPTH_FOLDER': '/home/yicheng/Github/pipeline/data/Hamlyn/rectified01/depth01','NEED_REVERSE':False, 'USE_GPU': True, 'fx':383.1901395})
#recon({'IF_MUTE': False, 'DATA_PATH' : 'data/Hamlyn/rectified17/image01', 'DEPTH_FOLDER': '/home/yicheng/Github/pipeline/data/Hamlyn/rectified17/depth01','NEED_REVERSE':False, 'USE_GPU': True, 'fx':417.9036255})
#recon({'IF_MUTE': False, 'DATA_PATH' : 'data/Hamlyn/rectified01/image01', 'DEPTH_FOLDER': '/home/yicheng/Github/pipeline/output/depth-anything:v2_20241108120532_output','NEED_REVERSE':True, 'USE_GPU': True, 'fx':383.1901395})
#recon({'IF_MUTE': False, 'DATA_PATH' : 'data/Hamlyn/rectified08/image01', 'DEPTH_FOLDER': '/home/yicheng/Github/pipeline/data/Hamlyn/rectified08/depth01_63','NEED_REVERSE':False, 'USE_GPU': True, 'fx':765.8236885})
#recon({'IF_MUTE': False, 'DATA_PATH' : 'data/Hamlyn/rectified08/image01', 'DEPTH_FOLDER': '/home/yicheng/Github/pipeline/data/Hamlyn/rectified08/depth01_picked','NEED_REVERSE':False, 'USE_GPU': True, 'fx':765.8236885})
#recon({'IF_MUTE': False, 'DATA_PATH' : 'data/Hamlyn/rectified22/image', 'DEPTH_FOLDER': '/home/yicheng/Github/pipeline/data/Hamlyn/rectified22/depth','NEED_REVERSE':False, 'USE_GPU': True, 'fx':417.903625})
#recon({'IF_MUTE': False, 'DATA_PATH' : 'data/Hamlyn/test22/color', 'DEPTH_FOLDER': '/home/yicheng/Github/pipeline/data/Hamlyn/test22/depth','NEED_REVERSE':False, 'USE_GPU': True, 'fx':417.903625})
