import torch, os
from pytorch3d.structures import Pointclouds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial import cKDTree
from datetime import datetime

def register_depth_map_to_pointcloud_new(depth_map_path, device, need_reverse, focal_length, texture_folder, universal_mask=None, depth_scale=1.0, req_depth_range = None):
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



    # Convert depth map to float32 and apply scaling factor
    depth_map = depth_map.astype(np.float32) * depth_scale

    # Print depth map statistics
    print(f"Processing depth map: {depth_map_path}")
    print(f"Depth map min: {depth_map.min()}, max: {depth_map.max()}")
    # Convert depth map from RGBA or RGB to grayscale if necessary
    if len(depth_map.shape) == 3:
        if depth_map.shape[2] == 4:
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGBA2GRAY)
        elif depth_map.shape[2] == 3:
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)
    min_matrix = np.min(depth_map)
    max_matrix = np.max(depth_map)
    if req_depth_range is not None:
        min_ran, max_ran = req_depth_range
        print('!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!')
        depth_map = ((depth_map - min_matrix) / (max_matrix - min_matrix)) * (max_ran - min_ran) + min_ran
    else:
        req_depth_range = [min_matrix, max_matrix]
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

    # Create a valid mask based on depth > 0 and depth < 255
    valid = np.logical_and(depth > 0, depth < 255)

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

    # Create a PyTorch3D Pointclouds object with colors
    point_cloud = Pointclouds(points=[points], features=[colors])

    return point_cloud, width, height, req_depth_range
    

def icp(A, B, max_iterations=20, tolerance=1e-5):
    """
    Perform Iterative Closest Point (ICP) algorithm to align point cloud B to point cloud A.

    Args:
        A (Pointclouds): The target point cloud.
        B (Pointclouds): The source point cloud to be aligned.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.

    Returns:
        (Pointclouds, np.ndarray): The aligned source point cloud with features (colors) and the transformation matrix.
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

        # Center the point clouds
        AA = closest_points - centroid_A
        BB = B_points - centroid_B

        # Compute the covariance matrix
        H = BB.T @ AA

        # Compute the Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure a proper rotation (det(R) should be 1)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = centroid_A - R @ centroid_B

        # Update the transformation matrix
        T_new = np.eye(4)
        T_new[:3, :3] = R
        T_new[:3, 3] = t
        T = T_new @ T

        # Apply the transformation to B_points
        B_points = (R @ B_points.T).T + t

        # Check for convergence
        mean_error = np.mean(distances)

        if prev_error is not None and abs(prev_error - mean_error) < tolerance:
            print("Convergence reached.")
            break
        prev_error = mean_error

    # Convert back to tensors and create aligned point cloud with features
    B_points_tensor = torch.from_numpy(B_points).float().to(B.device)
    B_colors_tensor = torch.from_numpy(B_colors).float().to(B.device)
    aligned_B = Pointclouds(points=[B_points_tensor], features=[B_colors_tensor])

    return aligned_B, T

def create_error_image(correspondences, source_points, target_points, K, width, height):
    # heatmap = np.zeros((height, width), dtype=float) #This works for o3d
    # for source_idx, target_idx in correspondences:
    #     source_point = source_points.points[source_idx]
    #     target_point = target_points.points[target_idx]
    #     i, j = project_to_image(K, source_point)
    #     if 0 <= int(j) < height and 0 <= int(i) < width:
    #         heatmap[int(j), int(i)] = np.linalg.norm(source_point - target_point)

    source_indices, target_indices = correspondences
    heatmap = np.zeros((height, width), dtype=float)
    for src_idx, tgt_idx in zip(source_indices, target_indices):
        source_point = source_points[src_idx]
        target_point = target_points[tgt_idx]
        i, j = project_to_image(K, source_point)
        if 0 <= int(j) < height and 0 <= int(i) < width:
            heatmap[int(j), int(i)] = np.linalg.norm(source_point - target_point)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max() * 255
    else:
        heatmap = heatmap * 255
    heatmap = heatmap.astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # Display the heatmap
    # plt.imshow(heatmap)
    # plt.title('Alignment Error Heatmap')
    # plt.axis('off')
    # plt.show()
    return colored_heatmap

def create_error_image_o3d(correspondences, source_points, target_points, K, width, height):
    heatmap = np.zeros((height, width), dtype=float) #This works for o3d
    for source_idx, target_idx in correspondences:
        source_point = source_points.points[source_idx]
        target_point = target_points.points[target_idx]
        i, j = project_to_image(K, source_point)
        if 0 <= int(j) < height and 0 <= int(i) < width:
            heatmap[int(j), int(i)] = np.linalg.norm(source_point - target_point)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max() * 255
    else:
        heatmap = heatmap * 255
    heatmap = heatmap.astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return colored_heatmap

def project_to_image(K, point_3D):
    homogeneous_3D = np.append(point_3D, 1)
    image_coords_homogeneous = K @ homogeneous_3D[:3]
    i = image_coords_homogeneous[0] / image_coords_homogeneous[2]
    j = image_coords_homogeneous[1] / image_coords_homogeneous[2]
    return i, j

def icp_with_overlap_filtering(
    A, B, max_iterations=10, tolerance=1e-5,
    initial_distance_threshold=10, final_distance_threshold=10,
    visualize=False, K=None, width=None, height=None, output_video=None
):
    """
    Perform ICP algorithm to align point cloud B to point cloud A with filtering of non-overlapping regions
    and generate an error video showing alignment errors during each iteration.

    Args:
        A (Pointclouds): The reference (first frame) point cloud.
        B (Pointclouds): The source point cloud to be aligned.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.
        initial_distance_threshold (float): Initial threshold to filter out non-overlapping regions.
        final_distance_threshold (float): Final threshold value for the last iteration.
        visualize (bool): Whether to visualize the alignment (optional).
        K (np.ndarray): Camera intrinsic matrix (3x3).
        width (int): Image width.
        height (int): Image height.
        output_video (str): Path to save the output video (optional).

    Returns:
        Pointclouds: The aligned source point cloud with features (colors).
        np.ndarray: The final transformation matrix.
    """
    # Convert point clouds to numpy arrays
    A_points = A.points_padded().squeeze(0).cpu().numpy()
    B_points = B.points_padded().squeeze(0).cpu().numpy()

    A_colors = A.features_padded().squeeze(0).cpu().numpy()
    B_colors = B.features_padded().squeeze(0).cpu().numpy()

    # Initialize variables
    prev_error = None
    T = np.eye(4)
    errors_over_time = []
    global_errors_over_time = []
    thresholds_over_time = []
    percentages_over_time = []

    # Initialize distance threshold
    distance_threshold = initial_distance_threshold

    # Initialize video writer if needed
    if K is not None and output_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 1  # Frames per second
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (int(width), int(height)))

    for i in range(max_iterations):
        print(f"Iteration {i+1}/{max_iterations}")

        # Build KD-Tree for target point cloud
        tree = cKDTree(A_points)

        # Find the nearest neighbors for each point in B
        distances, indices = tree.query(B_points)
        closest_points = A_points[indices]

        # Compute global error before filtering
        global_error = np.mean(distances)
        global_errors_over_time.append(global_error)

        # Adjust the distance threshold dynamically
        #percentile_90 = np.percentile(distances, 90)
        #distance_threshold = percentile_90
        # mean_distance = np.mean(distances)
        # distance_threshold = 1.5 * mean_distance        
        # median_distance = np.median(distances)
        # distance_threshold = 1.5 * median_distance
        # alpha = i / (max_iterations - 1)
        # distance_threshold = (1 - alpha) * initial_distance_threshold + alpha * final_distance_threshold
        global_std_error = np.std(distances)
        #max_error = np.max(distances)
        #distance_threshold = max_error * 0.8
        distance_threshold = global_error + 2 * global_std_error
        print(f"Adaptive distance threshold: {distance_threshold}")

        # Append the threshold
        thresholds_over_time.append(distance_threshold)

        # Filter out correspondences with distances greater than the threshold
        mask = distances < distance_threshold
        filtered_B_points = B_points[mask]
        filtered_closest_points = closest_points[mask]
        filtered_indices = indices[mask]
        filtered_distances = distances[mask]

        # Compute percentage of points kept
        percentage_kept = len(filtered_B_points) / len(B_points)
        percentages_over_time.append(percentage_kept)
        print(f"Percentage of points kept: {percentage_kept * 100:.2f}%")

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
        std_error = np.std(filtered_distances)
        errors_over_time.append(mean_error)
        print(f"Mean error (filtered points): {mean_error}")

        # Generate error image and write to video
        if K is not None and output_video is not None:
            error_image = create_error_image(
                (np.arange(len(B_points))[mask], filtered_indices),
                B_points, A_points, K, int(width), int(height)
            )
            # Write frame to video
            video_writer.write(error_image)

        # Check for convergence
        if prev_error is not None and abs(prev_error - mean_error) < tolerance:
            print("Convergence reached.")
            break
        prev_error = mean_error

        # Visualize the point clouds with the specified colors (optional)
        if visualize:
            visualize_alignment(
                A_points, B_points, mask, iteration=i, save_visualization=False
            )

    # Release video writer if used
    if K is not None and output_video is not None:
        video_writer.release()
        print(f"Video saved to {output_video}")

    # Convert back to tensors and create aligned point cloud with features
    B_points_tensor = torch.from_numpy(B_points).float().to(B.device)
    B_colors_tensor = torch.from_numpy(B_colors).float().to(B.device)
    aligned_B = Pointclouds(points=[B_points_tensor], features=[B_colors_tensor])

    # Save the error over iterations plot
    save_error_plot(
        errors_over_time,
        global_errors_over_time,
        thresholds_over_time,
        percentages_over_time,
        filename = f"output/errorplots/mean_error_over_iterations_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    )

    return aligned_B, T, global_error,global_std_error #return the matrix, error, and its std deviation

def align_point_clouds(pcd1, pcd2, width, height, K=None,
                       max_iterations=10,
                       output_video="alignment_error_video.avi"):
    """
    Align two point clouds using ICP and generate an error video.
    
    Args:
        pcd1 (o3d.geometry.PointCloud): First point cloud.
        pcd2 (o3d.geometry.PointCloud): Second point cloud to be aligned to the first.
        width (int): Image width.
        height (int): Image height.
        K (np.ndarray): Camera intrinsic matrix (3x3).
        max_iterations (int): Maximum number of ICP iterations.
        output_video (str): Path to save the error video.

    Returns:
        o3d.geometry.PointCloud: Aligned second point cloud.
        np.ndarray: Final transformation matrix.
    """
    # Video writer setup
    if K is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 1  # Frames per second
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Initial alignment with identity transformation
    threshold = 10 # Adjust threshold based on your units (e.g., meters)

    # Initialize ICP registration object
    reg_icp = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)
    )

    for i in range(max_iterations):
        # Apply ICP transformation
        reg_icp = o3d.pipelines.registration.registration_icp(
            pcd2, pcd1, threshold, reg_icp.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)
        )
        pcd2.transform(reg_icp.transformation)

        if K is not None:
            # Create error heatmap image
            img = create_error_image_o3d(reg_icp.correspondence_set, pcd2, pcd1,
                                     K, width, height)
            # Write frame to video
            video_writer.write(img)

    # Release the video writer
    if K is not None:
        video_writer.release()
        print(f"Video saved to {output_video}")

    return pcd1, reg_icp.transformation


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

def save_error_plot(errors_over_time, global_errors_over_time, thresholds_over_time, percentages_over_time, filename="mean_error_over_iterations.png"):
    """
    Save the mean error over iterations plot, along with global error, threshold, and percentage of points kept.

    Args:
        errors_over_time (list): List of mean errors over filtered points over iterations.
        global_errors_over_time (list): List of global mean errors over iterations.
        thresholds_over_time (list): List of thresholds chosen over iterations.
        percentages_over_time (list): List of percentages of points kept over iterations.
        filename (str): Filename to save the plot.
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    iterations = range(1, len(errors_over_time) + 1)

    # Subplot 1: Mean error over iterations (filtered points)
    axs[0, 0].plot(iterations, errors_over_time, marker='o')
    axs[0, 0].set_title('Mean Error Over Iterations (Filtered Points)')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Mean Error')
    axs[0, 0].grid(True)

    # Subplot 2: Global error over iterations (all points)
    axs[0, 1].plot(iterations, global_errors_over_time, marker='o', color='orange')
    axs[0, 1].set_title('Global Mean Error Over Iterations (All Points)')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Global Mean Error')
    axs[0, 1].grid(True)

    # Subplot 3: Threshold chosen over iterations
    axs[1, 0].plot(iterations, thresholds_over_time, marker='o', color='green')
    axs[1, 0].set_title('Threshold Chosen Over Iterations')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Threshold')
    axs[1, 0].grid(True)

    # Subplot 4: Percentage of points kept ovqer iterations
    percentages = [p * 100 for p in percentages_over_time]  # Convert to percentage
    axs[1, 1].plot(iterations, percentages, marker='o', color='red')
    axs[1, 1].set_title('Percentage of Points Kept Over Iterations')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Percentage (%)')
    axs[1, 1].grid(True)

    plt.tight_layout()

    # Save the figure to a file
    plt.savefig(filename, dpi=300)
    plt.close(fig)  # Close the figure to free memory


def pointclouds_to_open3d(pointcloud):
    """
    Convert a PyTorch3D Pointclouds object to an Open3D PointCloud object.
    
    Args:
        pointcloud (Pointclouds): A PyTorch3D Pointclouds object.

    Returns:
        o3d.geometry.PointCloud: An Open3D PointCloud object.
    """
    # Extract points and features from the Pointclouds object
    points = pointcloud.points_padded()[0]  # Assuming batch size of 1
    colors = pointcloud.features_padded()[0]  # Assuming features are colors

    # Convert to numpy arrays
    points_np = points.cpu().numpy()
    colors_np = colors.cpu().numpy()

    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # If colors are available, assign them
    if colors_np is not None:
        # Ensure colors are in [0, 1]
        if colors_np.max() > 1.0:
            colors_np = colors_np / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_np)

    return pcd
def open3d_to_pointclouds(pcd, device):
    """
    Convert an Open3D PointCloud object to a PyTorch3D Pointclouds object.
    
    Args:
        pcd (o3d.geometry.PointCloud): An Open3D PointCloud object.
        device (torch.device): The device to use.

    Returns:
        Pointclouds: A PyTorch3D Pointclouds object.
    """
    points_np = np.asarray(pcd.points)
    colors_np = np.asarray(pcd.colors)

    points = torch.from_numpy(points_np).float().to(device)
    colors = torch.from_numpy(colors_np).float().to(device)

    pointcloud = Pointclouds(points=[points], features=[colors])

    return pointcloud

def apply_transformation(point_cloud, T):
    """
    Apply a 4x4 transformation matrix T to a PyTorch3D Pointclouds object.

    Args:
        point_cloud (Pointclouds): The point cloud to transform.
        T (np.ndarray): A 4x4 transformation matrix.

    Returns:
        Pointclouds: The transformed point cloud.
    """
    # Extract points from the point cloud
    points = point_cloud.points_padded()[0]  # Shape: (N, 3)

    # Convert to homogeneous coordinates
    ones = torch.ones((points.shape[0], 1), device=points.device)
    points_hom = torch.cat([points, ones], dim=1)  # Shape: (N, 4)

    # Convert T to torch tensor
    T_tensor = torch.from_numpy(T).float().to(points.device)

    # Apply transformation
    transformed_points_hom = (T_tensor @ points_hom.T).T  # Shape: (N, 4)

    # Convert back to Cartesian coordinates
    transformed_points = transformed_points_hom[:, :3]

    # Create new Pointclouds object with transformed points and same features
    transformed_point_cloud = Pointclouds(points=[transformed_points], features=point_cloud.features_list())

    return transformed_point_cloud

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

def merge_point_clouds_kdtree(A, B, threshold_close=0.1, threshold_far=2, threshold_erroneous=5.0):
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
    mask_new_features = (distances > threshold_far) & (distances <= threshold_erroneous)

    # Keep points from B that are new features
    B_points_to_keep = B_points[mask_new_features]
    B_features_to_keep = B_features[mask_new_features]

    # Combine A's points with the selected B points
    merged_points = np.vstack((A_points, B_points_to_keep))
    merged_features = np.vstack((A_features, B_features_to_keep))

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

def downsample_point_cloud_to_match(pcd, target_point_count):
    """
    Downsamples a point cloud to match the target point count using voxel grid filtering.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to downsample.
        target_point_count (int): The desired number of points after downsampling.

    Returns:
        o3d.geometry.PointCloud: The downsampled point cloud.
    """
    import numpy as np

    # Compute the bounding box of the point cloud
    bounding_box = pcd.get_axis_aligned_bounding_box()
    bbox_extent = bounding_box.get_extent()
    volume = np.prod(bbox_extent)

    # Initial guess for voxel size
    voxel_size = (volume / target_point_count) ** (1/3)

    # Adjust voxel size iteratively
    scales = np.linspace(0.5, 2.0, num=20)
    for scale in scales:
        current_voxel_size = voxel_size * scale
        pcd_downsampled = pcd.voxel_down_sample(current_voxel_size)
        current_point_count = len(pcd_downsampled.points)
        print(f"Trying voxel_size={current_voxel_size:.4f}, resulting point count={current_point_count}")
        if abs(current_point_count - target_point_count) / target_point_count < 0.05:
            print("Desired point count achieved.")
            return pcd_downsampled

    # Return the downsampled point cloud with the closest point count
    return pcd.voxel_down_sample(voxel_size)



def align_point_clouds_with_scaling(pred_pcd, gt_pcd, threshold=1.0, max_iterations=50):
    """
    Aligns two point clouds using ICP with scaling estimation.

    Args:
        pred_pcd (o3d.geometry.PointCloud): Predicted point cloud.
        gt_pcd (o3d.geometry.PointCloud): Ground truth point cloud.
        threshold (float): Distance threshold for ICP.
        max_iterations (int): Maximum number of ICP iterations.

    Returns:
        np.ndarray: Transformation matrix that aligns the predicted point cloud to the ground truth.
        float: Root Mean Square Error (RMSE) of the alignment.
    """
    import open3d as o3d
    import numpy as np

    # Initial alignment
    initial_transformation = np.identity(4)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pred_pcd, gt_pcd, threshold, initial_transformation, estimation, criteria
    )

    print("Transformation matrix:")
    print(reg_p2p.transformation)
    print("Alignment RMSE:", reg_p2p.inlier_rmse)

    return reg_p2p.transformation, reg_p2p.inlier_rmse

# Visualize the alignment with textures
def visualize_aligned_pointclouds(gt_pcd, pred_pcd):
    """
    Visualizes the ground truth and predicted point clouds with textures.

    Args:
        gt_pcd (o3d.geometry.PointCloud): Ground truth point cloud.
        pred_pcd (o3d.geometry.PointCloud): Predicted point cloud after alignment.
    """
    # Optionally make the predicted point cloud semi-transparent
    pred_pcd_temp = o3d.geometry.PointCloud(pred_pcd)
    pred_pcd_temp.colors = pred_pcd.colors
    pred_pcd_temp.points = pred_pcd.points

    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Ground Truth and Predicted Point Clouds with Textures')

    # Add point clouds
    vis.add_geometry(gt_pcd)
    vis.add_geometry(pred_pcd_temp)

    # Optionally adjust point size and render options
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  # Increase point size for better visibility

    # Run the visualizer
    vis.run()
    vis.destroy_window()



def recon_o3d(config):
    mute = config['IF_MUTE']
    in_path = config['DEPTH_FOLDER']
    data_path = config['DATA_PATH']
    print(in_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() and config['USE_GPU'] else "cpu")
    if not mute:
        print(f'Device chosen: {device}')
    global_map = Pointclouds(points=[])
    global_map = global_map.to(device)
    
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
    for filename in file_list:
        file_paths.append(os.path.join(in_path, filename))
    previous_pointcloud = None
    T_global = np.eye(4)
    K = None  # Initialize K
    depth_range = None
    for file_path in file_paths:
        # Load the point cloud and get width and height
        current_pointcloud, width, height, current_depth_range = register_depth_map_to_pointcloud_new(
            file_path, device, need_reverse=config['NEED_REVERSE'],
            focal_length=config['fx'], texture_folder=config['DATA_PATH'],
            universal_mask=universal_mask
        )
        
        # Compute camera intrinsic matrix K if not already computed
        if K is None:
            fx = fy = config['fx']
            cx = width / 2.0
            cy = height / 2.0
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])

        if previous_pointcloud is None:
            # First frame
            global_map = current_pointcloud
            previous_pointcloud = current_pointcloud
            T_global = np.eye(4)
        else:
            # Convert to Open3D PointClouds
            pcd_prev = pointclouds_to_open3d(previous_pointcloud)
            pcd_curr = pointclouds_to_open3d(current_pointcloud)

            # Align current_pointcloud to previous_pointcloud
            aligned_pcd_curr, T_rel = align_point_clouds(
                pcd_prev, pcd_curr, width, height, K=K, max_iterations=10,
                output_video='alignment_error_video.avi'
            )

            # Convert aligned Open3D point cloud back to PyTorch3D Pointclouds
            aligned_pointcloud = open3d_to_pointclouds(aligned_pcd_curr, device)

            # Update global transformation
            T_global = T_global @ T_rel

            # Apply global transformation to current_pointcloud (if necessary)
            transformed_pointcloud = apply_transformation(current_pointcloud, T_global)

            # Merge transformed_pointcloud into global_map
            global_map = merge_point_clouds_kdtree(global_map, transformed_pointcloud)

            # Update previous_pointcloud
            previous_pointcloud = current_pointcloud

            # Optional visualization and saving
            if counter % 30 == 0:
                print(f'Counter: {counter}')
                output_ply(global_map)
                visualize_pointcloud_with_texture(global_map)
            counter += 1

    # Final visualization and output
    visualize_pointcloud_with_texture(global_map)
    return output_ply(global_map)
# Example usage
def recon(config, universal_mask = None, depth_range = None):# TODO: the depth range is not updating. in the middle of coding here.
    mute = config['IF_MUTE']
    in_path = config['DEPTH_FOLDER']
    data_path = config['DATA_PATH']
    print(in_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() and config['USE_GPU'] else "cpu")
    if not mute:
        print(f'Device chosen: {device}')
    global_map = Pointclouds(points=[])
    global_map = global_map.to(device)
    
    data_paths = []
    data_path_list = sorted([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))])
    for filename in data_path_list:
        data_paths.append(os.path.join(data_path, filename))
    if universal_mask is None:
        universal_mask = generate_universal_mask(data_paths, threshold=10)
    # universal_mask[:] = 1 #Toggle to disable masking
    # Visualize the universal mask
    # plt.imshow(universal_mask, cmap='gray')
    # plt.title('Universal Mask')
    # plt.show()
    file_list = sorted([f for f in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, f))])
    file_paths = []
    counter = 0
    for filename in file_list:
        file_paths.append(os.path.join(in_path, filename))
    previous_pointcloud = None
    T_global = np.eye(4)
    depth_scale = config['SCALE_FOR_DEPTH']
    for file_path in file_paths:
        # Load the point cloud
        current_pointcloud, width, height, current_depth_range = register_depth_map_to_pointcloud_new(
        file_path, device, need_reverse=config['NEED_REVERSE'],
        focal_length=config['fx'], texture_folder=config['DATA_PATH'],
        universal_mask=universal_mask, depth_scale=depth_scale
        )

        # Update depth_range
        fx = fy = config['fx']
        cx = width / 2.0
        cy = height / 2.0
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])
        if previous_pointcloud is None:
            # First frame
            global_map = current_pointcloud
            previous_pointcloud = current_pointcloud
            T_global = np.eye(4)
        else:
            # Align current_pointcloud to previous_pointcloud
            #aligned_pointcloud, T_rel = icp(previous_pointcloud, current_pointcloud)
            print(previous_pointcloud)
            #aligned_pointcloud, T_rel = icp_with_overlap_filtering(previous_pointcloud, current_pointcloud)
            aligned_pointcloud, T_rel, mean_error,std_error = icp_with_overlap_filtering(
                previous_pointcloud, current_pointcloud, max_iterations=15,initial_distance_threshold=10, 
                final_distance_threshold=0.1, K=K, width=width, height=height,
                output_video='alignment_error_video.avi', visualize=False
            )
            # Update global transformation
            T_global = T_global @ T_rel
            # Apply global transformation to current_pointcloud
            transformed_pointcloud = apply_transformation(current_pointcloud, T_global)
            # Merge transformed_pointcloud into global_map
            if mean_error < config['ERROR_TOLERANCE']:
                global_map = merge_point_clouds_kdtree(global_map, transformed_pointcloud,threshold_far=mean_error,threshold_erroneous=mean_error+2*std_error)
            # Update previous_pointcloud
            previous_pointcloud = current_pointcloud
            if counter % 30 == 0:
                print(f'Counter: {counter}')
                output_ply(global_map)
                visualize_pointcloud_with_texture(global_map)
            counter += 1
    visualize_pointcloud_with_texture(global_map)
    output_ply(global_map)
    return global_map, universal_mask

# if __name__ == "__main__":
#     # recon({
#     #     'IF_MUTE': False,
#     #     'DATA_PATH': 'data/Hamlyn/rectified01/image01',
#     #     'DEPTH_FOLDER': '/home/yicheng/Github/pipeline/data/Hamlyn/rectified01/depth01',
#     #     'NEED_REVERSE': False,
#     #     'USE_GPU': True,
#     #     'fx': 383.1901395
#     # })
#     #recon({'IF_MUTE': False, 'DATA_PATH' : 'data/Hamlyn/test22/color', 'DEPTH_FOLDER': '/home/yicheng/Github/pipeline/data/Hamlyn/test22/depth','NEED_REVERSE':False, 'USE_GPU': True, 'fx':417.903625})
#     #recon({'IF_MUTE': False, 'DATA_PATH' : 'data/Hamlyn/rectified01/image01', 'DEPTH_FOLDER': '/home/yicheng/Github/pipeline/output/depth-anything:v2_20241108120532_output','NEED_REVERSE':True, 'USE_GPU': True, 'fx':383.1901395})
#     recon({'IF_MUTE': False, 'DATA_PATH' : 'data/Hamlyn/test22/color', 'DEPTH_FOLDER': '/home/yicheng/Github/pipeline/data/Hamlyn/test22/depth','NEED_REVERSE':False, 'USE_GPU': True, 'fx':525, 'ERROR_TOLERANCE': 5})
    
# if __name__ == "__main__":
#     # Configurations for predictions and ground truths
#     config_pred = {
#         'IF_MUTE': False,
#         'DATA_PATH': 'data/Hamlyn/rectified22/color',  # Adjust as necessary
#         'DEPTH_FOLDER': 'data/Hamlyn/rectified22/depth_predict',  # Prediction folder
#         'NEED_REVERSE': True,  # Predictions require reversing
#         'USE_GPU': True,
#         'fx': 417.9036255,
#         'ERROR_TOLERANCE': 5,
#         'SCALE_FOR_DEPTH': 1
#     }

#     config_gt = {
#         'IF_MUTE': False,
#         'DATA_PATH': 'data/Hamlyn/rectified22/color',  # Adjust as necessary
#         'DEPTH_FOLDER': 'data/Hamlyn/rectified22/depth',  # Ground truth folder
#         'NEED_REVERSE': False,  # Ground truths do not require reversing
#         'USE_GPU': True,
#         'fx': 417.9036255,
#         'ERROR_TOLERANCE': 5,
#         'SCALE_FOR_DEPTH': 1
#     }


#     # Reconstruct ground truth depth maps
#     gt_pointcloud, uni_mask= recon(config_gt)
#     # Reconstruct predicted depth maps
#     pred_pointcloud, _ = recon(config_pred, universal_mask= uni_mask)

   

#     # Now align the predicted point cloud to the ground truth point cloud
#     # Convert pointclouds to Open3D point clouds
#     pred_pcd = pointclouds_to_open3d(pred_pointcloud)
#     gt_pcd = pointclouds_to_open3d(gt_pointcloud)

#     # Downsample the ground truth point cloud to match the predicted point count
#     pred_point_count = len(pred_pcd.points)
#     gt_pcd_downsampled = downsample_point_cloud_to_match(gt_pcd, pred_point_count)

#     # Align the point clouds
#     transformation, rmse = align_point_clouds_with_scaling(
#         pred_pcd, gt_pcd_downsampled, threshold=255.0, max_iterations=50
#     )

#     # Apply the transformation to the predicted point cloud
#     pred_pcd.transform(transformation)

#     # Visualize the alignment with textures
#     visualize_aligned_pointclouds(gt_pcd, pred_pcd)

if __name__ == "__main__":
    # Testing the alignment methods with synthetic data using existing functions

    import numpy as np
    import open3d as o3d

    # Generate random point cloud A
    N = 1000
    A_points = np.random.uniform(-1, 1, (N, 3))

    # Create scaling factor and noise
    scale_factor = 1.5  # True scaling factor
    noise_sigma = 0.01  # Noise standard deviation

    # Generate point cloud B = A * scale + noise
    noise = np.random.normal(0, noise_sigma, (N, 3))
    B_points = A_points * scale_factor + noise

    # Create Open3D point clouds
    A_pcd = o3d.geometry.PointCloud()
    A_pcd.points = o3d.utility.Vector3dVector(A_points)
    A_pcd.paint_uniform_color([1, 0, 0])  # Red for ground truth

    B_pcd = o3d.geometry.PointCloud()
    B_pcd.points = o3d.utility.Vector3dVector(B_points)
    B_pcd.paint_uniform_color([0, 1, 0])  # Green for prediction

    # Use the existing align_point_clouds_with_scaling function
    # This function should already be defined in your code
    transformation, rmse = align_point_clouds_with_scaling(
        B_pcd, A_pcd, threshold=999, max_iterations=50
    )

    print("Recovered transformation:")
    print(transformation)
    print("RMSE:", rmse)

    # Extract scaling factor from the transformation matrix
    R = transformation[:3, :3]
    estimated_scale_inverse = np.cbrt(np.linalg.det(R))  # Since det(R) = (1/s)^3
    estimated_scale = 1 / estimated_scale_inverse  # Recover the true scaling factor
    print(f"Estimated scaling factor: {estimated_scale}")
    print(f"True scaling factor: {scale_factor}")

    # Normalize the rotation matrix to remove scaling
    R_normalized = R / estimated_scale_inverse
    transformation_normalized = np.identity(4)
    transformation_normalized[:3, :3] = R_normalized
    transformation_normalized[:3, 3] = transformation[:3, 3] / estimated_scale_inverse

    # Apply the transformation to B_pcd
    B_pcd.transform(transformation)

    # Visualize the alignment using your existing function
    # The function visualize_aligned_pointclouds should already be defined
    # visualize_aligned_pointclouds(A_pcd, B_pcd)
