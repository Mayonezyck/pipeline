import torch, os
from pytorch3d.structures import Pointclouds
import matplotlib.pyplot as plt
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial import cKDTree
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
    valid = depth > 0

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

    # Convert back to tensors and create aligned point cloud with features
    B_points_tensor = torch.from_numpy(B_points).float().to(B.device)
    B_colors_tensor = torch.from_numpy(B_colors).float().to(B.device)
    aligned_B = Pointclouds(points=[B_points_tensor], features=[B_colors_tensor])

    return aligned_B

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
    # Visualize the universal mask
    plt.imshow(universal_mask, cmap='gray')
    plt.title('Universal Mask')
    plt.show()
    file_list = sorted([f for f in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, f))])
    file_paths = []
    for filename in file_list:
        file_paths.append(os.path.join(in_path, filename))
    for file_path in file_paths:
        if firstTime:
            firstTime = False
            worldmap = register_depth_map_to_pointcloud_new(file_path, device, need_reverse=config['NEED_REVERSE'], focal_length=config['fx'],texture_folder=config['DATA_PATH'],universal_mask=universal_mask)
        else:
            new_pointcloud = register_depth_map_to_pointcloud_new(file_path, device, need_reverse=config['NEED_REVERSE'], focal_length=config['fx'],texture_folder=config['DATA_PATH'],universal_mask=universal_mask)   
            worldmap = icp(worldmap, new_pointcloud)
        visualize_pointcloud_with_texture(worldmap)
    # visualize_pointcloud_open3d(worldmap)
    # rp = generate_random_pointcloud(1000, device)
    # visualize_pointcloud_open3d(rp)


#recon({'IF_MUTE': False, 'DATA_PATH' : 'data/Hamlyn/rectified17/image01_last45', 'DEPTH_FOLDER': 'temp','NEED_REVERSE':True, 'USE_GPU': True, 'fx':383.1901395})