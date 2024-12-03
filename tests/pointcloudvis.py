import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import open3d as o3d
def downsample_point_cloud(pcd, voxel_size):
    return pcd.voxel_down_sample(voxel_size=voxel_size)

# Load two point clouds from .ply files
# Replace 'point_cloud1.ply' and 'point_cloud2.ply' with your actual file paths
pcd1 = o3d.io.read_point_cloud('output/ICP_20241126114421.ply')
pcd2 = o3d.io.read_point_cloud('output/ICP_20241126114412.ply')

voxel_size = 10  # Adjust voxel size for the desired level of downsampling
pcd1_down = downsample_point_cloud(pcd1, voxel_size)
pcd2_down = downsample_point_cloud(pcd2, voxel_size)

# Convert point clouds to numpy arrays
point_cloud1 = np.asarray(pcd1_down.points)
point_cloud2 = np.asarray(pcd2_down.points)

# Initial scaling factor
initial_scale = 1.0

# Create the figure and a 3D subplot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the initial point clouds
scatter1 = ax.scatter(point_cloud1[:, 0], point_cloud1[:, 1], point_cloud1[:, 2],
                      c='blue', label='Point Cloud 1', s=1)
scatter2 = ax.scatter(point_cloud2[:, 0] * initial_scale,
                      point_cloud2[:, 1] * initial_scale,
                      point_cloud2[:, 2] * initial_scale,
                      c='red', label='Point Cloud 2', s=1)

# Set labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Adjust layout to make room for the slider
plt.subplots_adjust(bottom=0.25)

# Create a slider axis and slider
ax_scale = plt.axes([0.25, 0.1, 0.65, 0.03])
scale_slider = Slider(ax_scale, 'Scaling Factor', 0.1, 3.0, valinit=initial_scale)

# Update function to be called when the slider's value changes
def update(scale):
    global scatter2  # Declare scatter2 as global to modify it inside the function
    # Remove the previous scatter plot
    scatter2.remove()
    # Plot the second point cloud with the new scaling factor
    scaled_points = point_cloud2 * scale
    scatter2 = ax.scatter(scaled_points[:, 0],
                          scaled_points[:, 1],
                          scaled_points[:, 2],
                          c='red', label='Point Cloud 2', s=1)
    # Redraw the figure to update the plot
    fig.canvas.draw_idle()

# Connect the slider to the update function
scale_slider.on_changed(update)

plt.show()
