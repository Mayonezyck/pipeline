import open3d as o3d
import os

def display_point_cloud(file_path):
    # Read the point cloud from the .ply file
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Display the point cloud
    o3d.visualization.draw_geometries([pcd])

def display_all_point_clouds_in_folder(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".ply"):
                file_path = os.path.join(folder_path, file_name)
                display_point_cloud(file_path)

if __name__ == "__main__":
    file_path = "/home/yicheng/Github/pipeline/output/ICP_20241108114705.ply"
    
   
    display_point_cloud(file_path) 
    # folder_path = "/home/yicheng/Github/pipeline/see_ply"
    # display_all_point_clouds_in_folder(folder_path)