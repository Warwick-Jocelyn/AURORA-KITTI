import numpy as np
import open3d as o3d

# Load the point cloud data from the .bin file
def load_lidar_data(bin_file):
    # Each row in the .bin file has 6 values: (x, y, z, intensity, ring, timestamp)
    point_cloud = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 6)
    return point_cloud[:, :3]  # Only take the x, y, z coordinates for visualization

# Visualize the point cloud using open3d
def visualize_point_cloud(bin_file):
    points = load_lidar_data(bin_file)

    # Create an open3d point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    bin_file = '/path/to/aurora-root/Dataset/MUSE/muses/lidar/train/clear/day/REC0006_frame_043790_lidar.bin'
    visualize_point_cloud(bin_file)
