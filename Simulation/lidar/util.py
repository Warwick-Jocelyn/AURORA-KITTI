import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from vispy import scene
from vispy.visuals.transforms import STTransform
from vispy.scene import visuals
from vispy.scene import Grid





# Load point cloud from .pcd file
def load_pcd_pointcloud_0(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Extract the header and find where the data starts
    data_start_index = 0
    for i, line in enumerate(lines):
        if line.strip() == "DATA ascii":
            data_start_index = i + 1
            break
    data = np.loadtxt(lines[data_start_index:])
    x, y, z, intensity = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    points = np.vstack((x, y, z)).T
    return points, intensity

def load_pcd_pointcloud(file_path, radar=False):
    from pypcd4 import PointCloud
    pc = PointCloud.from_path(file_path)
    points = pc.numpy(("x", "y", "z"))
    if radar:
        points[:, 0] = - points[:, 0]
    try:
        intensity = pc.numpy("I")
    except:
        intensity = np.zeros((points.shape[0], 1))
    return points, intensity[:, 0]


def load_bin_pointcloud(file_path):
    pc = np.fromfile(file_path, dtype=np.float32)
    label_path = file_path.replace('pcd', 'labels').replace(".bin", ".label")
    pc = pc.reshape(-1, 4)
    label = np.fromfile(label_path, dtype=np.int32) if os.path.exists(label_path) else None
    return pc[:, :3], pc[:, 3], label


def load_mat_pointcloud(file_path):
    import scipy.io
    data = scipy.io.loadmat(file_path)
    x_input = data['x_input'].flatten()
    y_input = data['y_input'].flatten()
    z_input = data['z_input'].flatten()
    i_input = data['i_input'].flatten()
    pcl = np.stack((x_input, y_input, z_input), axis=1)
    return pcl, i_input

def load_npy_pointcloud(file_path):
    pcl_intensity = np.load(file_path)
    return pcl_intensity[:, :3], pcl_intensity[:, 3] # split to point cloud and intensity

# Visualize point cloud
def plot_pointcloud(pc_coords, title="Point Cloud", 
                    xlim=(0, 30.5), ylim=(-4.5, 4.5), zlim=(-1.5, 1.5), 
                    xticks=[0, 5, 10, 15, 20, 25, 30], 
                    yticks=[-4, -2, 0, 2, 4], zticks=[-1, 0, 1]):
    """
    Plot a 3D point cloud with specified axis limits and ticks.
    
    Parameters:
    - pc_coords (np.ndarray): Nx3 array of point cloud coordinates.
    - title (str): Title of the plot.
    - xlim, ylim, zlim (tuple): Axis limits.
    - xticks, yticks, zticks (list): Tick values for each axis.
    """
    fig = plt.figure(figsize=(12, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the points
    ax.scatter(pc_coords[:, 0], pc_coords[:, 1], pc_coords[:, 2], s=1, c="blue")

    # Set axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # Set axis ticks
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)

    # Set labels and title
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)


def prepare_color_for_vispy(intensity, min_color, max_color, logarithmic):
    num_levels = 100  # Number of discrete levels for the colormap
    base = 1.001
    cmap = plt.get_cmap('viridis', num_levels)  # Use 'viridis' colormap with discrete levels
    if logarithmic:
        log_levels = np.linspace(np.log(min_color) / np.log(base), np.log(max_color) / np.log(base), num_levels)
        intensity_levels = np.clip(1.001 ** log_levels, min_color, max_color)
    else:
        intensity_levels = np.linspace(min_color, max_color, num_levels)
    closest_levels = np.abs(intensity[..., None] - intensity_levels[None, ...]).argmin(axis=-1)
    colors = cmap(closest_levels)[..., :3]  # Extract RGB values (ignore alpha)
    # Add alpha channel (opacity)
    colors = np.concatenate([colors, np.ones((*colors.shape[:-1], 1))], axis=-1)
    colors[intensity == 0] = [1.0, 1.0, 1.0, 1.0] 


    return colors

def plot_pointcloud_vispy(pc_coords, color_intensity, lidar, title="Point Cloud", position=(100, 100), min_color=0.0, max_color=255.0, logarithmic=True):
    """
    Plot a 3D point cloud using VisPy with specified axis limits and intensity-based coloring,
    and visualize the corresponding depth image in the same window.
    
    Parameters:
    - pc_coords (np.ndarray): Nx3 array of point cloud coordinates.
    - intensity (np.ndarray): N array of intensity values.
    - title (str): Title of the point cloud plot.
    - xlim, ylim, zlim (tuple): Axis limits.
    - position (tuple): Position of the canvas window on the screen.
    """
    colors = prepare_color_for_vispy(color_intensity, min_color, max_color, logarithmic)
    # Create a VisPy canvas
    canvas = scene.SceneCanvas(keys='interactive', show=True, title=title, bgcolor='white', position=position)
    grid = canvas.central_widget.add_grid()
    # Add the 3D point cloud view to the left panel
    view_3d = grid.add_view(row=0, col=0, border_color='black', bgcolor='white')
    # view_3d.scene.light.set_state(enabled=False)
    scatter = visuals.Markers()
    scatter.set_gl_state( depth_test=False) 
    scatter.set_data(pc_coords, edge_color=colors, size=1)  # Use intensity-based colors
    scatter.set_data(pc_coords, edge_width=0.0, face_color=colors, size=2)  # Use intensity-based colors
    view_3d.add(scatter)
    view_3d.camera = 'turntable'  # 3D interactive camera
    view_3d.camera.fov = 45
    view_3d.camera.distance = 40
    # view_3d.camera.center = (np.mean(pc_coords[:, 0]), 
    #                          np.mean(pc_coords[:, 1]), 
    #                          np.mean(pc_coords[:, 2]))
    view_3d.camera.center = (0, 0, 0)
    view_3d.camera.elevation = 23  # Angle above the xy-plane (in degrees)
    view_3d.camera.azimuth = -110   # Rotation around the z-axis (in degrees)
    view_3d.camera.roll = 0        # Rotation around the view axis (in degrees)
    view_3d.camera.distance = 25  # Reduce distance to zoom in
    # Add axis to the 3D view
    axis = visuals.XYZAxis(parent=view_3d.scene)
    # axis.transform = STTransform(scale=(xlim[1] - xlim[0], 
    #                                     ylim[1] - ylim[0], 
    #                                     zlim[1] - zlim[0]),
    #                              translate=(xlim[0], ylim[0], zlim[0]))
    _, intensity_image, depth_image, _, _ = lidar.spinning_lidar_pcl_to_image(pc_coords, color_intensity)
    # Add the depth image view to the right panel
    view_depth = grid.add_view(row=1, col=0, border_color='black', bgcolor='white')
    colors = prepare_color_for_vispy(intensity_image, min_color, max_color, logarithmic)
    image_visual = visuals.Image(colors, parent=view_depth.scene)
    offset = 0
    # offset = 350
    image_visual.transform = scene.transforms.STTransform(translate=(0, offset))
    image_shape = depth_image.shape
    rectangle_visual = visuals.Rectangle(
    center=(image_shape[1] / 2, image_shape[0] / 2),
    width=image_shape[1],
    height=image_shape[0],
    color=None,  # No fill color
    border_color='black',  # Border color
    border_width=2,  # Thickness of the border
    parent=view_depth.scene)
    rectangle_visual.transform = scene.transforms.STTransform(translate=(0, offset))
    view_depth.camera = 'panzoom'
    view_depth.camera.set_range()
    view_depth.camera.aspect = 1.0

    return canvas






# Load YAML configuration file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    


if __name__ == '__main__':
    import sys
    import os
    from lidar import Lidar
    lidar = Lidar(load_yaml(os.path.join("configs", "lidar_ouster_config.yaml")))
    file_path = "Example_PointClouds\\points_frame_1.npy" 
    pcl_intensity = np.load(file_path)
    pc_coords = pcl_intensity[:, :3]
    pc_intensity = pcl_intensity[:, 3]
    pc_coords, pc_intensity, mask = lidar.filter_out_zero_depth(pc_coords, pc_intensity)
    ring_index = pcl_intensity[:, 6][mask]
    azimuth, elevation, _ , _ = lidar.spherical_trans_spinning_lidar(pc_coords[:, 0], pc_coords[:, 1], pc_coords[:, 2])
    channel_elevation_mean = {}
    channel_elevation_std = {}
    for ind in range(int(ring_index.max()) + 1):
        mask = ring_index == ind
        channel_elevation_mean[ind] = np.mean(elevation[mask])
        channel_elevation_std[ind] = np.std(elevation[mask])
        print(f"Ring {ind}: Mean elevation = {channel_elevation_mean[ind]:.4f} rad, Std. deviation = {channel_elevation_std[ind]:.4f} rad")

    clear_pc, clear_intensity = load_npy_pointcloud(file_path)

