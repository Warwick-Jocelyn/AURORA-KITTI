
import numpy as np

class weather_model_base:
    def __init__(self, lidar):
        self.beam_radius = lidar.beam_radius  # m
        self.lidar_max_range = lidar.max_range
        self.wavelength = lidar.wavelength
        self.pulse_width = lidar.pulse_width
        self.receiver_area = lidar.receiver_area
        self.v_channels = lidar.vertical_channels
        self.h_channels = lidar.horizontal_channels
        self.min_az_angle = lidar.min_az_angle
        self.max_az_angle = lidar.max_az_angle
        self.min_z_angle = lidar.min_elev_angle
        self.max_z_angle = lidar.max_elev_angle
        self.horizontal_res = lidar.horizontal_fov / self.h_channels
        self.vertical_res = lidar.vertical_fov / self.v_channels

    def pre_process(self, pcl, intensity):
        xyz_image, intensity_image, depth_image, elevation_image, azimuth_image = self.lidar.spinning_lidar_pcl_to_image(pcl, intensity)
        zero_depth_mask = np.logical_and.reduce([
        depth_image == 0,
        # self.lidar.azimuth_image >= -np.pi/4,
        # self.lidar.azimuth_image <= np.pi/4,
        self.lidar.elevation_image > -np.pi/18,
        ])
        min_drop_range = 10
        max_drop_range = self.lidar_max_range/2 
        # max_drop_range = self.lidar_max_range/2 if not snow else 15
        # depth_image[zero_depth_mask] = np.random.uniform(min_drop_range, max_drop_range, size=np.sum(zero_depth_mask))
        depth_image[zero_depth_mask] = 80
        x, y, z = self.lidar.spinning_lidar_model(self.lidar.azimuth_image, self.lidar.elevation_image, depth_image)
        xyz_image_p = np.stack([x, y, z], axis=-1)
        xyz_image[zero_depth_mask] = xyz_image_p[zero_depth_mask]
        elevation_image[zero_depth_mask] = self.lidar.elevation_image[zero_depth_mask]
        azimuth_image[zero_depth_mask] = self.lidar.azimuth_image[zero_depth_mask]
        return xyz_image.reshape(-1, 3), intensity_image.reshape(-1), depth_image.reshape(-1), elevation_image.reshape(-1), azimuth_image.reshape(-1), zero_depth_mask.reshape(-1)

    def simulate(self, pcl, intensity):
        pass