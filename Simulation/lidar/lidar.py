import numpy as np

class Lidar:
    def __init__(self, sensor_cfg):
        self.sensor_cfg = sensor_cfg
        self.elevation_angles = np.radians(sensor_cfg['lidar_elevation_angles']) if 'lidar_elevation_angles' in sensor_cfg else None
        # self.elevation_angles = None
        self.min_az_angle=np.radians(self.sensor_cfg['min_az_angle'])
        self.max_az_angle=np.radians(self.sensor_cfg['max_az_angle'])
        self.min_elev_angle=np.radians(self.sensor_cfg['min_elev_angle'])
        self.max_elev_angle=np.radians(self.sensor_cfg['max_elev_angle'])
        self.vertical_channels=self.sensor_cfg['v_channels']
        self.horizontal_channels=self.sensor_cfg['h_channels']
        self.max_range=self.sensor_cfg['max_range']
        self.beam_radius=self.sensor_cfg['beam_radius'] if 'beam_radius' in self.sensor_cfg else None
        self.wavelength=self.sensor_cfg['wavelength'] if 'wavelength' in self.sensor_cfg else None
        self.pulse_width=self.sensor_cfg['pulse_width'] if 'pulse_width' in self.sensor_cfg else None
        self.receiver_area=self.sensor_cfg['receiver_area'] if 'receiver_area' in self.sensor_cfg else None
        self.horizontal_fov = np.radians(self.sensor_cfg['horizontal_fov'])
        self.vertical_fov = np.radians(self.sensor_cfg['vertical_fov'])
        self.max_intensity = self.sensor_cfg['max_intensity']
        if self.elevation_angles is not None:
            self.azimuth_image, self.elevation_image = np.meshgrid(np.linspace(self.min_az_angle, self.max_az_angle, self.horizontal_channels), self.elevation_angles)
        else:
            self.azimuth_image, self.elevation_image = np.meshgrid(np.linspace(self.min_az_angle, self.max_az_angle, self.horizontal_channels),\
                                          np.linspace(self.min_elev_angle, self.max_elev_angle, self.vertical_channels))

    def spinning_lidar_model(self, azimuth, elevation, range_values):
        """
        Transform LiDAR azimuth-elevation-range frame to Cartesian coordinates.

        Parameters:
        - azimuth, elevation: Azimuth and elevation angles in radians.
        - range_values: Range values.
        - intensity: Intensity values.

        Returns:
        - x, y, z, intensity.
        """
        r_proj = range_values * np.cos(elevation)
        x = r_proj * np.cos(azimuth)
        y = r_proj * np.sin(azimuth)
        z = range_values * np.sin(elevation)
        return x, y, z
    
    def filter_out_zero_depth(self, pc_coords, intensity):
        """
        Filter out points with zero range values.

        Parameters:
        - pc_coords (np.ndarray): Nx3 array of point cloud coordinates (x, y, z).
        - intensity (np.ndarray): Intensity values.

        Returns:
        - pc_coords_filtered, intensity_filtered.
        """
        # Filter out points with zero range values
        non_zero_depth_mask = np.linalg.norm(pc_coords, axis=1) > 0.0
        pc_coords_filtered = pc_coords[non_zero_depth_mask]
        intensity_filtered = intensity[non_zero_depth_mask]
        return pc_coords_filtered, intensity_filtered, non_zero_depth_mask

    def spherical_trans_spinning_lidar(self, x, y, z, allowed_elevation_margin=np.pi):
        """
        Transform Cartesian coordinates to LiDAR azimuth-elevation-range frame.

        Parameters:
        - x, y, z: Cartesian coordinates.

        Returns:
        - azimuth, elevation, range_values, elevation_idx.
        """
        r_proj2 = x**2 + y**2
        range_values = np.sqrt(r_proj2 + z**2)
        azimuth = np.arctan2(y, x)
        elevation = np.arctan2(z, np.sqrt(r_proj2))
        if self.elevation_angles is not None:
                # Sort elevation angles and their indices
            elev_sorted = np.sort(self.elevation_angles)
            # Interpolate to find the nearest indices of sorted elevation angles for all points
            elevation_idx = np.interp(elevation, elev_sorted, np.arange(len(elev_sorted)))
            elevation_idx = np.round(elevation_idx).astype(int)  # Ensure integer indices
            # Validate elevation margin only for valid indices
            elevation_residual = elevation - self.elevation_angles[elevation_idx]
            # Mark points outside the allowed elevation margin as invalid
            too_far_idx = np.abs(elevation_residual) > allowed_elevation_margin
            elevation_idx[too_far_idx] = len(self.elevation_angles)   # Invalid indices
        else:
            elevation_idx = ((elevation - self.min_elev_angle) / (self.max_elev_angle - self.min_elev_angle) * self.vertical_channels).astype(int)
            elevation_idx = np.clip(elevation_idx, 0, self.vertical_channels - 1)

        return azimuth, elevation, range_values, elevation_idx

    def spinning_lidar_pcl_to_image(self, pc_coords, intensity):
        """
        Generate a depth image from a point cloud based on azimuth and elevation angles.

        Parameters:
        - pc_coords (np.ndarray): Nx3 array of point cloud coordinates (x, y, z).
        - min_az_angle (float): Minimum azimuth angle (radians).
        - max_az_angle (float): Maximum azimuth angle (radians).
        - min_elev_angle (float): Minimum elevation angle (radians).
        - max_elev_angle (float): Maximum elevation angle (radians).
        - vertical_channels (int): Number of vertical pixels (elevation resolution).
        - horizontal_channels (int): Number of horizontal pixels (azimuth resolution).

        Returns:
        - depth_image (np.ndarray): A 2D array (vertical_channels x horizontal_channels) representing range values.
        """
        # we set allow elevation margin to pi to allow all points to be considered for the moment
        azimuth, elevation, range_values, elevation_idx = self.spherical_trans_spinning_lidar(pc_coords[:, 0], pc_coords[:, 1], pc_coords[:, 2], allowed_elevation_margin=np.pi) 
        # elevation idx is the ring index
        # Normalize and map azimuth angles to horizontal indices
        azimuth_idx = ((azimuth - self.min_az_angle) / (self.max_az_angle - self.min_az_angle) * self.horizontal_channels).astype(int)
        # Clip indices to valid range
        azimuth_idx = np.clip(azimuth_idx, 0, self.horizontal_channels - 1)

        # Initialize depth image with infinity to indicate no data
        xyz_image = np.zeros((self.vertical_channels, self.horizontal_channels, 3))
        depth_image = np.zeros((self.vertical_channels, self.horizontal_channels))
        elevation_image = np.zeros((self.vertical_channels, self.horizontal_channels))
        azimuth_image = np.zeros((self.vertical_channels, self.horizontal_channels))
        intensity_image = np.zeros_like(depth_image)
        xyz_image[elevation_idx, azimuth_idx] = pc_coords
        intensity_image[elevation_idx, azimuth_idx] = intensity
        depth_image[elevation_idx, azimuth_idx] = range_values
        elevation_image[elevation_idx, azimuth_idx] = elevation
        azimuth_image[elevation_idx, azimuth_idx] = azimuth

        return xyz_image, intensity_image, depth_image, elevation_image, azimuth_image 