import numpy as np
import matplotlib.pyplot as plt


class lidar_calib:
    def __init__(self, noise_params, lidar):
        """
        Initialize the calibration error simulation.

        Parameters:
        - elevation_angles_degrees: List or array of default elevation angles in degrees.
        - elevation_margin_degrees: Allowed elevation margin in degrees (default: 0.1).
        """
        self.lidar = lidar
        v_channels = lidar.vertical_channels
        # self.elevation_angles = np.radians(lidar.elevation_angles) if lidar.elevation_angles is not None else None
        azimuth_std = noise_params['azimuth_std']
        elevation_std = noise_params['elevation_std']
        self.allowed_elevation_margin = np.radians(noise_params['elevation_margin_degrees'])  # Convert to radians
        self.azimuth_offsets = np.random.randn(v_channels) * np.radians(azimuth_std)  # 0.001 degree std
        self.elevation_offsets = np.random.randn(v_channels) * np.radians(elevation_std)  # 0.127 degree std
        self.intensity_offsets = np.random.randn(v_channels) * 5  # Intensity offsets
        self.order = 1

    def apply_noise(self, azimuth, elevation, range_values, intensity, ring_index):
        """
        Apply calibration noise to azimuth, elevation, and intensity.

        Parameters:
        - azimuth, elevation: Azimuth and elevation angles in radians.
        - range_values: Range values.
        - intensity: Intensity values.
        - ring_index: Ring indices.

        Returns:
        - azimuth_with_noise, elevation_with_noise, range_with_noise, intensity_with_noise.
        """
        # Extend offsets for NaN values
        azimuth_offsets = np.append(self.azimuth_offsets, np.nan)
        elevation_offsets = np.append(self.elevation_offsets, np.nan)
        intensity_offsets = np.append(self.intensity_offsets, np.nan)

        # Apply noise
        azimuth_with_noise = azimuth + azimuth_offsets[ring_index.astype(int)]
        elevation_with_noise = elevation + elevation_offsets[ring_index.astype(int)]
        range_with_noise = range_values  # No change to range
        intensity_with_noise = intensity + intensity_offsets[ring_index.astype(int)]
        # Clip intensity to valid range [0, 255]
        intensity_with_noise = np.clip(intensity_with_noise, 0, 255)
        return azimuth_with_noise, elevation_with_noise, range_with_noise, intensity_with_noise

    def simulate(self, pcl, intensity):
        """
        Simulate LiDAR calibration noise and transform back to Cartesian frame.

        Parameters:
        - x_input, y_input, z_input: Cartesian coordinates.
        - i_input: Intensity values.

        Returns:
        - x, y, z, intensity: Cartesian coordinates and intensity with noise applied.
        """
        # Transform to azimuth-elevation frame
        x_input, y_input, z_input = pcl[:, 0], pcl[:, 1], pcl[:, 2]
        azimuth, elevation, range_values, ring_index = self.lidar.spherical_trans_spinning_lidar(
            x_input, y_input, z_input, self.allowed_elevation_margin)

        # Apply noise
        azimuth_with_noise, elevation_with_noise, range_with_noise, intensity_sim = self.apply_noise(
            azimuth, elevation, range_values, intensity, ring_index
        )

        # Transform back to Cartesian frame
        x, y, z = self.lidar.spinning_lidar_model(
            azimuth_with_noise, elevation_with_noise, range_with_noise
        )
        pcl_sim = np.stack((x, y, z), axis=1)
        # remove nan points from the point cloud and intensity if any row has nan values
        nan_idx = np.isnan(pcl_sim).any(axis=1) & np.isnan(intensity)
        pcl_sim = pcl_sim[~nan_idx]
        intensity_sim = intensity_sim[~nan_idx]
        return pcl_sim, intensity_sim
