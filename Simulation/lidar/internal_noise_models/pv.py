import numpy as np
import matplotlib.pyplot as plt


class lidar_pv:
    def __init__(self, noise_params, lidar):
        """
        Initialize the calibration error simulation.

        Parameters:
        - elevation_angles_degrees: List or array of default elevation angles in degrees.
        - elevation_margin_degrees: Allowed elevation margin in degrees (default: 0.1).
        """
        self.lidar = lidar
        v_channels = lidar.vertical_channels
        power_std = noise_params['power_std']
        # self.elevation_angles = np.radians(lidar.elevation_angles) if lidar.elevation_angles is not None else None
        self.intensity_offsets = np.random.randn(v_channels) * power_std  # Intensity offsets
        self.order = 1

    def apply_noise(self, intensity, ring_index):
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
        intensity_offsets = np.append(self.intensity_offsets, np.nan)

        # Apply noise
        min_intensity = np.min(intensity)
        max_intensity = np.max(intensity)
        intensity_with_noise = intensity + intensity_offsets[ring_index.astype(int)]
        # Clip intensity to valid range [0, 255]
        intensity_with_noise = np.clip(intensity_with_noise, min_intensity, max_intensity) # TODO: look at this later
        return intensity_with_noise

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
        _, _, _, ring_index = self.lidar.spherical_trans_spinning_lidar(
            x_input, y_input, z_input, np.pi)

        # Apply noise
        intensity_sim = self.apply_noise(intensity, ring_index)

        # Transform back to Cartesian frame
        # remove nan points from the point cloud and intensity if any row has nan values
        nan_idx = np.isnan(intensity)
        pcl_sim = pcl[~nan_idx]
        intensity_sim = intensity_sim[~nan_idx]
        return pcl_sim, intensity_sim
