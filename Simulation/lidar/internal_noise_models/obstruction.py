import numpy as np

class lidar_obstruction:
    def __init__(self, obstruction_params, lidar):
        """
        Initialize the LiDAR obstruction simulation.

        Parameters:
        - obstruction_params (dict): Dictionary containing:
            - 'probability' (float): Probability of an obstruction occurring.
            - 'max_size' (int): Maximum size of the obstruction (square patch).
        - sensor_cfg (dict): Dictionary containing sensor configuration:
        """
        self.probability = obstruction_params['probability']
        self.max_size = obstruction_params['max_size']
        self.lidar = lidar
        self.order = 2

    def apply_obstruction(self, xyz_image, intensity_image):
        """
        Apply random obstructions to a depth image.

        Parameters:
        - depth_image (np.ndarray): Input depth image (2D array).

        Returns:
        - obstructed_depth_image (np.ndarray): Depth image with obstructions applied.
        """
        obstructed_xyz_image = xyz_image.copy()
        obstructed_intensity_image = intensity_image.copy()
        h, w, _ = obstructed_xyz_image.shape

        # Step 1: Find all pixels with obstructions
        obstruction_mask = np.random.rand(h, w) < self.probability
        obstruction_pixels = np.argwhere(obstruction_mask)  # Get indices of obstructed pixels
        # Step 2: Loop over the obstructed pixels
        for pixel in obstruction_pixels:
            row, col = pixel
            # Sample random size for the square patch
            # patch_size = np.random.randint(2, self.max_size + 1)
            patch_size = self.max_size
            # Determine patch boundaries (ensuring they stay within image dimensions)
            row_start = max(row - patch_size // 2, 0)
            row_end = min(row + patch_size // 2 + 1, h)
            col_start = max(col - patch_size // 2, 0)
            col_end = min(col + patch_size // 2 + 1, w)

            # Apply obstruction by setting the patch to small number
            obstructed_xyz_image[row_start:row_end, col_start:col_end] = [0, 0, 0]
            obstructed_intensity_image[row_start:row_end, col_start:col_end] = 0

        return obstructed_xyz_image, obstructed_intensity_image

    def filter_nans(self, pcl, intensity):
        """
        Filter out points with NaN values from the point cloud.

        Parameters:
        - pcl (np.ndarray): Nx3 point cloud array (x, y, z).
        - intensity (np.ndarray): N array of intensity values.

        Returns:
        - pcl_filtered (np.ndarray): Filtered point cloud.
        - intensity_filtered (np.ndarray): Filtered intensity values.
        """
        nan_idx = np.isnan(pcl).any(axis=1) | np.isnan(intensity)
        pcl_filtered = pcl[~nan_idx]
        intensity_filtered = intensity[~nan_idx]
        return pcl_filtered, intensity_filtered

    def simulate(self, xyz, intensity):
        """
        Simulate LiDAR obstructions by applying random masks on the depth image.

        Parameters:
        - pcl (np.ndarray): Nx3 point cloud array (x, y, z).
        - intensity (np.ndarray): N array of intensity values.

        Returns:
        - pcl_filtered (np.ndarray): Point cloud after obstructions and filtering.
        - intensity_filtered (np.ndarray): Intensity values after obstructions and filtering.
        """
        # Generate depth image from the point cloud using sensor_cfg
        xyz_image, intensity_image, _, _, _ = self.lidar.spinning_lidar_pcl_to_image(xyz,intensity)
        obstructed_xyz_image, obstructed_intensity_image = self.apply_obstruction(xyz_image, intensity_image)
        xyz_sim = np.reshape(obstructed_xyz_image, (-1, 3))
        intensity_sim = obstructed_intensity_image.flatten()
        xyz_sim, intensity_sim, _ = self.lidar.filter_out_zero_depth(xyz_sim, intensity_sim)
        return xyz_sim, intensity_sim
