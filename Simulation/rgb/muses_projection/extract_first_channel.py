import os
import numpy as np

def extract_first_channel(src_dir, dest_dir):
    """
    Extract the first channel of 3-channel .npy files from src_dir and save them in dest_dir,
    preserving the subfolder structure.
    
    Parameters:
        src_dir (str): Path to the source directory containing .npy files.
        dest_dir (str): Path to the destination directory to save the extracted files.
    """
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.npy'):
                # Construct full source and destination file paths
                src_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, src_dir)
                dest_subfolder = os.path.join(dest_dir, relative_path)
                dest_path = os.path.join(dest_subfolder, file)
                
                # Create the destination subfolder if it does not exist
                os.makedirs(dest_subfolder, exist_ok=True)
                
                # Load the .npy file and extract the first channel
                data = np.load(src_path)
                if data.ndim == 3 and data.shape[-1] >= 1:  # Ensure it's 3-channel data
                    first_channel = data[..., 0]
                    # Save the first channel to the destination file
                    np.save(dest_path, first_channel)
                    print(f"Saved first channel to: {dest_path}")
                else:
                    print(f"Skipping file (not valid 3-channel data): {src_path}")

# Example usage
src_folder = "/path/to/aurora-root/Code/MUSES/Dataset/lidar_kernel5_npy"
dest_folder = "/path/to/aurora-root/Code/MUSES/Dataset/lidar_kernel_5_distance_maps"
extract_first_channel(src_folder, dest_folder)
