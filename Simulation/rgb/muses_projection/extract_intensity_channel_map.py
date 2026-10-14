import os
import numpy as np
import cv2

def extract_second_channel(src_dir, dest_dir):
    """
    Extract the second channel of 3-channel .npy files from src_dir and save them as 16-bit PNG images in dest_dir,
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
                os.makedirs(dest_subfolder, exist_ok=True)
                
                # Load the .npy file and extract the second channel
                data = np.load(src_path)
                if data.ndim == 3 and data.shape[-1] >= 2:  # Ensure it's 3-channel data with at least 2 channels
                    second_channel = data[..., 1]
                    
                    # Print statistics
                    min_val = np.min(second_channel)
                    max_val = np.max(second_channel)
                    mean_val = np.mean(second_channel)
                    std_val = np.std(second_channel)
                    print(f"Processing {file}: min={min_val}, max={max_val}, mean={mean_val:.4f}, std={std_val:.4f}")
                    
                    # Normalize to 16-bit range (0-65535)
                    second_channel = (second_channel - min_val) / (max_val - min_val) * 65535
                    second_channel = second_channel.astype(np.uint16)
                    
                    # Save as 16-bit PNG
                    png_filename = os.path.splitext(file)[0] + ".png"
                    dest_path = os.path.join(dest_subfolder, png_filename)
                    cv2.imwrite(dest_path, second_channel)
                    print(f"Saved second channel to: {dest_path}")
                else:
                    print(f"Skipping file (not valid 3-channel data): {src_path}")

# Example usage
src_folder = "/path/to/aurora-root/Code/MUSES/Dataset/lidar_kernel5_npy/val"
dest_folder = "/path/to/aurora-root/Code/MUSES/Dataset/intensity_kernel5_16bitmaps"
extract_second_channel(src_folder, dest_folder)
