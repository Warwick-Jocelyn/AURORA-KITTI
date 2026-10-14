import numpy as np

def compare_npy_files_with_details(file1, file2):
    """
    Compare two three-channel .npy files and print differences in sum, mean, max, and min for each channel.
    
    Parameters:
        file1 (str): Path to the first .npy file.
        file2 (str): Path to the second .npy file.
    """
    try:
        # Load the .npy files
        data1 = np.load(file1)
        data2 = np.load(file2)
        
        # Check if the shapes are the same
        if data1.shape != data2.shape:
            print("Different (shapes mismatch)")
            print(f"Shape of file1: {data1.shape}, Shape of file2: {data2.shape}")
            return
        
        # Check if the contents are the same
        if np.array_equal(data1, data2):
            print("Same")
            return
        
        # Compute and print detailed differences for each channel
        print("Different (content mismatch)\n")
        for channel in range(data1.shape[-1]):
            sum_diff = np.sum(data1[..., channel]) - np.sum(data2[..., channel])
            mean_diff = np.mean(data1[..., channel]) - np.mean(data2[..., channel])
            max_diff = np.max(data1[..., channel]) - np.max(data2[..., channel])
            min_diff = np.min(data1[..., channel]) - np.min(data2[..., channel])
            
            print(f"Channel {channel + 1}:")
            print(f"  Sum Difference: {sum_diff}")
            print(f"  Mean Difference: {mean_diff}")
            print(f"  Max Difference: {max_diff}")
            print(f"  Min Difference: {min_diff}")
            print()
    except Exception as e:
        print(f"Error comparing files: {e}")



# Example usage
file1_path = "/path/to/aurora-root/Code/MUSES/Dataset/lidar_kernel_1_npy/val/clear/day/REC0008_frame_079714_lidar.npy"
file2_path = "/path/to/aurora-root/Code/MUSES/Dataset/lidar_kernel5_npy/val/clear/day/REC0008_frame_079714_lidar.npy"
 
compare_npy_files_with_details(file1_path, file2_path)