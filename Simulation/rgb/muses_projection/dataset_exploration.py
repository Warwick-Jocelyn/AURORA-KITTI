import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Input directory
    input_npy_dir = os.path.expanduser('/path/to/aurora-root/Code/MUSES/Dataset/lidar_kernel_1_distance_maps')
    print("###############")
    print(f"Input Directory: {input_npy_dir}")

    # Find all .npy files
    npy_files_to_process = sorted(glob.glob(os.path.join(input_npy_dir, '**', '*.npy'), recursive=True))
    print(f"Number of .npy files found: {len(npy_files_to_process)}")

    if not npy_files_to_process:
        print("No .npy files found in the specified directory.")
        return

    # Output directory
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)

    # File to save all depths
    npy_file_name = os.path.join(output_dir, 'all_depths.npy')
    all_depths_saved = os.path.exists(npy_file_name)

    if not all_depths_saved:
        # Use a list for better performance during accumulation
        all_depths_list = []

        # Process each .npy file
        for i, npy_file_path in enumerate(npy_files_to_process):
            sys.stdout.write(f'\rProcessing {i + 1} / {len(npy_files_to_process)}')
            sys.stdout.flush()

            try:
                # Load and flatten the data
                depth_data = np.load(npy_file_path)
            except Exception as e:
                print(f"\nWarning: Failed to load {npy_file_path}: {e}")
                continue

            if depth_data.ndim != 2:
                print(f"\nWarning: {npy_file_path} is not a 2D array, skipping.")
                continue

            # Append non-zero pixels (convert to float16 to save memory)
            valid_pixels = depth_data.flatten().astype(np.float16)
            all_depths_list.append(valid_pixels)

        # Concatenate all depths into one array
        all_depths = np.concatenate(all_depths_list)
        np.save(npy_file_name, all_depths)
        print(f"\nAll depths saved to {npy_file_name}")

    else:
        # Load pre-saved depths
        all_depths = np.load(npy_file_name)
        print(f"Loaded pre-saved depths from {npy_file_name}")

    # Compute statistics
    print('\nDepth Statistics (including zeros):')
    print(f"  Min:     {np.amin(all_depths):.2f}")
    print(f"  Max:     {np.amax(all_depths):.2f}")
    print(f"  Mean:    {np.mean(all_depths):.2f}")
    print(f"  Median:  {np.median(all_depths):.2f}")

    # Filter out zeros
    non_zero_depths = all_depths[all_depths > 0]
    print('\nDepth Statistics (excluding zeros):')
    print(f"  Min:     {np.amin(non_zero_depths):.2f}")
    print(f"  Max:     {np.amax(non_zero_depths):.2f}")
    print(f"  Mean:    {np.mean(non_zero_depths):.2f}")
    print(f"  Median:  {np.median(non_zero_depths):.2f}")
    print(f"  Non-zero Pixel Percentage: {100.0 * len(non_zero_depths) / len(all_depths):.2f}%")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(non_zero_depths, bins=50, color='blue', alpha=0.7)  # Fewer bins for faster plotting
    plt.title("Depth Data Distribution (Excluding Zeros)")
    plt.xlabel("Depth Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'depth_histogram.png'))
    plt.close()
    print(f"Histogram saved to {os.path.join(output_dir, 'depth_histogram.png')}")


if __name__ == '__main__':
    main()

# # Summary Statistics:
# Depth Statistics (including zeros):
#   Min:     0.00
#   Max:     200.00
#   Mean:    0.39
#   Median:  0.00

# Depth Statistics (excluding zeros):
#   Min:     1.47
#   Max:     200.00
#   Mean:    21.47
#   Median:  14.24
#   Non-zero Pixel Percentage: 1.79%
# Histogram saved to ./outputs/depth_histogram.png