import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np


def calculate_sparsity(npy_file_path):
    """
    Calculate the sparsity (percentage of non-zero pixels) for a single .npy file.
    :param npy_file_path: Path to the .npy file.
    :return: Sparsity as a percentage (0-100).
    """
    try:
        depth_data = np.load(npy_file_path)
    except Exception as e:
        print(f"\nError loading {npy_file_path}: {e}")
        return None

    if depth_data.ndim != 2:
        print(f"\nWarning: {npy_file_path} is not a 2D array, skipping.")
        return None

    num_pixels = depth_data.size
    num_non_zero_pixels = np.count_nonzero(depth_data)
    sparsity_percentage = (num_non_zero_pixels / num_pixels) * 100
    return sparsity_percentage


def main():
    # Input directory
    input_npy_dir = os.path.expanduser('/path/to/aurora-root/Code/MUSES/Dataset/lidar_kernel_1_distance_maps')
    print(f"Input directory: {input_npy_dir}")

    # Find all .npy files
    npy_files_to_process = sorted(glob.glob(os.path.join(input_npy_dir, '**', '*.npy'), recursive=True))
    print(f"Number of .npy files found: {len(npy_files_to_process)}")

    if not npy_files_to_process:
        print("No .npy files found in the specified directory.")
        return

    # Initialize a list to store sparsity percentages
    all_sparsities = []

    # Process each .npy file
    for i, npy_file_path in enumerate(npy_files_to_process):
        # Display progress
        sys.stdout.write(f'\rProcessing {i + 1} / {len(npy_files_to_process)}')
        sys.stdout.flush()

        # Calculate sparsity percentage
        sparsity = calculate_sparsity(npy_file_path)
        if sparsity is not None:
            all_sparsities.append(sparsity)

    print("\nDone processing all files.")

    # Convert sparsities to NumPy array
    all_sparsities = np.array(all_sparsities, dtype=np.float16)

    # Print sparsity statistics
    print("\nSparsity Statistics (Percentage):")
    print(f"  Min:    {np.amin(all_sparsities):.2f}%")
    print(f"  Max:    {np.amax(all_sparsities):.2f}%")
    print(f"  Mean:   {np.mean(all_sparsities):.2f}%")
    print(f"  Median: {np.median(all_sparsities):.2f}%")

    # Create output directory
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Save sparsity histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_sparsities, bins=20, color='blue', alpha=0.7)  # Use 20 bins for clarity
    plt.title("Sparsity Distribution (Percentage)")
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Frequency")
    plt.grid(True)
    histogram_path = os.path.join(output_dir, 'sparsity_histogram.png')
    plt.savefig(histogram_path)
    plt.close()

    print(f"Sparsity histogram saved to {histogram_path}")


if __name__ == '__main__':
    main()


# Sparsity Statistics (Percentage):
#   Min:    0.08%
#   Max:    2.51%
#   Mean:   1.79%
#   Median: 1.92%