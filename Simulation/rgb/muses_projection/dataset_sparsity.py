import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    input_depth_dir = os.path.expanduser(
        '/path/to/aurora-root/Code/MUSES/Dataset/lidar_kernel_1_distance_maps_16bit')

    # Use glob to recursively find all image files under the directory
    images_to_use = sorted(glob.glob(os.path.join(input_depth_dir, '**', '*.png'), recursive=True))

    if not images_to_use:
        print("No images found in the specified directory.")
        return

    # Pre-allocate sparsity array
    num_images = len(images_to_use)
    all_sparsities = np.zeros(num_images, dtype=np.float32)

    for i, depth_image_path in enumerate(images_to_use):
        # Print progress
        sys.stdout.write(f'\rProcessing index {i + 1} / {num_images}')
        sys.stdout.flush()

        # Load depth image
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
        if depth_image is None:
            print(f"\nWarning: Failed to read image {depth_image_path}")
            continue

        # Downscale depth precision and simplify sparsity computation
        depth_map = np.round(depth_image / 256.0, 1).astype(np.float32)

        # Count valid pixels and calculate sparsity
        num_valid_pixels = np.count_nonzero(depth_map > 0.0)
        num_pixels = depth_image.size  # Total number of pixels

        # Assume a sparsity adjustment factor (2/3 as in original)
        all_sparsities[i] = (num_valid_pixels / (num_pixels * 0.369)) * 100  # Convert to percentage

        # 36.9% 

    print('\n')
    print('Sparsity')
    print('Min:   ', np.amin(all_sparsities), '%')
    print('Max:   ', np.amax(all_sparsities), '%')
    print('Mean:  ', np.mean(all_sparsities), '%')
    print('Median:  ', np.median(all_sparsities), '%')

    # Save histogram
    os.makedirs('outputs', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(all_sparsities, bins=20, color='blue', alpha=0.7)
    plt.title("Sparsity Distribution", fontsize=16)
    plt.xlabel("Sparsity (%)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(color='lightgrey', linestyle='--', linewidth=0.5)
    plt.savefig('outputs/sparsity_histogram.png')
    plt.close()

if __name__ == '__main__':
    main()

# Processing index 7885 / 7885