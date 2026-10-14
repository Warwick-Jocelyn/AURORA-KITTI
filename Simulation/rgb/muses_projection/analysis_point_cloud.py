import os
import numpy as np

def analyze_distance_maps(folder_path):
    """
    Analyze .npy files in the given folder to calculate global and per-file statistics for distance maps.

    Prints:
    - Min value (including zeros)
    - Min value (excluding zeros) for each file.

    Returns:
        dict: Aggregated statistics including global min, global max, average mean, and average non-zero percentage.
    """
    global_min_with_zeros = float('inf')
    global_min_without_zeros = float('inf')
    global_max = float('-inf')
    total_mean_values = []
    total_non_zero_percentages = []
    total_files = 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                data = np.load(file_path)

                # Calculate per-file statistics
                min_with_zeros = data.min()  # Min value including zeros
                non_zero_data = data[data > 0]
                min_without_zeros = non_zero_data.min() if non_zero_data.size > 0 else None

                # Print per-file minimum values
                print(f"File: {file_path}")
                print(f"  Min (including zeros): {min_with_zeros:.4f}")
                print(f"  Min (excluding zeros): {min_without_zeros:.4f}" if min_without_zeros is not None else "  Min (excluding zeros): None")

                # Update global min and max
                global_min_with_zeros = min(global_min_with_zeros, min_with_zeros)
                if non_zero_data.size > 0:
                    global_min_without_zeros = min(global_min_without_zeros, min_without_zeros)
                global_max = max(global_max, data.max())

                # Exclude zero values for mean and percentage calculations
                total_pixels = data.size
                non_zero_pixels = non_zero_data.size
                if non_zero_pixels > 0:
                    mean_val = non_zero_data.mean()
                    non_zero_percentage = (non_zero_pixels / total_pixels) * 100

                    # Accumulate statistics
                    total_mean_values.append(mean_val)
                    total_non_zero_percentages.append(non_zero_percentage)

                total_files += 1

    # Calculate aggregated statistics
    avg_mean = np.mean(total_mean_values) if total_mean_values else None
    avg_non_zero_percentage = np.mean(total_non_zero_percentages) if total_non_zero_percentages else None

    return {
        "global_min_with_zeros": global_min_with_zeros if global_min_with_zeros != float('inf') else None,
        "global_min_without_zeros": global_min_without_zeros if global_min_without_zeros != float('inf') else None,
        "global_max": global_max if global_max != float('-inf') else None,
        "avg_mean": avg_mean,
        "avg_non_zero_percentage": avg_non_zero_percentage,
        "total_files": total_files,
    }


def print_summary(summary):
    """
    Print the summary statistics in a readable format.

    Parameters:
        summary (dict): Dictionary containing aggregated statistics.
    """
    print("\nSummary Statistics:")
    print(f"Total Files Processed: {summary['total_files']}")
    print(f"Global Min Value (including zeros): {summary['global_min_with_zeros']:.4f}" if summary['global_min_with_zeros'] is not None else "Global Min Value (including zeros): None")
    print(f"Global Min Value (excluding zeros): {summary['global_min_without_zeros']:.4f}" if summary['global_min_without_zeros'] is not None else "Global Min Value (excluding zeros): None")
    print(f"Global Max Value: {summary['global_max']:.4f}" if summary['global_max'] is not None else "Global Max Value: None")
    print(f"Average Mean Value: {summary['avg_mean']:.4f}" if summary['avg_mean'] is not None else "Average Mean Value: None")
    print(f"Average % of Non-Zero Pixels: {summary['avg_non_zero_percentage']:.2f}%" if summary['avg_non_zero_percentage'] is not None else "Average % of Non-Zero Pixels: None")


# Example usage
folder = "/path/to/aurora-root/Code/MUSES/Dataset/lidar_kernel_1_distance_maps"
summary = analyze_distance_maps(folder)
print_summary(summary)

# Summary Statistics:
# Total Files Processed: 2500
# Global Min Value (including zeros): 0.0000
# Global Min Value (except the zeros): 1.4727
# Global Max Value: 200.0237
# Average Mean Value: 21.1467
# Average % of Non-Zero Pixels: 1.79%