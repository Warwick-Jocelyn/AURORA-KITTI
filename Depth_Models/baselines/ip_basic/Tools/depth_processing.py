import os
import time
import cv2
import numpy as np
from ip_basic import depth_map_utils_original

class DepthProcessor:
    def __init__(self, fill_type='multiscale', extrapolate=False, blur_type='bilateral'):
        self.fill_type = fill_type
        self.extrapolate = extrapolate
        self.blur_type = blur_type

    def process_image(self, input_path, output_path, idx, total):
        """Processes a single depth image and saves it while maintaining relative paths."""
        # Timing
        start_time = time.time()

        # Load depth projections
        depth_image = cv2.imread(input_path, cv2.IMREAD_ANYDEPTH)
        if depth_image is None:
            print(f"Warning: Unable to read {input_path}")
            return

        projected_depths = np.float32(depth_image / 256.0)

        # Fill in missing depth
        if self.fill_type == 'fast':
            final_depths = depth_map_utils_original.fill_in_fast(
                projected_depths, extrapolate=self.extrapolate, blur_type=self.blur_type
            )
        elif self.fill_type == 'multiscale':
            final_depths, _ = depth_map_utils_original.fill_in_multiscale(
                projected_depths, extrapolate=self.extrapolate, blur_type=self.blur_type
            )
        else:
            raise ValueError(f"Invalid fill_type: {self.fill_type}")

        # Save processed depth image
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        final_depths = (final_depths * 256).astype(np.uint16)
        cv2.imwrite(output_path, final_depths)

        elapsed_time = time.time() - start_time
        print(f"\rProcessing {idx + 1}/{total}, Time: {elapsed_time:.4f}s", end='', flush=True)
