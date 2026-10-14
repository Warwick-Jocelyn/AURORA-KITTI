from PIL import Image
import numpy as np
import os

def get_image_info(image_path):
  # Open the image
  with Image.open(image_path) as img:
    # Convert image to numpy array
    img_array = np.array(img)

    # Get mode of the image to determine bit depth
    mode = img.mode
    if mode in ['1']:  # 1 bit per pixel
      bit_depth = 1
    elif mode in ['L', 'P']:  # 8 bits per pixel
      bit_depth = 8
    elif mode in ['I;16', 'I;16B', 'I;16L']:  # 16 bits per pixel
      bit_depth = 16
    elif mode in ['RGB', 'YCbCr']:  # 24 bits per pixel (8 bits for each of 3 channels)
      bit_depth = 24
    elif mode in ['RGBA', 'CMYK']:  # 32 bits per pixel (8 bits for each of 4 channels)
      bit_depth = 32
    else:
      bit_depth = 'Unknown'

    # Determine channel number based on the image array shape
    if len(img_array.shape) == 2:  # Grayscale image
      channels = 1
    elif len(img_array.shape) == 3:  # Color image
      channels = img_array.shape[2]
    else:
      channels = 'Unknown'

    # Calculate max, min, and mean
    max_val = img_array.max()
    min_val = img_array.min()
    mean_val = img_array.mean()

    return bit_depth, channels, max_val, min_val, mean_val

# Example usage
image_path = '/path/to/aurora-root/Code/UniK3D/assets/demo/dl3dv_depth_x256.png'
bit_depth, channels, max_val, min_val, mean_val = get_image_info(image_path)
print(f"The bit depth of the image is: {bit_depth}")
print(f"The number of channels in the image is: {channels}")
print(f"Maximum value in the image: {max_val/255}")
print(f"Minimum value in the image: {min_val/255}")
print(f"Average value in the image: {mean_val/255}")

# (muses) user@host:~/Dataset$ python Bit_check.py
# The bit depth of the image is: 16
# The number of channels in the image is: 1
# Maximum value in the image: 20298 * 85m
# Minimum value in the image: 0
# Average value in the image: 191.0244631242524

# The bit depth of the image is: Unknown
# The number of channels in the image is: 1
# Maximum value in the image: 14757
# Minimum value in the image: 0
# Average value in the image: 1877.033296833065

# The bit depth of the image is: 16bit
# The number of channels in the image is: 1
# Maximum value in the image: 7811
# Minimum value in the image: 0
# Average value in the image: 1591.766101986044

# The bit depth of the image is: 24
# The number of channels in the image is: 3
# Maximum value in the image: 255
# Minimum value in the image: 0
# Average value in the image: 84.4066323134729

# The bit depth of the image is: 24
# The number of channels in the image is: 3
# Maximum value in the image: 255
# Minimum value in the image: 0
# Average value in the image: 91.56773376274826