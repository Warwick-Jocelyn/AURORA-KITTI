import os
import cv2
import numpy as np
import pandas as pd
# from scikit-image.restoration import denoise_bilateral
from scipy.ndimage import minimum_filter


def transmission_map(depth_map, beta):
    # Normalize depth to [0,1] assuming 8-bit scaled by 255 mapping max distance 80
    t = np.exp(-beta * (depth_map/ 255* 80))
    return t



def get_dark_channel(image, window_size=15):
    """
    Compute the dark channel of the input RGB image.

    Parameters:
        image (numpy.ndarray): Input RGB image.
        window_size (int): Size of the square neighborhood window.

    Returns:
        I_dark (numpy.ndarray): Dark channel of the image.
        I_eroded (numpy.ndarray): Eroded image of the same size as input.
    """
    # Perform minimum filtering on each channel
    I_eroded = np.zeros_like(image)
    for c in range(3):  # Assuming image shape is (H, W, 3)
        I_eroded[:, :, c] = minimum_filter(image[:, :, c], size=window_size, mode='reflect')

    # Take the minimum across the color channels
    I_dark = np.min(I_eroded, axis=2)

    return I_dark


def estimate_atmospheric_light(image, dark_channel):
    """
    Estimate the atmospheric light in the image using the dark channel.

    Parameters:
        image (numpy.ndarray): Input RGB image.
        dark_channel (numpy.ndarray): Grayscale dark channel of the image.

    Returns:
        L (numpy.ndarray): Estimated atmospheric light (1x1x3).
        index_L (tuple): (row, col) index of the pixel with atmospheric light.
    """
    brightest_pixels_fraction = 1 / 1000
    height, width = dark_channel.shape
    number_of_pixels = height * width
    brightest_pixels_count = max(1, int(brightest_pixels_fraction * number_of_pixels))
    if brightest_pixels_count % 2 == 0:
        brightest_pixels_count += 1  # Ensure it's odd

    # Flatten dark channel and get indices of the brightest pixels
    I_dark_vector = dark_channel.flatten()
    indices = np.argsort(I_dark_vector)[::-1]  # Descending order
    brightest_pixels_indices = indices[:brightest_pixels_count]

    # Flatten grayscale version of the RGB image
    I_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    I_gray_vector = I_gray.flatten()
    I_gray_vector_brightest_pixels = I_gray_vector[brightest_pixels_indices]

    # Find median intensity among brightest pixels
    median_intensity = np.median(I_gray_vector_brightest_pixels)
    median_candidates = np.where(I_gray_vector_brightest_pixels == median_intensity)[0]
    index_median_intensity = median_candidates[0]

    index_L_flat = brightest_pixels_indices[index_median_intensity]
    row_L = index_L_flat // width
    column_L = index_L_flat % width

    L = image[row_L, column_L, :]

    return L


# def refine_transmission(t_map, image, sigma_spatial=20, sigma_intensity=0.1):
#     return scikit-image.denoise_bilateral(t_map, sigma_color=sigma_intensity,
#                               sigma_spatial=sigma_spatial, channel_axis=None)

def synthesize_foggy_image(clear_image, trans_map, atmospheric_light):
    """
    Synthesizes the foggy image using the transmission map and atmospheric light.
    """
    I0 = clear_image.astype(np.float32) / 255.0
    max_distance = np.full(trans_map[..., np.newaxis].shape, 1)
    t =  max_distance - trans_map[..., np.newaxis]
    A = atmospheric_light[np.newaxis, np.newaxis, :] / 255.0
    foggy = I0 * t + A * (1 - t)
    return (foggy*255).astype(np.uint8)

# def synthesize_foggy_image(clear_image, trans_map, atmospheric_light):
#     """
#     Synthesizes the foggy image using the transmission map and atmospheric light.
#     """
#     I0 = clear_image.astype(np.float32) / 255.0
#     max_distance = np.full(trans_map[:, :, 0][..., np.newaxis].shape, 1)
#     t =  max_distance - trans_map[:, :, 0][..., np.newaxis]
#     A = atmospheric_light[np.newaxis, np.newaxis, :] / 255.0
#     foggy = I0 * t + A * (1 - t)
#     return (foggy*255).astype(np.uint8)

if __name__ == '__main__':
    # Configuration
    csv_path = '/path/to/aurora-root/Dataset/AuroraDepth/Datasets/train.csv'
    root = '/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset'
    split = 'train'
    output_root = '/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset'
    beta = 0.008  # adjust for heavy fog
    add_suffix = None
    suffix = 'heavyfog'

    df = pd.read_csv(csv_path)
    df_h = df[(df.weather=='Fog') & (df.severity=='heavy')]
    generated_count = 0

    for _, row in df_h.iterrows():
        name = row['name']  # e.g. ..._sync_image_02_0000000006
        # Parse camera and frame
        seq, tail = name.split('_sync_')
        parts = tail.split('_')
        cam = f'image_{parts[1]}'
        frame = parts[2]

        # Paths
        rgb_path = os.path.join(root, 'frame_image', split, f'{name}.png')
        depth_name = name.replace('sync_image', 'sync_8bitdepth')
        depth_path = os.path.join(root, '8bit_clean_depth', split, f'{depth_name}.png')

        # Read
        if not os.path.isfile(rgb_path) or not os.path.isfile(depth_path):
            print(f'Missing input for {name}')
            continue
        img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        # handle scaling
        # if depth.max() > 80:
        #     depth /= 256.0
        # interpolate missing
        mask = (depth<=0)|np.isnan(depth)
        if mask.any(): depth[mask] = np.nanmedian(depth)

        # Fog pipeline
        t0 = transmission_map(depth, beta)
        # t1 = refine_transmission(t0, img)
        dark = get_dark_channel(img, window_size=15)
        A = estimate_atmospheric_light(img, dark)
        fog = synthesize_foggy_image(img, t0, A)




        # Output
        out_dir = os.path.join(output_root, 'frame_image', split)
        os.makedirs(out_dir, exist_ok=True)
        base_fn = f'{name}.png'
        out_path = os.path.join(out_dir, base_fn)
        cv2.imwrite(out_path, cv2.cvtColor(fog, cv2.COLOR_RGB2BGR))
        generated_count += 1

        if add_suffix:
            suf_path = os.path.join(out_dir, f'{name}_{suffix}.png')
            cv2.imwrite(suf_path, cv2.cvtColor(fog, cv2.COLOR_RGB2BGR))
        print(f'Generated foggy for {name} in {out_path}')

    print(f'\nTotal foggy images generated: {generated_count}')

