import os
import shutil
from glob import glob
from tqdm import tqdm

# du -h --max-depth=1

# <sequence_id>/<camera>/<frame_id>/<modality>/  
# 变成<sequence_id>_<camera>_<frame_id>_<modality>,
# 并且最后加天气之后变成<sequence_id>_<camera>_<frame_id>_<modality>_<weather>_<severity>.（png/bin)

# #Flatten PNG Files
def flatten_png_files(src_root, dst_root):
    os.makedirs(dst_root, exist_ok=True)

    # Count total number of files to process for progress bar
    total_files = 0
    for sequence_path in glob(os.path.join(src_root, '*')):
        for camera in ['image_02', 'image_03']:
            camera_path = os.path.join(sequence_path, camera, 'data')
            if os.path.isdir(camera_path):
                total_files += len(glob(os.path.join(camera_path, '*.png')))

    with tqdm(total=total_files, desc="Flattening PNGs") as pbar:
        for sequence_path in glob(os.path.join(src_root, '*')):
            sequence_id = os.path.basename(sequence_path)

            for camera in ['image_02', 'image_03']:
                camera_path = os.path.join(sequence_path, camera, 'data')
                if not os.path.isdir(camera_path):
                    continue

                for file_path in glob(os.path.join(camera_path, '*.png')):
                    frame_id = os.path.splitext(os.path.basename(file_path))[0]

                    new_filename = f"{sequence_id}_{camera}_{frame_id}.png"
                    dst_file = os.path.join(dst_root, new_filename)

                    shutil.copy(file_path, dst_file)
                    pbar.update(1)

# Example usage
flatten_png_files(
    src_root="/cifs/Shares/WMGData/PhD-Project/Processing_AuroDepth_Data/Fog/fog/75m_resized/",
    dst_root="/cifs/Shares/WMGData/PhD-Project/Processing_AuroDepth_Data/Fog/75mFlattened/"
)

def flatten_proj_depth(src_root, dst_root):
    dst_root = os.path.abspath(dst_root)
    os.makedirs(dst_root, exist_ok=True)
    print(f"Saving projected depth to: {dst_root}")

    total_files = 0
    for sequence_path in glob(os.path.join(src_root, '*')):
        for camera in ['image_02', 'image_03']:
            depth_path = os.path.join(sequence_path, 'proj_depth', 'velodyne_raw', camera)
            if os.path.isdir(depth_path):
                total_files += len(glob(os.path.join(depth_path, '*.png')))

    with tqdm(total=total_files, desc="Flattening Projected Depth PNGs") as pbar:
        for sequence_path in glob(os.path.join(src_root, '*')):
            sequence_id = os.path.basename(sequence_path)

            for camera in ['image_02', 'image_03']:
                depth_path = os.path.join(sequence_path, 'proj_depth', 'velodyne_raw', camera)
                if not os.path.isdir(depth_path):
                    continue

                for file_path in glob(os.path.join(depth_path, '*.png')):
                    frame_id = os.path.splitext(os.path.basename(file_path))[0]
                    camera_id = camera.split('_')[-1]

                    new_filename = f"{sequence_id}_proj_depth_{camera_id}_{frame_id}.png"
                    dst_file = os.path.join(dst_root, new_filename)

                    try:
                        shutil.copy(file_path, dst_file)
                    except Exception as e:
                        print(f"Failed to copy {file_path} to {dst_file}: {e}")

                    pbar.update(1)

# # Example usage
# flatten_proj_depth(
#     src_root="/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Clean_proj_depth_ori_fixed_46968",
#     dst_root="/path/to/aurora-root/Dataset/AuroraDepth/Flattened_Dataset/16bit_Camera_proj_depth"
# )

def flatten_lidar_bin(src_root, dst_root):
    dst_root = os.path.abspath(dst_root)
    os.makedirs(dst_root, exist_ok=True)
    print(f"Saving lidar bin files to: {dst_root}")

    total_files = 0
    for sequence_path in glob(os.path.join(src_root, '*')):
        lidar_path = os.path.join(sequence_path, 'velodyne_points', 'data')
        if os.path.isdir(lidar_path):
            total_files += len(glob(os.path.join(lidar_path, '*.bin')))

    with tqdm(total=total_files, desc="Flattening Lidar BINs") as pbar:
        for sequence_path in glob(os.path.join(src_root, '*')):
            sequence_id = os.path.basename(sequence_path)
            lidar_path = os.path.join(sequence_path, 'velodyne_points', 'data')
            if not os.path.isdir(lidar_path):
                continue

            for file_path in glob(os.path.join(lidar_path, '*.bin')):
                frame_id = os.path.splitext(os.path.basename(file_path))[0]
                new_filename = f"{sequence_id}_velodyne_points_{frame_id}.bin"
                dst_file = os.path.join(dst_root, new_filename)

                try:
                    shutil.copy(file_path, dst_file)
                except Exception as e:
                    print(f"Failed to copy {file_path} to {dst_file}: {e}")

                pbar.update(1)

# Example usage

# flatten_lidar_bin(
#     src_root="/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Clear_Lidar",
#     dst_root="/path/to/aurora-root/Dataset/AuroraDepth/Flattened_Dataset/raw_velodyne_points"
# )

def flatten_gt_depth(src_root, dst_root):
    dst_root = os.path.abspath(dst_root)
    os.makedirs(dst_root, exist_ok=True)
    print(f"Saving GT depth files to: {dst_root}")

    total_files = 0
    for sequence_path in glob(os.path.join(src_root, '*')):
        for camera in ['image_02', 'image_03']:
            gt_path = os.path.join(sequence_path, 'proj_depth', 'groundtruth', camera)
            if os.path.isdir(gt_path):
                total_files += len(glob(os.path.join(gt_path, '*.png')))

    with tqdm(total=total_files, desc="Flattening GT Depth PNGs") as pbar:
        for sequence_path in glob(os.path.join(src_root, '*')):
            sequence_id = os.path.basename(sequence_path)

            for camera in ['image_02', 'image_03']:
                gt_path = os.path.join(sequence_path, 'proj_depth', 'groundtruth', camera)
                if not os.path.isdir(gt_path):
                    continue

                for file_path in glob(os.path.join(gt_path, '*.png')):
                    frame_id = os.path.splitext(os.path.basename(file_path))[0]
                    camera_id = camera.split('_')[-1]

                    new_filename = f"{sequence_id}_depthGT_{camera_id}_{frame_id}.png"
                    dst_file = os.path.join(dst_root, new_filename)

                    try:
                        shutil.copy(file_path, dst_file)
                    except Exception as e:
                        print(f"Failed to copy {file_path} to {dst_file}: {e}")

                    pbar.update(1)

# flatten_gt_depth(
#     src_root="/path/to/aurora-root/Dataset/AuroraDepth/Datasets/data_depth_GT_fixed_46968",
#     dst_root="/path/to/aurora-root/Dataset/AuroraDepth/Flattened_Dataset/Depth_GT"
# )

def flatten_8bit_depth(src_root, dst_root):
    dst_root = os.path.abspath(dst_root)
    os.makedirs(dst_root, exist_ok=True)
    print(f"Saving 8-bit depth files to: {dst_root}")

    # Step 1: Count total files
    total_files = 0
    for sequence_path in glob(os.path.join(src_root, '*')):
        for camera in ['image_02', 'image_03']:
            depth_path = os.path.join(sequence_path, camera, 'data')
            if os.path.isdir(depth_path):
                total_files += len(glob(os.path.join(depth_path, '*.png')))

    # Step 2: Flatten and copy with proper naming
    with tqdm(total=total_files, desc="Flattening 8-bit Depth PNGs") as pbar:
        for sequence_path in glob(os.path.join(src_root, '*')):
            sequence_id = os.path.basename(sequence_path)

            for camera in ['image_02', 'image_03']:
                depth_path = os.path.join(sequence_path, camera, 'data')
                if not os.path.isdir(depth_path):
                    continue

                for file_path in glob(os.path.join(depth_path, '*.png')):
                    frame_id = os.path.splitext(os.path.basename(file_path))[0]
                    camera_id = camera.split('_')[-1]
                    new_filename = f"{sequence_id}_8bitdepth_{camera_id}_{frame_id}.png"
                    dst_file = os.path.join(dst_root, new_filename)

                    try:
                        shutil.copy(file_path, dst_file)
                    except Exception as e:
                        print(f"Failed to copy {file_path} to {dst_file}: {e}")

                    pbar.update(1)
# flatten_8bit_depth(
#     src_root="/path/to/aurora-root/Dataset/AuroraDepth_depth_anything_batch_output/Clear_RGB_fixed_46968_original_depth",
#     dst_root="/path/to/aurora-root/Dataset/AuroraDepth/Flattened_Dataset/8bit_clean_depth"
# )
