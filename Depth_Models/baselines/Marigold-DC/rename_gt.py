import os

def rename_images(folder_path):
    for filename in os.listdir(folder_path):
        # 只处理包含指定字符串的文件
        if "_sync_image_" in filename:
            old_path = os.path.join(folder_path, filename)
            new_filename = filename.replace(
                "_sync_image_", "_sync_groundtruth_depth_"
            )
            new_path = os.path.join(folder_path, new_filename)

            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    folder_path = "/path/to/aurora-root/Code/Marigold-DC/Marigold_DC_result_16bit_Aurora_test"  # ← 改成你的文件夹路径
    rename_images(folder_path)
