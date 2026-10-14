import os
import pandas as pd
import shutil

class WeatherPathReader:
    def __init__(self, csv_path, root_dir, split, weather_dirs=None):
        """
        Args:
            csv_path (str): Path to CSV containing metadata
            root_dir (str): Base flattened dataset directory
            split (str): 'train','val','test'
            weather_dirs (dict): Optional mapping from weather to src dirs
        """
        self.df = pd.read_csv(csv_path)
        self.root = root_dir
        self.split = split
        self.weather_dirs = weather_dirs or {}
        # modality -> (subfolder, ext)
        self.modality_map = {
            'image':      ('frame_image',      '.png'),
            '8bitdepth':  ('8bit_clean_depth', '.png'),
            'velodyne':   ('raw_lidar',        '.bin'),
            'proj_depth': ('16bit_proj_depth', '.png'),
            'depthGT':    ('depth_GT',         '.png'),
        }

    def _build_path(self, name, modality, base_root=None):
        root = base_root or self.root
        folder, ext = self.modality_map[modality]
        seq, tail = name.split('_sync_')
        if modality != 'image':
            tail = tail.replace('image', modality, 1)
        fn = f"{seq}_sync_{tail}{ext}"
        return os.path.join(root, folder, self.split, fn)

    def get_input_paths(self, idx):
        row = self.df.iloc[idx]
        name = row['name']
        return {m: self._build_path(name, m) for m in self.modality_map}

    def copy_remaining_images(self, modality, dst_base_dir, exclude_weather, exclude_severity):
        """
        Copy all images of given modality except those matching exclude tags.
        """
        df_rest = self.df[~((self.df['weather']==exclude_weather)&(self.df['severity']==exclude_severity))]
        copied = []
        for _, row in df_rest.iterrows():
            name = row['name']
            src = self._build_path(name, modality, base_root=self.root)
            dst = self._build_path(name, modality, base_root=dst_base_dir)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            copied.append(dst)
        return copied

    def replace_and_tag_images(self, modality, weather, severity,
                               src_base_dir, dst_base_dir,
                               add_suffix=True, suffix=None):
        """
        For images matching weather+severity, copy from src_base_dir to dst, optionally tagging.
        """
        df_sub = self.df[(self.df['weather']==weather)&(self.df['severity']==severity)]
        results = []
        tag = suffix or f"{severity.lower()}{weather.lower()}"
        for _, row in df_sub.iterrows():
            name = row['name']
            # parse name
            seq, tail = name.split('_sync_')
            parts = tail.split('_')
            cam = f"image_{parts[1]}"; frame = parts[2]
            src = os.path.join(src_base_dir, severity, f"{seq}_sync", cam, f"{frame}.png")
            if not os.path.isfile(src):
                print(f"Src missing: {src}"); continue
            dst_orig = self._build_path(name, modality, base_root=dst_base_dir)
            os.makedirs(os.path.dirname(dst_orig), exist_ok=True)
            shutil.copyfile(src, dst_orig)
            if add_suffix:
                base, ext = os.path.splitext(dst_orig)
                dst_tag = f"{base}_{tag}{ext}"
                shutil.copyfile(src, dst_tag)
                results.append((dst_orig, dst_tag))
            else:
                results.append((dst_orig, None))
        return results

    def validate_replacement(self, modality, weather, severity,
                             tagged_pairs=None, src_base_dir=None):
        """
        Validate based on source availability.
        """
        df_sub = self.df[(self.df['weather']==weather)&(self.df['severity']==severity)]
        expected = len(df_sub)
        actual = len(tagged_pairs or [])
        missing_src = []
        if src_base_dir:
            for _, row in df_sub.iterrows():
                name = row['name']
                seq, tail = name.split('_sync_')
                parts = tail.split('_')
                cam = f"image_{parts[1]}"; frame = parts[2]
                src = os.path.join(src_base_dir, severity, f"{seq}_sync", cam, f"{frame}.png")
                if not os.path.isfile(src): missing_src.append(src)
        return expected, actual, missing_src

# 示例 (val split):
# ---------- Updated Example Invocation (val split) ----------

# ---------- Example (val split) ----------
if __name__=='__main__':
    reader = WeatherPathReader(
        csv_path= "/path/to/aurora-root/Dataset/AuroraDepth/Datasets/train.csv",
        # csv_path='/path/to/aurora-root/Dataset/AuroraDepth/Datasets/val.csv',
        root_dir='/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset/AuroDepth_frame_image',
        split='train'
    )
    # # 1) 先复制除 light-Fog 外的所有原图到新目录
    # copied = reader.copy_remaining_images(
    #     modality='image',
    #     dst_base_dir='/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset/AuroDepth_frame_image',
    #     exclude_weather='Fog',
    #     exclude_severity='moderate'
    # )
    # print(f"Copied {len(copied)} non-moderate-fog images")

    # 2) 再替换 light-Fog，并可选是否打 _lightfog 后缀
    tagged = reader.replace_and_tag_images(
        modality='image', weather='Fog', severity='moderate',
        src_base_dir='/cifs/Shares/WMGData/PhD-Project/Processing_AuroDepth_Data/Fog/fog',
        dst_base_dir='/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset/AuroDepth_frame_image',
        add_suffix=None
    )
    print(f"Processed {len(tagged)} moderate-fog images")

    # 3) 校验 source 完整性
    exp, act, miss = reader.validate_replacement(
        modality='image', weather='Fog', severity='moderate',
        tagged_pairs=tagged,
        src_base_dir='/cifs/Shares/WMGData/PhD-Project/Processing_AuroDepth_Data/Fog/fog'
    )
    print(f"Expected: {exp}, Actual: {act}")
    if miss:
        print("Src missing:", miss)
    else:
        print("All source images present!")