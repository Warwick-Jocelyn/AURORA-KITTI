import os
import pandas as pd
import shutil

class WeatherPathReader:
    def __init__(self, csv_path, root_dir, split, weather_dirs=None):
        self.df = pd.read_csv(csv_path)
        self.root = root_dir
        self.split = split
        self.weather_dirs = weather_dirs or {}
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

    def replace_snow_images(self, modality, src_base_dir, severities=None):
        """
        Replace all Snow images with specified severities from src_base_dir to root_dir.
        Args:
            modality (str): image modality
            src_base_dir (str): base source path (e.g., snowgan_fixed_XXXX)
            severities (list of str): optional list of severities to filter (e.g., ['light', 'moderate'])
        """
        df_snow = self.df[self.df['weather'] == 'Rain']
        if severities:
            df_snow = df_snow[df_snow['severity'].isin(severities)]

        total_replaced = 0
        replaced_by_severity = {}

        for _, row in df_snow.iterrows():
            name = row['name']
            severity = row['severity']
            seq, tail = name.split('_sync_')
            parts = tail.split('_')
            cam = f"image_{parts[1]}"
            frame = parts[2]

            # Construct source path
            src_path = os.path.join(src_base_dir, f"{seq}_sync_{cam}_{frame}.png")
            if not os.path.isfile(src_path):
                print(f"[Missing] {src_path}")
                continue

            dst_path = self._build_path(name, modality, base_root=self.root)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            shutil.copyfile(src_path, dst_path)
            total_replaced += 1
            replaced_by_severity[severity] = replaced_by_severity.get(severity, 0) + 1

        print(f"✅ Total Replaced: {total_replaced} Snow images in split '{self.split}'")
        for sev, count in replaced_by_severity.items():
            print(f"  - {sev}: {count}")


# ---------- Example Usage ----------
if __name__ == '__main__':
    SPLIT = 'train'  # 可改为 'train' 或 'test'

    reader = WeatherPathReader(
        csv_path=f"/path/to/aurora-root/Dataset/AuroraKITTI/Datasets/{SPLIT}.csv",
        root_dir="/path/to/aurora-root/Dataset/AuroraKITTI/Datasets/Split_Flattened_Dataset",
        split=SPLIT
    )

    reader.replace_snow_images(
        modality='image',
        src_base_dir=f"/path/to/aurora-root/Dataset/AuroraKITTI/Codes_RGB_AuraKITTI_Generation_Tools/Rain_generation/output/frame_image/{SPLIT}",
        severities=['light']
    )
 
#  /path/to/aurora-root/Dataset/AuroraKITTI/Codes_RGB_AuraKITTI_Generation_Tools/Rain_generation/output/frame_image/val/2011_09_26_drive_0002_sync_image_02_0000000015.png