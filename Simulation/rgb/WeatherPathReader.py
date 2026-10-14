import os
import pandas as pd

class WeatherPathReader:
    def __init__(self, csv_path, root_dir, split, weather_dirs=None):
        """
        Args:
            csv_path (str): Path to train/val/test CSV
            root_dir (str): Base dataset directory
            split (str): One of 'train','val','test'
            weather_dirs (dict): Optional mapping from weather names to pre-generated output dirs
                                 e.g. {'Rain': '/path/to/rain_images', 'Snow': '/path/to/snow_images'}
        """
        self.df = pd.read_csv(csv_path)
        self.root = root_dir
        self.split = split
        self.weather_dirs = weather_dirs or {}

        # Subfolders and extensions for each modality
        self.modality_map = {
            'image':      ('frame_image',      '.png'),
            '8bitdepth':  ('8bit_clean_depth', '.png'),
            'velodyne':   ('raw_lidar',        '.bin'),
            'proj_depth': ('16bit_proj_depth', '.png'),
            'depthGT':    ('depth_GT',         '.png'),
        }

    def _build_path(self, name, modality):
        """
        构造给定模态的完整文件路径。
        """
        folder, ext = self.modality_map[modality]
        seq, tail = name.split('_sync_')  # seq, e.g. '2011_09_26_drive_0002'; tail, e.g. 'image_02_0000000006'
        if modality != 'image':
            tail = tail.replace('image', modality, 1)
        filename = f"{seq}_sync_{tail}{ext}"
        return os.path.join(self.root, folder, self.split, filename)

    def get_row(self, idx):
        return self.df.iloc[idx]

    def get_input_paths(self, idx):
        """
        返回标准输入路径：RGB 图像、8bit 深度、原始点云，以及新增的投影深度和深度 GT。
        """
        row = self.get_row(idx)
        name = row['name']
        return {
            'image':      self._build_path(name, 'image'),
            'depth':      self._build_path(name, '8bitdepth'),
            'lidar':      self._build_path(name, 'velodyne'),
            'proj_depth': self._build_path(name, 'proj_depth'),
            'depthGT':    self._build_path(name, 'depthGT'),
            'meta':       row.to_dict()
        }

    def get_weather_output(self, idx):
        """
        若预生成了天气效果图，则返回对应路径，否则 None。
        """
        row = self.get_row(idx)
        w = row['weather']
        if w in self.weather_dirs:
            name = row['name'] + '.png'
            return os.path.join(self.weather_dirs[w], self.split, name)
        return None

    def all_paths(self, idx):
        """
        返回包含所有输入以及（可选）天气效果输出的路径字典。
        """
        out = self.get_input_paths(idx)
        wout = self.get_weather_output(idx)
        if wout:
            out['weather_image'] = wout
        return out

# Example usage:
# reader = WeatherPathReader(
#     csv_path='csv/val.csv',
#     root_dir='Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset',
#     split='val',
#     weather_dirs={
#         'Rain': '/prebuilt/rain',
#         'Snow': '/prebuilt/snow',
#         'Fog': '/prebuilt/fog'
#     }
# )
# paths = reader.all_paths(0)
# print(paths)