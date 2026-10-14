# kitti.py
import random
import os
import numpy as np
import glob
from PIL import Image
import torch
import augs
import cv2

import csv
import re

__all__ = [
    'KITTI',
]

TARGET_HEIGHT = 375
TARGET_WIDTH = 1242


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


class KITTI(torch.utils.data.Dataset):
    """
    kitti depth completion dataset
    Adds labels from CSV:
      - label_2d: Tensor([tod_id, weather_id])  where Night=0 Day=1; Clear=1 Rain=2 Snow=3 Fog=4
      - cls:      Tensor(cls_id) in [0..7] where cls_id = weather_index*2 + tod_index
                 weather_index: Clear=0 Rain=1 Snow=2 Fog=3
                 tod_index:     Night=0 Day=1
    """

    def __init__(self, path='datas/kitti', mode='train',
                 height=256, width=1216,
                 mean=(90.9950, 96.2278, 94.3213),
                 std=(79.2382, 80.5267, 82.1483),
                 RandCrop=False, tp_min=50,
                 train_csv=None, val_csv=None,
                 *args, **kwargs):

        self.base_dir = path
        self.height = height
        self.width = width
        self.mode = mode

        if mode == 'train':
            self.transform = augs.Compose([
                augs.Jitter(),
                augs.Flip(),
                augs.Norm(mean=mean, std=std),
            ])
        else:
            self.transform = augs.Compose([
                augs.Norm(mean=mean, std=std),
            ])

        self.RandCrop = RandCrop and mode == 'train'
        self.tp_min = tp_min

        # ----------------- label 定义（2D + 8-class） -----------------
        # 2D label ids (as you requested):
        self.tod_to_id = {'Night': 0, 'Day': 1}
        self.weather_to_id = {'Clear': 1, 'Rain': 2, 'Snow': 3, 'Fog': 4}

        # for cls_id (0..7) mapping + debug
        self._weather_list_for_cls = ['Clear', 'Rain', 'Snow', 'Fog']  # index 0..3
        self._tod_list_for_cls = ['Night', 'Day']                     # index 0..1
        self.num_weather = 4
        self.num_tod = 2
        self.num_cls = 8

        # dicts: (sequence_id, cam, frame10) -> label
        self.sample_to_2d = {}   # -> (tod_id, weather_id)
        self.sample_to_cls = {}  # -> cls_id

        # your absolute CSV defaults
        if train_csv is None:
            # train_csv = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3Cdatas/kitti/train.csv"
            train_csv = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/train.csv"
        if val_csv is None:
            val_csv = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val.csv"
    

        # Only load labels for train/val/selval. test has no GT label.
        if self.mode == 'train' and os.path.isfile(train_csv):
            self.sample_to_cls, self.sample_to_2d = self._load_csv_as_label_maps(train_csv)
        elif self.mode in ['val', 'selval'] and os.path.isfile(val_csv):
            self.sample_to_cls, self.sample_to_2d = self._load_csv_as_label_maps(val_csv)

        self._warn_missing_label_once = True  # avoid spamming

        # ----------------- 文件列表 -----------------
        if mode == 'train':
            # large-scale train raw data
            self.depth_path = os.path.join(self.base_dir, 'data_depth_annotated', 'train')
            self.lidar_path = os.path.join(self.base_dir, 'data_depth_velodyne', 'train')
            self.depths = list(sorted(glob.iglob(self.depth_path + "/**/*.png", recursive=True)))
            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/**/*.png", recursive=True)))
            self.images = None  # train not directly using this list

        elif mode in ['val', 'selval']:
            # val_selection_cropped
            self.base_val_dir = os.path.join(self.base_dir, 'val_selection_cropped')
            self.lidar_path = os.path.join(self.base_val_dir, 'velodyne_raw')
            self.depth_path = os.path.join(self.base_val_dir, 'groundtruth_depth')
            self.image_path = os.path.join(self.base_val_dir, 'image')

            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/*.png", recursive=True)))
            # align depth & image with lidar filenames
            self.depths = [x.replace('velodyne_raw', 'groundtruth_depth').replace(
                '_velodyne_raw_', '_groundtruth_depth_') for x in self.lidars]
            self.images = [x.replace('velodyne_raw', 'image').replace(
                '_velodyne_raw_', '_image_') for x in self.lidars]

        elif mode == 'test':
            self.lidar_path = os.path.join(self.base_dir, 'test_depth_completion_anonymous', 'velodyne_raw')
            self.image_path = os.path.join(self.base_dir, 'test_depth_completion_anonymous', 'image')
            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/*.png", recursive=True)))
            self.images = [x.replace('velodyne_raw', 'image') for x in self.lidars]
            self.depths = self.lidars  # placeholder

        else:
            raise ValueError(f"Unknown mode: {mode}")

        assert len(self.depths) == len(self.lidars)
        self.names = [os.path.split(path)[-1] for path in self.depths]

        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        self.xy = np.stack((xx, yy), axis=-1)

    # ----------------- debug helpers -----------------
    # def cls_id_to_str(self, cls_id: int) -> str:
    #     """cls_id (0..7) -> 'Rain,Night'"""
    #     if cls_id < 0 or cls_id >= self.num_cls:
    #         return "Unknown,Unknown"
    #     weather_idx = cls_id // 2
    #     tod_idx = cls_id % 2
    #     return f"{self._tod_list_for_cls[tod_idx]},{self._weather_list_for_cls[weather_idx]}"

    def cls_id_to_str(self, cls_id: int) -> str:
        """cls_id (0..7) -> 'Day,Fog' (TOD first, then Weather)"""
        if cls_id < 0 or cls_id >= self.num_cls:
            return "Unknown,Unknown"
        weather_idx = cls_id // 2
        tod_idx = cls_id % 2
        return f"{self._tod_list_for_cls[tod_idx]},{self._weather_list_for_cls[weather_idx]}"


    def label_2d_to_str(self, label_2d) -> str:
        """[tod_id, weather_id] -> 'Day,Fog' (TOD first, then Weather)"""
        tod_id = int(label_2d[0])
        weather_id = int(label_2d[1])

        tod = 'Day' if tod_id == 1 else ('Night' if tod_id == 0 else 'Unknown')
        inv_weather = {v: k for k, v in self.weather_to_id.items()}
        weather = inv_weather.get(weather_id, 'Unknown')
        return f"{tod},{weather}"

    # def label_2d_to_str(self, label_2d) -> str:
    #     """[tod_id, weather_id] -> 'Rain,Night/Day'"""
    #     tod_id = int(label_2d[0])
    #     weather_id = int(label_2d[1])

    #     tod = 'Day' if tod_id == 1 else ('Night' if tod_id == 0 else 'Unknown')
    #     inv_weather = {v: k for k, v in self.weather_to_id.items()}
    #     weather = inv_weather.get(weather_id, 'Unknown')
    #     return f"{tod},{weather}"

    # ----------------- CSV loader -----------------
    def _load_csv_as_label_maps(self, csv_path: str):
        """
        CSV columns:
          sequence_id,name,Scenario,weather,severity,Ground_Condition,Sky_Condition,Lens_Condition,Time_of_Day

        name pattern (guaranteed cam 02/03):
          {sequence_id}_image_{02|03}_{frame}
        frame expected 10 digits (we enforce zfill(10) anyway)
        """
        mapping_cls = {}
        mapping_2d = {}

        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                seq = row['sequence_id'].strip()
                name = row['name'].strip()

                weather = row['weather'].strip()
                tod = row['Time_of_Day'].strip()

                if '_image_02_' in name:
                    cam = 'image_02'
                elif '_image_03_' in name:
                    cam = 'image_03'
                else:
                    raise ValueError(f"CSV name cannot parse camera (expect image_02/03): {name}")

                # last token is frame (digits)
                frame_raw = name.split('_')[-1].strip()
                if not frame_raw.isdigit():
                    raise ValueError(f"CSV name cannot parse numeric frame: {name}")
                frame = frame_raw.zfill(10)  # enforce 10 digits

                if weather not in self.weather_to_id:
                    raise ValueError(
                        f"Unknown weather '{weather}' in {csv_path}. "
                        f"Expected one of {list(self.weather_to_id.keys())}"
                    )
                if tod not in self.tod_to_id:
                    raise ValueError(
                        f"Unknown Time_of_Day '{tod}' in {csv_path}. "
                        f"Expected one of {list(self.tod_to_id.keys())}"
                    )

                # 2D label
                tod_id = self.tod_to_id[tod]              # Night=0 Day=1
                weather_id = self.weather_to_id[weather]  # Clear=1 Rain=2 Snow=3 Fog=4
                mapping_2d[(seq, cam, frame)] = (tod_id, weather_id)

                # cls_id (0..7): weather_index*2 + tod_index
                weather_index = self._weather_list_for_cls.index(weather)  # 0..3
                tod_index = self._tod_list_for_cls.index(tod)              # 0..1
                cls_id = weather_index * 2 + tod_index
                mapping_cls[(seq, cam, frame)] = cls_id

        return mapping_cls, mapping_2d

    def _get_labels_for_current_sample(self, *, drive: str, cam: str, frame: str):
        """
        drive == sequence_id
        cam:   'image_02'/'image_03'
        frame: digits string (we will enforce zfill(10))
        returns: (cls_id, (tod_id, weather_id))
        """
        frame10 = frame.zfill(10)
        if (not self.sample_to_cls) and (not self.sample_to_2d):
            return -1, (-1, -1)

        key = (drive, cam, frame10)
        cls_id = self.sample_to_cls.get(key, -1)
        label_2d = self.sample_to_2d.get(key, (-1, -1))

        if cls_id < 0 and self._warn_missing_label_once:
            print(f"[KITTI] Warning: missing label for key={key}. "
                  f"Returning cls=-1 and label_2d=[-1,-1]. (This warning shows once)")
            self._warn_missing_label_once = False

        return cls_id, label_2d

    # ----------------- basic dataset -----------------
    def __len__(self):
        return len(self.depths)

    def __getitem__(self, index):
        return self.get_item(index)

    def get_item(self, index):
        depth = self.pull_DEPTH(self.depths[index])
        depth = np.expand_dims(depth, axis=2)
        lidar = self.pull_DEPTH(self.lidars[index])
        lidar = np.expand_dims(lidar, axis=2)
        K_cam = self.pull_K_cam(index).astype(np.float32)

        # ----------------- build RGB / DISP paths -----------------
        if self.mode == 'train':
            # depth path:
            # <base>/data_depth_annotated/train/<date_drive_sync>/proj_depth/groundtruth/image_0x/000000.png
            file_names = self.depths[index].split('/')
            fname = file_names[-1]           # 0000000005.png
            drive = file_names[-5]           # 2011_09_26_drive_0001_sync
            date = drive.split('_drive')[0]  # 2011_09_26
            cam = file_names[-2]             # image_02 or image_03

            # RGB from <base>/raw
            rgb_path = os.path.join(self.base_dir, 'raw', date, drive, cam, 'data', fname)

            # disp from sibling folder <parent_of_base>/kitti_depth_DC/raw
            parent_dir = os.path.dirname(self.base_dir)
            disp_path = os.path.join(parent_dir, 'kitti_depth_DC', 'raw', date, drive, cam, 'disp', fname)

            # labels from CSV
            frame = os.path.splitext(fname)[0]  # 0000000005
            cls_id, label_2d = self._get_labels_for_current_sample(drive=drive, cam=cam, frame=frame)

        elif self.mode in ['val', 'selval', 'test']:
            rgb_path = self.images[index]
            rgb_dir, rgb_name = os.path.split(rgb_path)
            disp_dir = rgb_dir.replace(os.sep + 'image', os.sep + 'disp')
            disp_path = os.path.join(disp_dir, rgb_name)

            if self.mode in ['val', 'selval']:
                # val_selection_cropped image filename:
                # 2011_09_26_drive_0002_sync_image_0000000009_image_02.png
                base = os.path.splitext(rgb_name)[0]
                m = re.match(r'^(?P<drive>.+)_image_(?P<frame>\d+)_image_(?P<cam>0[23])$', base)
                if m is None:
                    raise ValueError(f"Cannot parse val image name: {rgb_name}")
                drive = m.group('drive')
                frame = m.group('frame').zfill(10)
                cam = f"image_{m.group('cam')}"  # "02"/"03" -> "image_02"/"image_03"
                cls_id, label_2d = self._get_labels_for_current_sample(drive=drive, cam=cam, frame=frame)
            else:
                # test has no labels
                cls_id, label_2d = -1, (-1, -1)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        rgb = self.pull_RGB(rgb_path)
        disp = self.pull_DISP(disp_path)

        rgb = rgb.astype(np.float32)
        disp = disp.astype(np.float32)[:, :, None]
        lidar = lidar.astype(np.float32)
        depth = depth.astype(np.float32)

        if self.transform:
            rgb, disp, lidar, depth, K_cam = self.transform(rgb, disp, lidar, depth, K_cam)

        rgb = rgb.transpose(2, 0, 1).astype(np.float32)
        disp = disp.transpose(2, 0, 1).astype(np.float32)
        lidar = lidar.transpose(2, 0, 1).astype(np.float32)
        depth = depth.transpose(2, 0, 1).astype(np.float32)

        tp = rgb.shape[1] - self.height
        lp = (rgb.shape[2] - self.width) // 2
        if self.RandCrop:
            tp = random.randint(self.tp_min, tp)
            lp = random.randint(0, rgb.shape[2] - self.width)

        rgb = rgb[:, tp:tp + self.height, lp:lp + self.width]
        disp = disp[:, tp:tp + self.height, lp:lp + self.width]
        lidar = lidar[:, tp:tp + self.height, lp:lp + self.width]
        depth = depth[:, tp:tp + self.height, lp:lp + self.width]

        K_cam[0, 2] -= lp
        K_cam[1, 2] -= tp

        label_2d = torch.tensor(label_2d, dtype=torch.long)  # [tod_id, weather_id]
        cls = torch.tensor(cls_id, dtype=torch.long)         # 0..7

        # keep original returns + append labels at the end
        return rgb, disp, lidar, K_cam, depth, label_2d, cls

    # ----------------- DISP read (resize only in train) -----------------
    def pull_DISP(self, path):
        disp = np.array(Image.open(path).convert('L'), dtype=np.uint8)
        if self.mode == 'train':
            h, w = disp.shape[:2]
            if (w, h) != (TARGET_WIDTH, TARGET_HEIGHT):
                disp = cv2.resize(
                    disp,
                    (TARGET_WIDTH, TARGET_HEIGHT),
                    interpolation=cv2.INTER_LINEAR
                )
        return disp

    def pull_RGB(self, path):
        img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
        return img

    def pull_DEPTH(self, path):
        depth_png = np.array(Image.open(path), dtype=int)
        assert (np.max(depth_png) > 255)
        depth_image = (depth_png / 256.).astype(np.float32)
        return depth_image

    def pull_K_cam(self, index):
        """
        Unified from <base_dir>/raw/<date>/calib_cam_to_cam.txt
        """
        if self.mode == 'train':
            path = self.depths[index]
            parts = path.split('/')
            drive = parts[-5]           # 2011_09_26_drive_0001_sync
            date = drive.split('_drive')[0]
            cam_str = parts[-2]         # image_02 / image_03

        elif self.mode == 'val':
            # depth filename: 2011_09_26_drive_0002_sync_groundtruth_depth_0000000006_image_02.png
            fname = os.path.basename(self.depths[index])
            drive_full = fname.split('_groundtruth_depth')[0]     # 2011_09_26_drive_0002_sync
            date = drive_full.split('_drive')[0]                  # 2011_09_26
            cam_str = 'image_02' if 'image_02' in fname else 'image_03'

        elif self.mode in ['selval', 'test']:
            fname = os.path.basename(self.images[index])
            drive_full = fname.split('_image_')[0]                # 2011_09_26_drive_0002_sync
            date = drive_full.split('_drive')[0]
            cam_str = 'image_02' if 'image_02' in fname else 'image_03'

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        calib_path = os.path.join(self.base_dir, 'raw', date, 'calib_cam_to_cam.txt')
        filedata = read_calib_file(calib_path)
        P_rect_20 = np.reshape(filedata['P_rect_02'], (3, 4))
        P_rect_30 = np.reshape(filedata['P_rect_03'], (3, 4))

        if cam_str == 'image_02':
            K_cam = P_rect_20[0:3, 0:3]
        elif cam_str == 'image_03':
            K_cam = P_rect_30[0:3, 0:3]
        else:
            raise ValueError(f"Unknown camera string: {cam_str}")

        return K_cam
