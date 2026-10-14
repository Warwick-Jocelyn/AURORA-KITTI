import random
import os
import numpy as np
import glob
from PIL import Image
import torch
import augs
import cv2  # 新增
from torchvision.utils import save_image

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
    """

    def __init__(self, path='datas/kitti', mode='train',
                 height=256, width=1216,
                 mean=(90.9950, 96.2278, 94.3213),
                 std=(79.2382, 80.5267, 82.1483),
                 RandCrop=False, tp_min=50, *args, **kwargs):

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

        # ----------------- 文件列表 -----------------
        if mode == 'train':
            # 大规模 train raw data
            self.depth_path = os.path.join(self.base_dir, 'data_depth_annotated', 'train')
            self.lidar_path = os.path.join(self.base_dir, 'data_depth_velodyne', 'train')
            self.depths = list(sorted(glob.iglob(self.depth_path + "/**/*.png", recursive=True)))
            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/**/*.png", recursive=True)))
            self.images = None  # train 不直接用这个

        elif mode in ['val', 'selval']:
            # val_selection_cropped
            self.base_val_dir = os.path.join(self.base_dir, 'val_selection_cropped')
            self.lidar_path = os.path.join(self.base_val_dir, 'velodyne_raw')
            self.depth_path = os.path.join(self.base_val_dir, 'groundtruth_depth')
            self.image_path = os.path.join(self.base_val_dir, 'image')

            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/*.png", recursive=True)))
            # 用 lidar 名称构造其余两个，确保对齐
            self.depths = [x.replace('velodyne_raw', 'groundtruth_depth').replace(
                '_velodyne_raw_', '_groundtruth_depth_') for x in self.lidars]
            self.images = [x.replace('velodyne_raw', 'image').replace(
                '_velodyne_raw_', '_image_') for x in self.lidars]

        elif mode == 'test':
            self.lidar_path = os.path.join(self.base_dir, 'test_depth_completion_anonymous', 'velodyne_raw')
            self.image_path = os.path.join(self.base_dir, 'test_depth_completion_anonymous', 'image')
            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/*.png", recursive=True)))
            self.images = [x.replace('velodyne_raw', 'image') for x in self.lidars]
            # test 没有 GT，用 lidar 占位
            self.depths = self.lidars

        else:
            raise ValueError(f"Unknown mode: {mode}")

        assert len(self.depths) == len(self.lidars)
        self.names = [os.path.split(path)[-1] for path in self.depths]

        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        xy = np.stack((xx, yy), axis=-1)
        self.xy = xy

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

        # ----------------- 构造 RGB / DISP 路径 -----------------
        if self.mode == 'train':
            # depth path: <base>/data_depth_annotated/train/<date_drive_sync>/proj_depth/groundtruth/image_0x/000000.png
            file_names = self.depths[index].split('/')
            fname = file_names[-1]           # 0000000005.png
            drive = file_names[-5]           # 2011_09_26_drive_0001_sync
            date = drive.split('_drive')[0]  # 2011_09_26
            cam = file_names[-2]             # image_02 or image_03

            # RGB from <base>/raw
            rgb_path = os.path.join(self.base_dir, 'raw', date, drive, cam, 'data', fname)

            # Try1: Using clean guided disp maps: disp from sibling folder <parent_of_base>/kitti_depth_DC/raw
            parent_dir = os.path.dirname(self.base_dir)
            disp_path = os.path.join(
                parent_dir, 'kitti_depth_DC', 'raw', date, drive, cam, 'da3', fname
            )

            # # Try2: Using the noisy guided disp maps: disp from original folder <parent_of_base>/kitti/raw
            # parent_dir = os.path.dirname(self.base_dir)
            # disp_path = os.path.join(
            #     parent_dir, 'kitti', 'raw', date, drive, cam, 'disp', fname
            # )

        elif self.mode in ['val', 'selval', 'test']:
            # 你已经把 val_selection_cropped resize+crop 好了，这里直接用
            rgb_path = self.images[index]
            # /image/ → /disp/，文件名不变
            rgb_dir, rgb_name = os.path.split(rgb_path)
            disp_dir = rgb_dir.replace(os.sep + 'image', os.sep + 'da3')
            disp_path = os.path.join(disp_dir, rgb_name)
            if not os.path.exists(disp_path):  # fallback for datasets without da3 (e.g. DENSE uses disp/)
                disp_dir = rgb_dir.replace(os.sep + 'image', os.sep + 'disp')
                disp_path = os.path.join(disp_dir, rgb_name)
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

        return rgb, disp, lidar, K_cam, depth

    # ----------------- DISP 读取（只在 train 里 resize） -----------------
    # def pull_DISP(self, path):
    #     disp = np.array(Image.open(path).convert('L'), dtype=np.uint8)
    #     if self.mode == 'train':
    #         h, w = disp.shape[:2]
    #         if (w, h) != (TARGET_WIDTH, TARGET_HEIGHT):
    #             disp = cv2.resize(
    #                 disp,
    #                 (TARGET_WIDTH, TARGET_HEIGHT),
    #                 interpolation=cv2.INTER_LINEAR
    #             )
    #     return disp

    def pull_DISP(self, path):
        disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if disp is None:
            raise FileNotFoundError(path)

        # 兼容旧 disp 三通道
        if disp.ndim == 3:
            disp = disp[:, :, 0]

        if self.mode == 'train':
            h, w = disp.shape[:2]
            if (w, h) != (TARGET_WIDTH, TARGET_HEIGHT):
                disp = cv2.resize(disp, (TARGET_WIDTH, TARGET_HEIGHT),
                                interpolation=cv2.INTER_LINEAR)

        disp = disp.astype(np.float32)

        # 如果是 uint16（>255），先还原到“米”
        if disp.max() > 255.0:
            depth_m = disp / 256.0   # 你当前 da3 的保存方式看起来是 *256
            # 再映射回 0~255，保持旧输入分布（80m 是你可视化也在用的典型上限）
            depth_m = np.clip(depth_m, 0.0, 81.0)
            disp = depth_m / 81.0 * 255.0

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
        统一从 <base_dir>/raw/<date>/calib_cam_to_cam.txt 读取 K_cam
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
            # val_selection_cropped 的文件名格式为 scene1_day_clear_0.png，
            # 对应内参文件在 val_selection_cropped/intrinsics/<stem>.txt
            fname = os.path.basename(self.images[index])
            stem = os.path.splitext(fname)[0]
            intrinsics_path = os.path.join(
                self.base_dir, 'val_selection_cropped', 'intrinsics', stem + '.txt')
            K_cam = np.loadtxt(intrinsics_path).astype(np.float32).reshape(3, 3)
            return K_cam
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
