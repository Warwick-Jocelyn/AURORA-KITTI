# -*- coding: utf-8 -*-
import random
import os
import numpy as np
import glob
from PIL import Image
import torch
import augs
import cv2

__all__ = ['KITTI']

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
    returns:
      train: (rgb, disp_clean, disp_noisy, lidar, K_cam, depth)
      val/test: (rgb, disp_clean, lidar, K_cam, depth)
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

        # ----------------- file list -----------------
        if mode == 'train':
            self.depth_path = os.path.join(self.base_dir, 'data_depth_annotated', 'train')
            self.lidar_path = os.path.join(self.base_dir, 'data_depth_velodyne', 'train')
            self.depths = list(sorted(glob.iglob(self.depth_path + "/**/*.png", recursive=True)))
            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/**/*.png", recursive=True)))
            self.images = None

        elif mode in ['val', 'selval']:
            self.base_val_dir = os.path.join(self.base_dir, 'val_selection_cropped')
            self.lidar_path = os.path.join(self.base_val_dir, 'velodyne_raw')
            self.depth_path = os.path.join(self.base_val_dir, 'groundtruth_depth')
            self.image_path = os.path.join(self.base_val_dir, 'image')

            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/*.png", recursive=True)))
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

        # ----------------- build RGB & DISP paths -----------------
        if self.mode == 'train':
            parts = self.depths[index].split('/')
            fname = parts[-1]
            drive = parts[-5]
            date = drive.split('_drive')[0]
            cam = parts[-2]  # image_02 / image_03

            rgb_path = os.path.join(self.base_dir, 'raw', date, drive, cam, 'data', fname)

            parent_dir = os.path.dirname(self.base_dir)

            # clean disp
            disp_clean_path = os.path.join(parent_dir, 'kitti_depth_DC', 'raw', date, drive, cam, 'disp', fname)
            # noisy disp
            disp_noisy_path = os.path.join(parent_dir, 'kitti', 'raw', date, drive, cam, 'disp', fname)

        elif self.mode in ['val', 'selval', 'test']:
            rgb_path = self.images[index]
            rgb_dir, rgb_name = os.path.split(rgb_path)
            disp_dir = rgb_dir.replace(os.sep + 'image', os.sep + 'disp')
            disp_clean_path = os.path.join(disp_dir, rgb_name)
            disp_noisy_path = None
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        rgb = self.pull_RGB(rgb_path)
        disp_clean = self.pull_DISP(disp_clean_path)

        if self.mode == 'train':
            disp_noisy = self.pull_DISP(disp_noisy_path)
        else:
            disp_noisy = None

        rgb = rgb.astype(np.float32)
        disp_clean = disp_clean.astype(np.float32)[:, :, None]
        lidar = lidar.astype(np.float32)
        depth = depth.astype(np.float32)

        if self.mode == 'train':
            disp_noisy = disp_noisy.astype(np.float32)[:, :, None]

            # stack as (H,W,2) to guarantee same flip/crop
            disp2 = np.concatenate([disp_clean, disp_noisy], axis=2)

            if self.transform:
                try:
                    rgb, disp2, lidar, depth, K_cam = self.transform(rgb, disp2, lidar, depth, K_cam)
                except Exception as e:
                    # Fallback: if augs assumes single-channel disp (rare), apply transform on clean only
                    # and then apply the same crop later. (Flip randomness may differ if Flip uses RNG inside.)
                    # Best is to update augs to be channel-agnostic; this fallback keeps training runnable.
                    rgb, disp_clean, lidar, depth, K_cam = self.transform(rgb, disp_clean, lidar, depth, K_cam)
                    disp2 = np.concatenate([disp_clean, disp_noisy], axis=2)

            disp_clean = disp2[:, :, 0:1]
            disp_noisy = disp2[:, :, 1:2]

        else:
            if self.transform:
                rgb, disp_clean, lidar, depth, K_cam = self.transform(rgb, disp_clean, lidar, depth, K_cam)

        # to CHW
        rgb = rgb.transpose(2, 0, 1).astype(np.float32)
        disp_clean = disp_clean.transpose(2, 0, 1).astype(np.float32)
        lidar = lidar.transpose(2, 0, 1).astype(np.float32)
        depth = depth.transpose(2, 0, 1).astype(np.float32)
        if disp_noisy is not None:
            disp_noisy = disp_noisy.transpose(2, 0, 1).astype(np.float32)

        # crop
        tp = rgb.shape[1] - self.height
        lp = (rgb.shape[2] - self.width) // 2
        if self.RandCrop:
            tp = random.randint(self.tp_min, tp)
            lp = random.randint(0, rgb.shape[2] - self.width)

        rgb = rgb[:, tp:tp + self.height, lp:lp + self.width]
        disp_clean = disp_clean[:, tp:tp + self.height, lp:lp + self.width]
        lidar = lidar[:, tp:tp + self.height, lp:lp + self.width]
        depth = depth[:, tp:tp + self.height, lp:lp + self.width]
        if disp_noisy is not None:
            disp_noisy = disp_noisy[:, tp:tp + self.height, lp:lp + self.width]

        K_cam[0, 2] -= lp
        K_cam[1, 2] -= tp

        if self.mode == 'train':
            return rgb, disp_clean, disp_noisy, lidar, K_cam, depth
        else:
            return rgb, disp_clean, lidar, K_cam, depth

    # ----------------- DISP reader -----------------
    def pull_DISP(self, path):
        disp = np.array(Image.open(path).convert('L'), dtype=np.uint8)
        if self.mode == 'train':
            h, w = disp.shape[:2]
            if (w, h) != (TARGET_WIDTH, TARGET_HEIGHT):
                disp = cv2.resize(disp, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
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
        if self.mode == 'train':
            path = self.depths[index]
            parts = path.split('/')
            drive = parts[-5]
            date = drive.split('_drive')[0]
            cam_str = parts[-2]
        elif self.mode == 'val':
            fname = os.path.basename(self.depths[index])
            drive_full = fname.split('_groundtruth_depth')[0]
            date = drive_full.split('_drive')[0]
            cam_str = 'image_02' if 'image_02' in fname else 'image_03'
        elif self.mode in ['selval', 'test']:
            fname = os.path.basename(self.images[index])
            drive_full = fname.split('_image_')[0]
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
