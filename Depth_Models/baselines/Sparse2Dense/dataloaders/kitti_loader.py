import os
import os.path
import glob
import fnmatch  # pattern matching
import numpy as np
from numpy import linalg as LA
from random import choice
from PIL import Image
import torch
import torch.utils.data as data
import cv2
from dataloaders import transforms
from dataloaders.pose_estimator import get_pose_pnp

input_options = ['d', 'rgb', 'rgbd', 'g', 'gd']


def load_calib():
    """
    Temporarily hardcoding the calibration matrix using calib file from 2011_09_26
    """
    calib = open("dataloaders/calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
    P_rect_line = lines[25]

    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                      (3, 4)).astype(np.float32)
    K = Proj[:3, :3]  # camera matrix

    # note: we will take the center crop of the images during augmentation
    # that changes the optical centers, but not focal lengths
    K[0, 2] = K[
        0,
        2] - 13  # from width = 1242 to 1216, with a 13-pixel cut on both sides
    K[1, 2] = K[
        1,
        2] - 11.5  # from width = 375 to 352, with a 11.5-pixel cut on both sides
    return K


def get_paths_and_transform(split, args):
    assert (args.use_d or args.use_rgb
            or args.use_g), 'no proper input selected'

    if split == "train":
        transform = train_transform
        glob_d = os.path.join(
            args.data_folder,
            'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
        )
        glob_gt = os.path.join(
            args.data_folder,
            'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
        )

        def get_rgb_paths(p):
            ps = p.split('/')
            pnew = '/'.join([args.data_folder] + ['data_rgb'] + ps[-6:-4] +
                            ps[-2:-1] + ['data'] + ps[-1:])
            return pnew
    elif split == "val":
        if args.val == "full":
            transform = val_transform
            glob_d = os.path.join(
                args.data_folder,
                'data_depth_velodyne/val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
            )
            glob_gt = os.path.join(
                args.data_folder,
                'data_depth_annotated/val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            )
            def get_rgb_paths(p):
                ps = p.split('/')
                pnew = '/'.join(ps[:-7] +  
                    ['data_rgb']+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
                return pnew
        elif args.val == "select":
            transform = no_transform
            glob_d = os.path.join(
                args.data_folder,
                "depth_selection/val_selection_cropped/velodyne_raw/*.png")
            glob_gt = os.path.join(
                args.data_folder,
                "depth_selection/val_selection_cropped/groundtruth_depth/*.png"
            )
            def get_rgb_paths(p):
                return p.replace("groundtruth_depth","image")
    elif split == "test_completion":
        transform = no_transform
        glob_d = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png"
        )
        glob_gt = None  #"test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_completion_anonymous/image/*.png")
    elif split == "test_prediction":
        transform = no_transform
        glob_d = None
        glob_gt = None  #"test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_prediction_anonymous/image/*.png")
    else:
        raise ValueError("Unrecognized split " + str(split))

    if glob_gt is not None:
        # train or val-full or val-select
        paths_d = sorted(glob.glob(glob_d)) 
        paths_gt = sorted(glob.glob(glob_gt)) 
        paths_rgb = [get_rgb_paths(p) for p in paths_gt]
    else:  
        # test only has d or rgb
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_gt = [None] * len(paths_rgb)
        if split == "test_prediction":
            paths_d = [None] * len(
                paths_rgb)  # test_prediction has no sparse depth
        else:
            paths_d = sorted(glob.glob(glob_d))

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_d) == 0 and args.use_d:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0 and args.use_rgb:
        raise (RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) == 0 and args.use_g:
        raise (RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        raise (RuntimeError("Produced different sizes for datasets"))

    paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    return paths, transform


def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename).convert('RGB')  # 强制去掉 alpha / 调色板
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth


oheight, owidth = 352, 1216


def drop_depth_measurements(depth, prob_keep):
    mask = np.random.binomial(1, prob_keep, depth.shape)
    depth *= mask
    return depth


def train_transform(rgb, sparse, target, rgb_near, args):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

    transform_geometric = transforms.Compose([
        # transforms.Rotate(angle),
        # transforms.Resize(s),
        transforms.BottomCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ])
    if sparse is not None:
        sparse = transform_geometric(sparse)
    target = transform_geometric(target)
    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)
        contrast = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        saturation = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)
        transform_rgb = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, 0),
            transform_geometric
        ])
        rgb = transform_rgb(rgb)
        if rgb_near is not None:
            rgb_near = transform_rgb(rgb_near)
    # sparse = drop_depth_measurements(sparse, 0.9)

    return rgb, sparse, target, rgb_near


def val_transform(rgb, sparse, target, rgb_near, args):
    transform = transforms.Compose([
        transforms.BottomCrop((oheight, owidth)),
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)
    if rgb_near is not None:
        rgb_near = transform(rgb_near)
    return rgb, sparse, target, rgb_near


def no_transform(rgb, sparse, target, rgb_near, args):
    return rgb, sparse, target, rgb_near


to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()


def handle_gray(rgb, args):
    if rgb is None:
        return None, None
    if not args.use_g:
        return rgb, None
    else:
        img = np.array(Image.fromarray(rgb).convert('L'))
        img = np.expand_dims(img, -1)
        if not args.use_rgb:
            rgb_ret = None
        else:
            rgb_ret = rgb
        return rgb_ret, img

def get_rgb_near(path, args):
    """获取附近帧，如果找不到，直接返回原始帧"""
    assert path is not None, "path is None"

    def extract_frame_id(filename):
        head, tail = os.path.split(filename)
        number_string = tail[0:tail.find('.')]
        number = int(number_string)
        return head, number

    def get_nearby_filename(filename, new_id):
        head, _ = os.path.split(filename)
        return os.path.join(head, '%010d.png' % new_id)

    head, number = extract_frame_id(path)
    max_frame_diff = 3
    candidates = [i for i in range(-max_frame_diff, max_frame_diff + 1) if i != 0]

    # 尝试寻找 nearby frame
    for _ in range(20):
        random_offset = choice(candidates)
        path_near = get_nearby_filename(path, number + random_offset)
        if os.path.exists(path_near):
            return rgb_read(path_near)

    # 如果找不到附近帧，直接返回原始帧
    return rgb_read(path)

class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset with filtering for valid nearby frames"""
    def __init__(self, split, args):
        self.args = args
        self.split = split
        paths, transform = get_paths_and_transform(split, args)
        self.paths = paths
        self.transform = transform
        self.K = load_calib()
        self.threshold_translation = 0.1

        # ---------------------------
        # 仅保留可以找到 nearby frame 的训练样本
        # ---------------------------
        if self.split == 'train' and self.args.use_pose:
            self.valid_indices = []
            max_frame_diff = 3  # 与原 get_rgb_near 保持一致
            for i, rgb_path in enumerate(self.paths['rgb']):
                number = int(os.path.basename(rgb_path).split('.')[0])
                head = os.path.dirname(rgb_path)
                found_nearby = False
                for offset in range(-max_frame_diff, max_frame_diff + 1):
                    if offset == 0:
                        continue
                    nearby_file = os.path.join(head, f"{number + offset:010d}.png")
                    if os.path.exists(nearby_file):
                        found_nearby = True
                        break
                if found_nearby:
                    self.valid_indices.append(i)
            print(f"[INFO] Training samples with valid nearby frame: {len(self.valid_indices)}")
        else:
            # val/test 或不使用 pose，全部样本有效
            self.valid_indices = list(range(len(self.paths['rgb'])))

    def __getraw__(self, index):
        rgb, sparse, target = None, None, None
        real_index = self.valid_indices[index]  # 使用过滤后的索引
        if self.paths['rgb'][real_index] is not None and (self.args.use_rgb or self.args.use_g):
            rgb = rgb_read(self.paths['rgb'][real_index])
        if self.paths['d'][real_index] is not None and self.args.use_d:
            sparse = depth_read(self.paths['d'][real_index])
        if self.paths['gt'][real_index] is not None:
            target = depth_read(self.paths['gt'][real_index])
        rgb_near = get_rgb_near(self.paths['rgb'][real_index], self.args) if \
            (self.split == 'train' and self.args.use_pose) else None
        return rgb, sparse, target, rgb_near

    def __getitem__(self, index):
        try:
            rgb, sparse, target, rgb_near = self.__getraw__(index)
            rgb, sparse, target, rgb_near = self.transform(rgb, sparse, target, rgb_near, self.args)
            r_mat, t_vec = None, None
            if self.split == 'train' and self.args.use_pose:
                success, r_vec, t_vec = get_pose_pnp(rgb, rgb_near, sparse, self.K)
                success = success and LA.norm(t_vec) > self.threshold_translation
                if success:
                    r_mat, _ = cv2.Rodrigues(r_vec)
                else:
                    rgb_near = rgb
                    t_vec = np.zeros((3, 1))
                    r_mat = np.eye(3)
            rgb, gray = handle_gray(rgb, self.args)
            candidates = {"rgb": rgb, "d": sparse, "gt": target,
                        "g": gray, "r_mat": r_mat, "t_vec": t_vec, "rgb_near": rgb_near}
            items = {key: to_float_tensor(val) for key, val in candidates.items() if val is not None}
            return items
        except Exception as e:
            print(f"[WARN] Skipping sample index {index} due to {e}")
            return self.__getitem__((index + 1) % len(self))

    def __len__(self):
        return len(self.valid_indices)  # 返回有效样本数
