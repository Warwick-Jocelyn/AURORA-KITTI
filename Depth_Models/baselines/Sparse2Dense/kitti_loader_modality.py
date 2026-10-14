"""
kitti_loader_modality.py
========================
Drop-in replacement for dataloaders/kitti_loader.py for the modality-ablation
runs (distill.py + main_modality.py).

Key differences from the original:
  1. Inlines BottomCrop + HorizontalFlip (avoids importing transforms.py which
     requires skimage — not available in the 'bp' conda env).
  2. Adds teacher (da3) loading with pixel-exact geometric alignment:
       - resize da3 to RGB/GT native size with INTER_NEAREST (cv2)
       - apply identical BottomCrop + HorizontalFlip as GT
  3. Teacher normalisation replicates DMD3C/datasets/kitti.py pull_DISP for da3
     (handled inside distill.py / load_da3_as_disp).
  4. Does NOT use use_pose / rgb_near / PnP logic — ablation is dense-supervised
     only (--train-mode dense).
  5. Supports cap_samples for smoke-test subset.

DO NOT EDIT kitti_loader.py or transforms.py.
"""

import os
import glob
import numpy as np
from PIL import Image
import cv2
import torch
import torch.utils.data as data

from distill import load_da3_as_disp

# DepthCompletionNet input options (must match model.py logic)
input_options = ['d', 'rgb', 'rgbd', 'g', 'gd']

DA3_ROOT = (
    "/path/to/aurora-root/PhD_Project_2025/"
    "DMD3C/BP-Net/DMD3C/datas/kitti_depth_DC/raw"
)

oheight, owidth = 352, 1216  # target crop size (matches kitti_loader.py)


# ---------------------------------------------------------------------------
# Calibration (verbatim from kitti_loader.py)
# ---------------------------------------------------------------------------

def load_calib():
    """Hardcoded calibration matrix from calib_cam_to_cam.txt (2011_09_26)."""
    calib = open("dataloaders/calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
    P_rect_line = lines[25]
    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    Proj = np.reshape(np.array([float(p) for p in Proj_str]), (3, 4)).astype(np.float32)
    K = Proj[:3, :3]
    K[0, 2] = K[0, 2] - 13
    K[1, 2] = K[1, 2] - 11.5
    return K


# ---------------------------------------------------------------------------
# Image readers (verbatim from kitti_loader.py)
# ---------------------------------------------------------------------------

def rgb_read(filename: str) -> np.ndarray:
    assert os.path.exists(filename), f"file not found: {filename}"
    img_file = Image.open(filename).convert("RGB")
    rgb_png = np.array(img_file, dtype="uint8")
    img_file.close()
    return rgb_png


def depth_read(filename: str) -> np.ndarray:
    """Load a 16-bit depth PNG and return float32 (H, W, 1) in metres."""
    assert os.path.exists(filename), f"file not found: {filename}"
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    assert np.max(depth_png) > 255, (
        f"np.max(depth_png)={np.max(depth_png)}, path={filename}"
    )
    depth = depth_png.astype(np.float32) / 256.0
    depth = np.expand_dims(depth, -1)
    return depth


# ---------------------------------------------------------------------------
# Inlined geometric transforms (avoids skimage dependency)
# ---------------------------------------------------------------------------

def _bottom_crop(img: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Bottom-centre crop to (th, tw)."""
    h, w = img.shape[0], img.shape[1]
    i = h - th
    j = int(round((w - tw) / 2.0))
    if img.ndim == 3:
        return img[i : i + th, j : j + tw, :]
    return img[i : i + th, j : j + tw]


def _hflip(img: np.ndarray, do_flip: bool) -> np.ndarray:
    if do_flip:
        return np.fliplr(img)
    return img


# ---------------------------------------------------------------------------
# Teacher path resolution
# ---------------------------------------------------------------------------

def gt_path_to_teacher_path(gt_path: str) -> str:
    """
    Map a GT depth path to the corresponding da3 teacher path.

    GT path pattern (AURORA train):
        .../data_depth_annotated/train/<drive_sync>/proj_depth/groundtruth/<cam>/<frame>

    Teacher path pattern:
        DA3_ROOT/<date>/<drive_sync>/<cam>/da3/<frame>

    where date = first three underscore-separated tokens of drive_sync
    (e.g. 2011_09_26_drive_0001_sync -> 2011_09_26).
    """
    frame      = os.path.basename(gt_path)
    cam        = os.path.basename(os.path.dirname(gt_path))
    # going up: <cam> / groundtruth / proj_depth / <drive_sync>
    drive_sync = os.path.basename(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(gt_path)))
        )
    )
    date = "_".join(drive_sync.split("_")[:3])
    return os.path.join(DA3_ROOT, date, drive_sync, cam, "da3", frame)


# ---------------------------------------------------------------------------
# Path discovery (mirrors kitti_loader.py logic)
# ---------------------------------------------------------------------------

def get_paths(split: str, args) -> dict:
    """Return dict with keys 'rgb', 'd', 'gt' (all lists of the same length)."""
    assert args.use_d or args.use_rgb, "no input modality selected"

    if split == "train":
        glob_d = os.path.join(
            args.data_folder,
            "data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png",
        )
        glob_gt = os.path.join(
            args.data_folder,
            "data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png",
        )

        def get_rgb_path(p):
            ps = p.split("/")
            # ps[-6:-4] = [<drive_sync>, 'proj_depth'] — but we derive from gt
            # gt: .../data_depth_annotated/train/<drive>_sync/proj_depth/groundtruth/<cam>/<frame>
            # rgb: .../data_rgb/train/<drive>_sync/<cam>/data/<frame>
            # Replicating kitti_loader.py:
            #   ps[-6:-4] + ps[-2:-1] + ['data'] + ps[-1:]
            pnew = "/".join(
                [args.data_folder] + ["data_rgb"] + ps[-6:-4] + ps[-2:-1] + ["data"] + ps[-1:]
            )
            return pnew

    elif split == "val":
        if args.val == "full":
            glob_d = os.path.join(
                args.data_folder,
                "data_depth_velodyne/val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png",
            )
            glob_gt = os.path.join(
                args.data_folder,
                "data_depth_annotated/val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png",
            )

            def get_rgb_path(p):
                ps = p.split("/")
                pnew = "/".join(
                    ps[:-7] + ["data_rgb"] + ps[-6:-4] + ps[-2:-1] + ["data"] + ps[-1:]
                )
                return pnew

        elif args.val == "select":
            glob_d  = os.path.join(
                args.data_folder,
                "depth_selection/val_selection_cropped/velodyne_raw/*.png",
            )
            glob_gt = os.path.join(
                args.data_folder,
                "depth_selection/val_selection_cropped/groundtruth_depth/*.png",
            )

            def get_rgb_path(p):
                return p.replace("groundtruth_depth", "image")

        else:
            raise ValueError(f"Unknown val mode: {args.val}")
    else:
        raise ValueError(f"Unsupported split for modality ablation: {split}")

    paths_d  = sorted(glob.glob(glob_d))
    paths_gt = sorted(glob.glob(glob_gt))
    paths_rgb = [get_rgb_path(p) for p in paths_gt]

    if len(paths_gt) == 0:
        raise RuntimeError(f"Found 0 ground-truth images under {glob_gt}")
    if len(paths_d) != len(paths_gt):
        raise RuntimeError(
            f"Sparse depth count ({len(paths_d)}) != GT count ({len(paths_gt)})"
        )

    return {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}


# ---------------------------------------------------------------------------
# ToTensor (matches original to_float_tensor = ToTensor().float())
# ---------------------------------------------------------------------------

def _to_float_tensor(arr: np.ndarray) -> torch.Tensor:
    """
    Convert numpy (H,W,C) or (H,W) to float32 torch tensor (C,H,W) or (1,H,W).
    Mirrors the original:  to_tensor = transforms.ToTensor()
                           to_float_tensor = lambda x: to_tensor(x).float()
    """
    if arr.ndim == 3:
        t = torch.from_numpy(arr.transpose((2, 0, 1)).copy())
    elif arr.ndim == 2:
        t = torch.from_numpy(arr.copy())
    else:
        raise RuntimeError(f"Expected 2D or 3D array, got {arr.ndim}D")
    return t.float()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class KittiDepthModality(data.Dataset):
    """
    KITTI depth completion dataset for modality ablation.
    Supports --input rgbd | rgb | d.
    Adds teacher (da3) with pixel-exact geometric augmentation for train split.
    """

    def __init__(self, split: str, args, cap_samples: int = 0):
        """
        Args:
            split:       'train' or 'val'
            args:        parsed argparse namespace
            cap_samples: if >0, truncate dataset to this many samples (smoke test)
        """
        self.split       = split
        self.args        = args
        self.cap_samples = cap_samples

        paths = get_paths(split, args)
        self.paths_rgb = paths["rgb"]
        self.paths_d   = paths["d"]
        self.paths_gt  = paths["gt"]

        if cap_samples > 0:
            self.paths_rgb = self.paths_rgb[:cap_samples]
            self.paths_d   = self.paths_d[:cap_samples]
            self.paths_gt  = self.paths_gt[:cap_samples]

        print(
            f"[KittiDepthModality] split={split}, samples={len(self.paths_gt)}, "
            f"input={args.input}, cap={cap_samples}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_raw(self, index):
        """Load raw arrays (no augmentation). Returns (rgb, sparse, target, da3_path)."""
        rgb    = rgb_read(self.paths_rgb[index]) if self.args.use_rgb else None
        sparse = depth_read(self.paths_d[index]) if self.args.use_d   else None
        target = depth_read(self.paths_gt[index])

        da3_path = None
        if self.split == "train":
            tp = gt_path_to_teacher_path(self.paths_gt[index])
            if os.path.exists(tp):
                da3_path = tp
            else:
                # Should not happen given verified 100% coverage, but degrade
                # gracefully rather than crash.
                print(f"[WARN] teacher not found: {tp}")

        return rgb, sparse, target, da3_path

    def _apply_train_transform(self, rgb, sparse, target, da3_disp):
        """
        Apply shared geometric augmentation (BottomCrop + random HFlip).
        Color jitter on RGB only; never on sparse/target/teacher.

        da3_disp: np.ndarray (H_orig, W_orig) float32 — already loaded.
            Resized to (H_gt, W_gt) with INTER_NEAREST before shared crop.
        """
        do_flip = np.random.uniform(0.0, 1.0) < 0.5

        # --- resize teacher to match GT native size (INTER_NEAREST) ---
        if da3_disp is not None:
            gt_h, gt_w = target.shape[0], target.shape[1]
            if da3_disp.shape[0] != gt_h or da3_disp.shape[1] != gt_w:
                da3_disp = cv2.resize(
                    da3_disp, (gt_w, gt_h), interpolation=cv2.INTER_NEAREST
                )

        # --- shared geometric transform (identical params for all arrays) ---
        if sparse is not None:
            sparse = _bottom_crop(sparse, oheight, owidth)
            sparse = _hflip(sparse, do_flip)
        target = _bottom_crop(target, oheight, owidth)
        target = _hflip(target, do_flip)
        if da3_disp is not None:
            da3_disp = _bottom_crop(da3_disp, oheight, owidth)
            da3_disp = _hflip(da3_disp, do_flip)

        # --- color jitter on RGB only ---
        if rgb is not None:
            jitter = self.args.jitter
            brightness  = np.random.uniform(max(0, 1 - jitter), 1 + jitter)
            contrast    = np.random.uniform(max(0, 1 - jitter), 1 + jitter)
            saturation  = np.random.uniform(max(0, 1 - jitter), 1 + jitter)

            pil = Image.fromarray(rgb)
            from PIL import ImageEnhance
            pil = ImageEnhance.Brightness(pil).enhance(brightness)
            pil = ImageEnhance.Contrast(pil).enhance(contrast)
            pil = ImageEnhance.Color(pil).enhance(saturation)
            rgb = np.array(pil, dtype=np.uint8)

            # geometric after color jitter
            rgb = _bottom_crop(rgb, oheight, owidth)
            rgb = _hflip(rgb, do_flip)

        return rgb, sparse, target, da3_disp

    def _apply_val_transform(self, rgb, sparse, target):
        """Val: BottomCrop only (no flip, no jitter).
        val_selection_cropped images are already 352x1216 — crop is a no-op."""
        if sparse is not None:
            sparse = _bottom_crop(sparse, oheight, owidth)
        target = _bottom_crop(target, oheight, owidth)
        if rgb is not None:
            rgb = _bottom_crop(rgb, oheight, owidth)
        return rgb, sparse, target

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.paths_gt)

    def __getitem__(self, index):
        try:
            return self._get_item_inner(index)
        except Exception as e:
            print(f"[WARN] Skipping index {index} due to: {e}")
            return self.__getitem__((index + 1) % len(self))

    def _get_item_inner(self, index):
        rgb, sparse, target, da3_path = self._load_raw(index)

        # --- load da3 disp (float32 H×W, no channel dim yet) ---
        da3_disp = None
        if da3_path is not None:
            da3_disp = load_da3_as_disp(da3_path)  # (H, W) float32

        if self.split == "train":
            rgb, sparse, target, da3_disp = self._apply_train_transform(
                rgb, sparse, target, da3_disp
            )
        else:
            rgb, sparse, target = self._apply_val_transform(rgb, sparse, target)
            # no teacher at val time

        # --- build output dict ---
        candidates = {}
        if rgb is not None:
            candidates["rgb"] = rgb                              # uint8 (H,W,3)
        if sparse is not None:
            candidates["d"] = sparse                            # float32 (H,W,1)
        candidates["gt"] = target                               # float32 (H,W,1)
        if da3_disp is not None:
            # add channel dim so it stacks the same way as depth maps
            candidates["teacher_disp"] = da3_disp[..., np.newaxis]  # (H,W,1)

        items = {k: _to_float_tensor(v) for k, v in candidates.items()}
        return items
