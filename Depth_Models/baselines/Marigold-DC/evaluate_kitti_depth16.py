# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import argparse
# import os
# from glob import glob
# import math
# import numpy as np
# from PIL import Image
# import torch

# # ----------------------------
# # Metrics (same as your snippet)
# # ----------------------------
# lg_e_10 = math.log(10)

# def log10(x):
#     return torch.log(x) / lg_e_10

# class Result(object):
#     def __init__(self):
#         self.irmse = 0
#         self.imae = 0
#         self.mse = 0
#         self.rmse = 0
#         self.mae = 0
#         self.absrel = 0
#         self.squared_rel = 0
#         self.lg10 = 0
#         self.delta1 = 0
#         self.delta2 = 0
#         self.delta3 = 0
#         self.data_time = 0
#         self.gpu_time = 0
#         self.silog = 0
#         self.photometric = 0

#     def set_to_worst(self):
#         self.irmse = np.inf
#         self.imae = np.inf
#         self.mse = np.inf
#         self.rmse = np.inf
#         self.mae = np.inf
#         self.absrel = np.inf
#         self.squared_rel = np.inf
#         self.lg10 = np.inf
#         self.silog = np.inf
#         self.delta1 = 0
#         self.delta2 = 0
#         self.delta3 = 0
#         self.data_time = 0
#         self.gpu_time = 0

#     def update(self, irmse, imae, mse, rmse, mae, absrel, squared_rel, lg10,
#                delta1, delta2, delta3, gpu_time, data_time, silog, photometric=0):
#         self.irmse = irmse
#         self.imae = imae
#         self.mse = mse
#         self.rmse = rmse
#         self.mae = mae
#         self.absrel = absrel
#         self.squared_rel = squared_rel
#         self.lg10 = lg10
#         self.delta1 = delta1
#         self.delta2 = delta2
#         self.delta3 = delta3
#         self.data_time = data_time
#         self.gpu_time = gpu_time
#         self.silog = silog
#         self.photometric = photometric

#     def evaluate(self, output, target, photometric=0):
#         epsilon = 1e-6
#         valid_mask = target > 0.1  # meters

#         if valid_mask.sum() == 0:
#             self.irmse = float('nan')
#             self.imae = float('nan')
#             self.rmse = float('nan')
#             self.mae = float('nan')
#             self.silog = float('nan')
#             self.absrel = float('nan')
#             self.squared_rel = float('nan')
#             self.lg10 = float('nan')
#             self.delta1 = 0
#             self.delta2 = 0
#             self.delta3 = 0
#             self.photometric = float(photometric)
#             return

#         output_valid = output[valid_mask]
#         target_valid = target[valid_mask]

#         abs_diff = (output_valid - target_valid).abs()
#         self.mse = float((abs_diff ** 2).mean())
#         self.rmse = float(torch.sqrt(torch.tensor(self.mse)))
#         self.mae = float(abs_diff.mean())
#         self.lg10 = float((log10(output_valid + epsilon) - log10(target_valid + epsilon)).abs().mean())
#         self.absrel = float((abs_diff / target_valid).mean())
#         self.squared_rel = float(((abs_diff / target_valid) ** 2).mean())

#         maxRatio = torch.max(output_valid / target_valid, target_valid / output_valid)
#         self.delta1 = float((maxRatio < 1.25).float().mean())
#         self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
#         self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())

#         err_log = torch.log(target_valid + epsilon) - torch.log(output_valid + epsilon)
#         normalized_squared_log = (err_log ** 2).mean()
#         log_mean = err_log.mean()
#         self.silog = math.sqrt(normalized_squared_log - log_mean * log_mean) * 100

#         inv_mask = output_valid > 0
#         if inv_mask.sum() == 0:
#             self.irmse = float('nan')
#             self.imae = float('nan')
#         else:
#             inv_output = 1.0 / (output_valid[inv_mask] * 1e-3)
#             inv_target = 1.0 / (target_valid[inv_mask] * 1e-3)
#             abs_inv_diff = (inv_output - inv_target).abs()
#             self.irmse = float(torch.sqrt((abs_inv_diff ** 2).mean()))
#             self.imae = float(abs_inv_diff.mean())

#         self.data_time = 0
#         self.gpu_time = 0
#         self.photometric = float(photometric)

# class AverageMeter(object):
#     def __init__(self):
#         self.reset()
#     def reset(self):
#         self.count = 0.0
#         self.sum_irmse = 0
#         self.sum_imae = 0
#         self.sum_mse = 0
#         self.sum_rmse = 0
#         self.sum_mae = 0
#         self.sum_absrel = 0
#         self.sum_squared_rel = 0
#         self.sum_lg10 = 0
#         self.sum_delta1 = 0
#         self.sum_delta2 = 0
#         self.sum_delta3 = 0
#         self.sum_data_time = 0
#         self.sum_gpu_time = 0
#         self.sum_photometric = 0
#         self.sum_silog = 0
#     def update(self, result, gpu_time=0.0, data_time=0.0, n=1):
#         self.count += n
#         self.sum_irmse += n * result.irmse
#         self.sum_imae += n * result.imae
#         self.sum_mse += n * result.mse
#         self.sum_rmse += n * result.rmse
#         self.sum_mae += n * result.mae
#         self.sum_absrel += n * result.absrel
#         self.sum_squared_rel += n * result.squared_rel
#         self.sum_lg10 += n * result.lg10
#         self.sum_delta1 += n * result.delta1
#         self.sum_delta2 += n * result.delta2
#         self.sum_delta3 += n * result.delta3
#         self.sum_data_time += n * data_time
#         self.sum_gpu_time += n * gpu_time
#         self.sum_silog += n * result.silog
#         self.sum_photometric += n * result.photometric
#     def average(self):
#         avg = Result()
#         if self.count > 0:
#             avg.update(
#                 self.sum_irmse / self.count, self.sum_imae / self.count,
#                 self.sum_mse / self.count, self.sum_rmse / self.count,
#                 self.sum_mae / self.count, self.sum_absrel / self.count,
#                 self.sum_squared_rel / self.count, self.sum_lg10 / self.count,
#                 self.sum_delta1 / self.count, self.sum_delta2 / self.count,
#                 self.sum_delta3 / self.count, self.sum_gpu_time / self.count,
#                 self.sum_data_time / self.count, self.sum_silog / self.count,
#                 self.sum_photometric / self.count)
#         return avg

# # ----------------------------
# # IO helpers
# # ----------------------------
# def load_depth_m_from_16bit_png(png_path: str) -> torch.Tensor:
#     """16-bit PNG (value = depth[m]*256) -> torch.float32 meters."""
#     arr = np.array(Image.open(png_path), dtype=np.uint16)
#     depth_m = arr.astype(np.float32) / 256.0
#     return torch.from_numpy(depth_m)

# def load_depth_m_from_npy(npy_path: str) -> torch.Tensor:
#     """NPY (meters) -> torch.float32 meters."""
#     arr = np.load(npy_path).astype(np.float32)
#     return torch.from_numpy(arr)

# # def gt_path_from_pred_path(pred_path: str, gt_dir: str) -> str:
# #     """
# #     Map PRED filename like:
# #       2011_09_26_drive_0002_sync_image_0000000006_image_02.png
# #     to GT filename (NPY) like:
# #       2011_09_26_drive_0002_sync_velodyne_raw_0000000006_image_02.npy
# #     Rule: replace first occurrence of "_image_" with "_velodyne_raw_"
# #     """
# #     base = os.path.basename(pred_path)        # ..._sync_image_0000000006_image_02.png
# #     stem, _ = os.path.splitext(base)          # ..._sync_image_0000000006_image_02
# #     stem_gt = stem.replace("_image_", "_velodyne_raw_", 1)
# #     return os.path.join(gt_dir, stem_gt + ".npy")
# import os
# import re

# def gt_path_from_pred_path(pred_path: str, gt_dir: str) -> str:
#     """
#     更健壮的 GT 路径映射：
#     1) 把 _sync_image_ 替换为 _sync_velodyne_raw_
#     2) 如果尾部是 _velodyne_raw_XX，则尝试：
#         - _image_XX
#         - _velodyne_image_XX
#        并返回第一个真实存在的 GT 文件路径
#     """
#     base = os.path.basename(pred_path)          # xxx.png
#     stem, _ = os.path.splitext(base)            # xxx

#     # 先把 sync 段替换正确（只替换一次）
#     s1 = stem.replace("_sync_image_", "_sync_velodyne_raw_", 1)

#     candidates = []

#     # 候选1：直接用 s1
#     candidates.append(s1 + ".npy")

#     # 若末尾出现 _velodyne_raw_XX，构造更多候选
#     m = re.search(r"_velodyne_raw_(\d{2})$", s1)
#     if m:
#         cam = m.group(1)
#         s_img = re.sub(r"_velodyne_raw_\d{2}$", f"_image_{cam}", s1)
#         s_vimg = re.sub(r"_velodyne_raw_\d{2}$", f"_velodyne_image_{cam}", s1)
#         candidates.append(s_img + ".npy")
#         candidates.append(s_vimg + ".npy")

#     # 按候选顺序找第一个存在的
#     for fname in candidates:
#         p = os.path.join(gt_dir, fname)
#         if os.path.exists(p):
#             return p

#     # 都不存在：返回“最可能”的第一个，让外层打印 MISS_GT
#     return os.path.join(gt_dir, candidates[0])

# # ----------------------------
# # Main
# # ----------------------------
# def main():
#     ap = argparse.ArgumentParser(
#         description="Evaluate metrics: PRED=16-bit PNG (m*256), GT=NPY (meters). Pred name '_sync_image_' -> GT name '_sync_velodyne_raw_'."
#     )
#     ap.add_argument("--gt-dir", required=True, help="GT directory (NPY meters).")
#     ap.add_argument("--pred-dir", required=True, help="Pred directory (16-bit PNG, m*256).")
#     ap.add_argument("--out-summary", required=True, help="Path to save summary .txt")
#     args = ap.parse_args()

#     gt_dir = args.gt_dir
#     pred_dir = args.pred_dir
#     out_txt = args.out_summary
#     os.makedirs(os.path.dirname(out_txt), exist_ok=True)

#     # iterate over PRED PNGs
#     pred_files = sorted(glob(os.path.join(pred_dir, "*.png")))

#     meter = AverageMeter()
#     n_total = 0
#     n_eval = 0
#     n_missing_gt = 0
#     n_size_mismatch = 0
#     lines = []

#     header = "filename,irmse,imae,rmse,mae,absrel,sq_rel,lg10,delta1,delta2,delta3,silog\n"
#     lines.append(header)

#     for pred_path in pred_files:
#         n_total += 1
#         gt_path = gt_path_from_pred_path(pred_path, gt_dir)

#         if not os.path.exists(gt_path):
#             n_missing_gt += 1
#             print(f"[MISS_GT] {os.path.basename(gt_path)}")
#             continue

#         pred_m = load_depth_m_from_16bit_png(pred_path)  # meters
#         gt_m   = load_depth_m_from_npy(gt_path)          # meters

#         if pred_m.shape != gt_m.shape:
#             n_size_mismatch += 1
#             print(f"[SKIP] size mismatch for {os.path.basename(pred_path)}: pred {pred_m.shape} vs gt {gt_m.shape}")
#             continue

#         res = Result()
#         res.evaluate(output=pred_m, target=gt_m)
#         meter.update(res, n=1)
#         n_eval += 1

#         name = os.path.basename(pred_path)
#         lines.append(
#             f"{name},{res.irmse:.6f},{res.imae:.6f},"
#             f"{res.rmse:.6f},{res.mae:.6f},{res.absrel:.6f},{res.squared_rel:.6f},"
#             f"{res.lg10:.6f},{res.delta1:.6f},{res.delta2:.6f},{res.delta3:.6f},{res.silog:.6f}\n"
#         )

#     avg = meter.average()
#     lines.append("\n# SUMMARY\n")
#     lines.append(f"# total_pred_png: {n_total}, evaluated: {n_eval}, missing_gt: {n_missing_gt}, size_mismatch: {n_size_mismatch}\n")
#     if n_eval > 0:
#         lines.append(
#             "AVG,"
#             f"{avg.irmse:.6f},{avg.imae:.6f},{avg.rmse:.6f},{avg.mae:.6f},"
#             f"{avg.absrel:.6f},{avg.squared_rel:.6f},{avg.lg10:.6f},"
#             f"{avg.delta1:.6f},{avg.delta2:.6f},{avg.delta3:.6f},{avg.silog:.6f}\n"
#         )
#     else:
#         lines.append("AVG,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN\n")

#     with open(out_txt, "w") as f:
#         f.writelines(lines)

#     print(f"[DONE] Wrote summary to: {out_txt}")
#     print(f"      total_pred_png={n_total}, eval={n_eval}, missing_gt={n_missing_gt}, size_mismatch={n_size_mismatch}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from glob import glob
import math
import numpy as np
from PIL import Image
import torch
import re

# ----------------------------
# Metrics
# ----------------------------
lg_e_10 = math.log(10)

def log10(x):
    return torch.log(x) / lg_e_10

def psnr_from_mse(mse: float, max_i: float = 65535.0) -> float:
    """PSNR = 10 * log10(MAX_I^2 / MSE). mse<=0 -> inf/nan"""
    if mse is None or (isinstance(mse, float) and (math.isnan(mse))):
        return float("nan")
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((max_i * max_i) / mse)

class Result(object):
    def __init__(self):
        self.irmse = 0
        self.imae = 0
        self.mse = 0
        self.rmse = 0
        self.mae = 0
        self.absrel = 0
        self.squared_rel = 0
        self.lg10 = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0
        self.silog = 0
        self.photometric = 0
        self.psnr = float("nan")  # optional (16-bit single-channel PSNR)

    def set_to_worst(self):
        self.irmse = np.inf
        self.imae = np.inf
        self.mse = np.inf
        self.rmse = np.inf
        self.mae = np.inf
        self.absrel = np.inf
        self.squared_rel = np.inf
        self.lg10 = np.inf
        self.silog = np.inf
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0
        self.psnr = float("nan")

    def update(self, irmse, imae, mse, rmse, mae, absrel, squared_rel, lg10,
               delta1, delta2, delta3, gpu_time, data_time, silog, photometric=0, psnr=float("nan")):
        self.irmse = irmse
        self.imae = imae
        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.absrel = absrel
        self.squared_rel = squared_rel
        self.lg10 = lg10
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
        self.data_time = data_time
        self.gpu_time = gpu_time
        self.silog = silog
        self.photometric = photometric
        self.psnr = psnr

    def evaluate(self, output_m, target_m, photometric=0, pred_u16=None, gt_u16=None, compute_psnr=False):
        """
        output_m / target_m: torch.float32 meters
        pred_u16 / gt_u16: torch.float32 in [0,65535] (single-channel 16-bit domain)
        """
        epsilon = 1e-6
        valid_mask = target_m > 0.1  # meters

        if valid_mask.sum() == 0:
            self.irmse = float('nan')
            self.imae = float('nan')
            self.rmse = float('nan')
            self.mae = float('nan')
            self.silog = float('nan')
            self.absrel = float('nan')
            self.squared_rel = float('nan')
            self.lg10 = float('nan')
            self.delta1 = 0
            self.delta2 = 0
            self.delta3 = 0
            self.photometric = float(photometric)
            self.psnr = float("nan")
            return

        output_valid = output_m[valid_mask]
        target_valid = target_m[valid_mask]

        abs_diff = (output_valid - target_valid).abs()
        self.mse = float((abs_diff ** 2).mean())
        self.rmse = float(torch.sqrt(torch.tensor(self.mse)))
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output_valid + epsilon) - log10(target_valid + epsilon)).abs().mean())
        self.absrel = float((abs_diff / target_valid).mean())
        self.squared_rel = float(((abs_diff / target_valid) ** 2).mean())

        maxRatio = torch.max(output_valid / target_valid, target_valid / output_valid)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())

        err_log = torch.log(target_valid + epsilon) - torch.log(output_valid + epsilon)
        normalized_squared_log = (err_log ** 2).mean()
        log_mean = err_log.mean()
        self.silog = math.sqrt(float(normalized_squared_log - log_mean * log_mean)) * 100

        inv_mask = output_valid > 0
        if inv_mask.sum() == 0:
            self.irmse = float('nan')
            self.imae = float('nan')
        else:
            inv_output = 1.0 / (output_valid[inv_mask] * 1e-3)
            inv_target = 1.0 / (target_valid[inv_mask] * 1e-3)
            abs_inv_diff = (inv_output - inv_target).abs()
            self.irmse = float(torch.sqrt((abs_inv_diff ** 2).mean()))
            self.imae = float(abs_inv_diff.mean())

        # Optional PSNR in 16-bit single-channel domain
        if compute_psnr and (pred_u16 is not None) and (gt_u16 is not None):
            pred_u16_v = pred_u16[valid_mask]
            gt_u16_v = gt_u16[valid_mask]
            mse_u16 = float(((pred_u16_v - gt_u16_v) ** 2).mean())
            self.psnr = psnr_from_mse(mse_u16, max_i=65535.0)
        else:
            self.psnr = float("nan")

        self.data_time = 0
        self.gpu_time = 0
        self.photometric = float(photometric)

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.count = 0.0
        self.sum_irmse = 0
        self.sum_imae = 0
        self.sum_mse = 0
        self.sum_rmse = 0
        self.sum_mae = 0
        self.sum_absrel = 0
        self.sum_squared_rel = 0
        self.sum_lg10 = 0
        self.sum_delta1 = 0
        self.sum_delta2 = 0
        self.sum_delta3 = 0
        self.sum_data_time = 0
        self.sum_gpu_time = 0
        self.sum_photometric = 0
        self.sum_silog = 0
        self.sum_psnr = 0.0
        self.count_psnr = 0.0  # only count valid psnr values

    def update(self, result, gpu_time=0.0, data_time=0.0, n=1):
        self.count += n
        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_squared_rel += n * result.squared_rel
        self.sum_lg10 += n * result.lg10
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3
        self.sum_data_time += n * data_time
        self.sum_gpu_time += n * gpu_time
        self.sum_silog += n * result.silog
        self.sum_photometric += n * result.photometric

        if not math.isnan(result.psnr):
            self.sum_psnr += n * result.psnr
            self.count_psnr += n

    def average(self):
        avg = Result()
        if self.count > 0:
            avg_psnr = float("nan")
            if self.count_psnr > 0:
                avg_psnr = self.sum_psnr / self.count_psnr

            avg.update(
                self.sum_irmse / self.count, self.sum_imae / self.count,
                self.sum_mse / self.count, self.sum_rmse / self.count,
                self.sum_mae / self.count, self.sum_absrel / self.count,
                self.sum_squared_rel / self.count, self.sum_lg10 / self.count,
                self.sum_delta1 / self.count, self.sum_delta2 / self.count,
                self.sum_delta3 / self.count, self.sum_gpu_time / self.count,
                self.sum_data_time / self.count, self.sum_silog / self.count,
                self.sum_photometric / self.count, psnr=avg_psnr
            )
        return avg

# ----------------------------
# IO helpers
# ----------------------------
def load_pred_from_16bit_png(png_path: str):
    """
    16-bit single-channel PNG:
      raw value = depth[m] * 256
    Return:
      pred_m : torch.float32 meters
      pred_u16f : torch.float32 in [0, 65535] (16-bit domain)
    """
    arr_u16 = np.array(Image.open(png_path), dtype=np.uint16)  # HxW, 16-bit
    pred_u16f = torch.from_numpy(arr_u16.astype(np.float32))
    pred_m = pred_u16f / 256.0
    return pred_m, pred_u16f

def load_depth_m_from_npy(npy_path: str) -> torch.Tensor:
    """NPY (meters) -> torch.float32 meters."""
    arr = np.load(npy_path).astype(np.float32)
    return torch.from_numpy(arr)

def gt_u16_from_gt_m(gt_m: torch.Tensor) -> torch.Tensor:
    """
    Convert GT meters -> 16-bit domain float:
      u16 = round(clip(depth[m]*256, 0..65535))
    """
    return torch.clamp(gt_m * 256.0, 0.0, 65535.0).round()

def gt_path_from_pred_path(pred_path: str, gt_dir: str) -> str:
    """
    更健壮的 GT 路径映射：
    1) 把 _sync_image_ 替换为 _sync_velodyne_raw_
    2) 如果尾部是 _velodyne_raw_XX，则尝试：
        - _image_XX
        - _velodyne_image_XX
       并返回第一个真实存在的 GT 文件路径
    """
    base = os.path.basename(pred_path)
    stem, _ = os.path.splitext(base)

    s1 = stem.replace("_sync_image_", "_sync_velodyne_raw_", 1)

    candidates = []
    candidates.append(s1 + ".npy")

    m = re.search(r"_velodyne_raw_(\d{2})$", s1)
    if m:
        cam = m.group(1)
        s_img = re.sub(r"_velodyne_raw_\d{2}$", f"_image_{cam}", s1)
        s_vimg = re.sub(r"_velodyne_raw_\d{2}$", f"_velodyne_image_{cam}", s1)
        candidates.append(s_img + ".npy")
        candidates.append(s_vimg + ".npy")

    for fname in candidates:
        p = os.path.join(gt_dir, fname)
        if os.path.exists(p):
            return p

    return os.path.join(gt_dir, candidates[0])

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate metrics: PRED=16-bit PNG (m*256), GT=NPY (meters). Optional PSNR computed in 16-bit single-channel domain."
    )
    ap.add_argument("--gt-dir", required=True, help="GT directory (NPY meters).")
    ap.add_argument("--pred-dir", required=True, help="Pred directory (16-bit PNG, m*256).")
    ap.add_argument("--out-summary", required=True, help="Path to save summary .txt")
    ap.add_argument("--with-psnr", action="store_true",
                    help="Also compute PSNR in 16-bit single-channel domain (MAX_I=65535), on valid pixels (gt>0.1m).")
    args = ap.parse_args()

    gt_dir = args.gt_dir
    pred_dir = args.pred_dir
    out_txt = args.out_summary
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)

    pred_files = sorted(glob(os.path.join(pred_dir, "*.png")))

    meter = AverageMeter()
    n_total = 0
    n_eval = 0
    n_missing_gt = 0
    n_size_mismatch = 0
    lines = []

    # header
    if args.with_psnr:
        header = "filename,irmse,imae,rmse,mae,absrel,sq_rel,lg10,delta1,delta2,delta3,silog,psnr16\n"
    else:
        header = "filename,irmse,imae,rmse,mae,absrel,sq_rel,lg10,delta1,delta2,delta3,silog\n"
    lines.append(header)

    for pred_path in pred_files:
        n_total += 1
        gt_path = gt_path_from_pred_path(pred_path, gt_dir)

        if not os.path.exists(gt_path):
            n_missing_gt += 1
            print(f"[MISS_GT] {os.path.basename(gt_path)}")
            continue

        pred_m, pred_u16f = load_pred_from_16bit_png(pred_path)  # meters + 16-bit domain
        gt_m = load_depth_m_from_npy(gt_path)                    # meters

        if pred_m.shape != gt_m.shape:
            n_size_mismatch += 1
            print(f"[SKIP] size mismatch for {os.path.basename(pred_path)}: pred {pred_m.shape} vs gt {gt_m.shape}")
            continue

        gt_u16f = gt_u16_from_gt_m(gt_m) if args.with_psnr else None

        res = Result()
        res.evaluate(
            output_m=pred_m,
            target_m=gt_m,
            pred_u16=pred_u16f,
            gt_u16=gt_u16f,
            compute_psnr=args.with_psnr
        )

        meter.update(res, n=1)
        n_eval += 1

        name = os.path.basename(pred_path)
        if args.with_psnr:
            lines.append(
                f"{name},{res.irmse:.6f},{res.imae:.6f},"
                f"{res.rmse:.6f},{res.mae:.6f},{res.absrel:.6f},{res.squared_rel:.6f},"
                f"{res.lg10:.6f},{res.delta1:.6f},{res.delta2:.6f},{res.delta3:.6f},{res.silog:.6f},"
                f"{res.psnr:.6f}\n"
            )
        else:
            lines.append(
                f"{name},{res.irmse:.6f},{res.imae:.6f},"
                f"{res.rmse:.6f},{res.mae:.6f},{res.absrel:.6f},{res.squared_rel:.6f},"
                f"{res.lg10:.6f},{res.delta1:.6f},{res.delta2:.6f},{res.delta3:.6f},{res.silog:.6f}\n"
            )

    avg = meter.average()
    lines.append("\n# SUMMARY\n")
    lines.append(f"# total_pred_png: {n_total}, evaluated: {n_eval}, missing_gt: {n_missing_gt}, size_mismatch: {n_size_mismatch}\n")

    if n_eval > 0:
        if args.with_psnr:
            avg_psnr_str = "NaN" if math.isnan(avg.psnr) else f"{avg.psnr:.6f}"
            lines.append(
                "AVG,"
                f"{avg.irmse:.6f},{avg.imae:.6f},{avg.rmse:.6f},{avg.mae:.6f},"
                f"{avg.absrel:.6f},{avg.squared_rel:.6f},{avg.lg10:.6f},"
                f"{avg.delta1:.6f},{avg.delta2:.6f},{avg.delta3:.6f},{avg.silog:.6f},"
                f"{avg_psnr_str}\n"
            )
        else:
            lines.append(
                "AVG,"
                f"{avg.irmse:.6f},{avg.imae:.6f},{avg.rmse:.6f},{avg.mae:.6f},"
                f"{avg.absrel:.6f},{avg.squared_rel:.6f},{avg.lg10:.6f},"
                f"{avg.delta1:.6f},{avg.delta2:.6f},{avg.delta3:.6f},{avg.silog:.6f}\n"
            )
    else:
        if args.with_psnr:
            lines.append("AVG,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN\n")
        else:
            lines.append("AVG,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN\n")

    with open(out_txt, "w") as f:
        f.writelines(lines)

    print(f"[DONE] Wrote summary to: {out_txt}")
    print(f"      total_pred_png={n_total}, eval={n_eval}, missing_gt={n_missing_gt}, size_mismatch={n_size_mismatch}")
    if args.with_psnr:
        print("      PSNR: computed in 16-bit single-channel domain (MAX_I=65535) on valid pixels (gt>0.1m).")

if __name__ == "__main__":
    main()
