# # -*- coding: utf-8 -*-
# import glob
# import os
# import sys
# sys.path.append('/path/to/aurora-root/Code/ip_basic')

# import time
# import csv
# import cv2
# import numpy as np

# from ip_basic import depth_map_utils_original

# PY2 = sys.version_info[0] == 2

# def now_time():
#     # 高精度计时，兼容老版本
#     try:
#         return time.perf_counter()
#     except AttributeError:
#         return time.time()

# def compute_metrics(pred, gt):
#     """
#     pred, gt: float32 depth in meters (same shape). Only evaluate where gt > 0.
#     Returns: rmse, mae, irmse, imae
#     """
#     mask = gt > 0
#     if not np.any(mask):
#         return np.nan, np.nan, np.nan, np.nan

#     p = pred[mask]
#     g = gt[mask]

#     diff = p - g
#     rmse = float(np.sqrt(np.mean(diff ** 2)))
#     mae  = float(np.mean(np.abs(diff)))

#     # inverse depth metrics (m^-1); add small epsilon to avoid division by zero
#     eps = 1e-6
#     inv_p = 1.0 / np.maximum(p, eps)
#     inv_g = 1.0 / np.maximum(g, eps)
#     inv_diff = inv_p - inv_g
#     irmse = float(np.sqrt(np.mean(inv_diff ** 2)))
#     imae  = float(np.mean(np.abs(inv_diff)))

#     return rmse, mae, irmse, imae

# def ensure_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def open_csv_writer(path):
#     # Python 2 需要 'wb'；Python 3 用 newline='' 防止空行
#     if PY2:
#         return open(path, 'wb')
#     else:
#         return open(path, 'w', newline='')

# def main():
#     """
#     推理 + 保存预测 + 评估（RMSE/MAE/iRMSE/iMAE）+ 每图耗时统计
#     """

#     ##############################
#     # 路径设置（按你的要求）
#     ##############################
#     # 输入：扁平化的 velodyne_raw 图像目录
#     input_depth_dir = '/path/to/aurora-root/Dataset/AuroraKITTI/data/depth_selection/val_selection_cropped/velodyne_raw'
#     # 对应 GT 根目录（不强制使用，但保留配置）
#     gt_root_dir = '/path/to/aurora-root/Dataset/AuroraKITTI/data/depth_selection/val_selection_cropped/groundtruth_depth'
#     # 输出预测保存目录
#     output_root = '/path/to/aurora-root/Code/ip_basic/results/val'

#     ##############################
#     # 方法参数（与你当前配置一致）
#     ##############################
#     fill_type = 'multiscale'     # 'fast' 或 'multiscale'
#     extrapolate = False
#     blur_type = 'bilateral'
#     show_process = False         # 此脚本默认不弹可视化窗口
#     save_depth_maps = True

#     ensure_dir(output_root)

#     # 枚举所有 PNG
#     images_to_use = sorted(glob.glob(os.path.join(input_depth_dir, '*.png')))
#     if not images_to_use:
#         print('No images found under: {}'.format(input_depth_dir))
#         return

#     # 评估输出文件
#     csv_path = os.path.join(output_root, 'metrics_val.csv')
#     summary_path = os.path.join(output_root, 'metrics_summary.txt')

#     # 时间统计（滑动平均，仅用于进度显示）
#     avg_time_arr_length = 10
#     last_fill_times = np.repeat([1.0], avg_time_arr_length)
#     last_total_times = np.repeat([1.0], avg_time_arr_length)

#     # 汇总（两种口径：按像素“微平均”和按图“宏平均”）
#     sum_sq_err = 0.0
#     sum_abs_err = 0.0
#     sum_sq_inv_err = 0.0
#     sum_abs_inv_err = 0.0
#     global_valid_count = 0

#     # 宏平均：逐图指标的平均
#     per_img_metrics = []

#     # 耗时统计：逐图 & 平均
#     per_img_fill_times = []
#     per_img_total_times = []

#     num_images = len(images_to_use)
#     print("##################################################")
#     print("Found {} images".format(num_images))

#     # 打开 CSV（逐图）
#     fcsv = open_csv_writer(csv_path)
#     try:
#         writer = csv.writer(fcsv)
#         writer.writerow([
#             'filename', 'rmse', 'mae', 'irmse', 'imae', 'valid_pixels',
#             'fill_time_s', 'total_time_s'
#         ])

#         for i, depth_image_path in enumerate(images_to_use):
#             # 平均时间（用于进度显示）
#             avg_fill_time = float(np.mean(last_fill_times))
#             avg_total_time = float(np.mean(last_total_times))

#             # 进度条
#             et_remain = max(0.0, avg_total_time * (num_images - (i + 1)))
#             sys.stdout.write('\rProcessing {} / {}, Avg Fill: {:.5f}s, Avg Total: {:.5f}s, ETRem: {:.3f}s'.format(
#                 i + 1, num_images, avg_fill_time, avg_total_time, et_remain))
#             sys.stdout.flush()

#             start_total_time = now_time()

#             # 读取预测输入（uint16 → meters）
#             depth_u16 = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
#             if depth_u16 is None:
#                 sys.stdout.write("\n[WARN] Failed to read: {}\n".format(depth_image_path))
#                 writer.writerow([os.path.basename(depth_image_path),
#                                  np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan])
#                 per_img_fill_times.append(np.nan)
#                 per_img_total_times.append(np.nan)
#                 # 更新时间统计（保持滑动）
#                 last_fill_times = np.roll(last_fill_times, -1); last_fill_times[-1] = 0.0
#                 last_total_times = np.roll(last_total_times, -1); last_total_times[-1] = 0.0
#                 continue

#             projected_depths = np.float32(depth_u16) / 256.0  # meters

#             # 推理
#             start_fill_time = now_time()
#             if fill_type == 'fast':
#                 final_depths = depth_map_utils_original.fill_in_fast(
#                     projected_depths, extrapolate=extrapolate, blur_type=blur_type)
#                 process_dict = None
#             elif fill_type == 'multiscale':
#                 final_depths, process_dict = depth_map_utils_original.fill_in_multiscale(
#                     projected_depths, extrapolate=extrapolate, blur_type=blur_type, show_process=show_process)
#             else:
#                 raise ValueError('Invalid fill_type {}'.format(fill_type))
#             end_fill_time = now_time()

#             fill_time = end_fill_time - start_fill_time

#             # 保存预测（uint16 PNG，×256）
#             if save_depth_maps:
#                 out_path = os.path.join(output_root, os.path.basename(depth_image_path))
#                 pred_u16 = (np.clip(final_depths, 0, np.finfo(np.float32).max) * 256.0).astype(np.uint16)
#                 ok = cv2.imwrite(out_path, pred_u16)
#                 if not ok:
#                     sys.stdout.write("\n[WARN] Failed to write prediction: {}\n".format(out_path))

#             # 构造 GT 路径（名字相同，仅把 velodyne_raw 替换为 groundtruth_depth）
#             gt_path = depth_image_path.replace('/velodyne_raw/', '/groundtruth_depth/') \
#                                       .replace('_velodyne_raw_', '_groundtruth_depth_')

#             gt_u16 = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH)
#             if gt_u16 is None:
#                 sys.stdout.write("\n[WARN] GT not found for {} at {}\n".format(
#                     os.path.basename(depth_image_path), gt_path))
#                 rmse = mae = irmse = imae = np.nan
#                 valid_count = 0
#             else:
#                 gt = np.float32(gt_u16) / 256.0  # meters

#                 # 尺寸不一致时，调整预测尺寸到 GT
#                 pred = final_depths
#                 if pred.shape != gt.shape:
#                     pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

#                 rmse, mae, irmse, imae = compute_metrics(pred, gt)

#                 # 统计微平均
#                 mask = gt > 0
#                 valid_count = int(mask.sum())
#                 if valid_count > 0:
#                     p = pred[mask]; g = gt[mask]
#                     d = p - g
#                     sum_sq_err += float(np.sum(d ** 2))
#                     sum_abs_err += float(np.sum(np.abs(d)))

#                     eps = 1e-6
#                     inv_p = 1.0 / np.maximum(p, eps)
#                     inv_g = 1.0 / np.maximum(g, eps)
#                     inv_d = inv_p - inv_g
#                     sum_sq_inv_err += float(np.sum(inv_d ** 2))
#                     sum_abs_inv_err += float(np.sum(np.abs(inv_d)))
#                     global_valid_count += valid_count
#                 else:
#                     valid_count = 0

#             end_total_time = now_time()
#             total_time = end_total_time - start_total_time

#             # 写入逐图指标 + 耗时
#             writer.writerow([
#                 os.path.basename(depth_image_path),
#                 rmse, mae, irmse, imae, valid_count,
#                 fill_time, total_time
#             ])
#             per_img_metrics.append((rmse, mae, irmse, imae))
#             per_img_fill_times.append(fill_time)
#             per_img_total_times.append(total_time)

#             # 更新时间统计（进度显示）
#             last_fill_times = np.roll(last_fill_times, -1); last_fill_times[-1] = fill_time
#             last_total_times = np.roll(last_total_times, -1); last_total_times[-1] = total_time
#     finally:
#         fcsv.close()

#     # 汇总：宏平均（逐图指标的平均，忽略 nan）
#     per_img_arr = np.array(per_img_metrics, dtype=np.float64) if per_img_metrics else np.empty((0,4))
#     with np.errstate(invalid='ignore'):
#         macro_rmse  = float(np.nanmean(per_img_arr[:, 0])) if per_img_arr.size else np.nan
#         macro_mae   = float(np.nanmean(per_img_arr[:, 1])) if per_img_arr.size else np.nan
#         macro_irmse = float(np.nanmean(per_img_arr[:, 2])) if per_img_arr.size else np.nan
#         macro_imae  = float(np.nanmean(per_img_arr[:, 3])) if per_img_arr.size else np.nan

#     # 汇总：微平均（把所有有效像素合起来）
#     if global_valid_count > 0:
#         micro_rmse  = float(np.sqrt(sum_sq_err / global_valid_count))
#         micro_mae   = float(        sum_abs_err / global_valid_count)
#         micro_irmse = float(np.sqrt(sum_sq_inv_err / global_valid_count))
#         micro_imae  = float(        sum_abs_inv_err / global_valid_count)
#     else:
#         micro_rmse = micro_mae = micro_irmse = micro_imae = np.nan

#     # 耗时平均（忽略 nan）
#     avg_fill  = float(np.nanmean(per_img_fill_times)) if len(per_img_fill_times) else np.nan
#     avg_total = float(np.nanmean(per_img_total_times)) if len(per_img_total_times) else np.nan

#     # 保存 summary
#     with open(summary_path, 'w') as fs:
#         fs.write("===== Depth Completion Evaluation (val) =====\n")
#         fs.write("Images evaluated: {}\n".format(len(images_to_use)))
#         fs.write("Global valid pixels: {}\n\n".format(global_valid_count))

#         fs.write("[Macro Average per-image]\n")
#         fs.write("RMSE  (m): {}\n".format(macro_rmse))
#         fs.write("MAE   (m): {}\n".format(macro_mae))
#         fs.write("iRMSE(1/m): {}\n".format(macro_irmse))
#         fs.write("iMAE (1/m): {}\n\n".format(macro_imae))

#         fs.write("[Micro Average over all valid pixels]\n")
#         fs.write("RMSE  (m): {}\n".format(micro_rmse))
#         fs.write("MAE   (m): {}\n".format(micro_mae))
#         fs.write("iRMSE(1/m): {}\n".format(micro_irmse))
#         fs.write("iMAE (1/m): {}\n\n".format(micro_imae))

#         fs.write("[Timing]\n")
#         fs.write("Avg fill time per image (s):  {}\n".format(avg_fill))
#         fs.write("Avg total time per image (s): {}\n".format(avg_total))

#     print("\n\n===== Done =====")
#     print("Per-image metrics & timing saved to: {}".format(csv_path))
#     print("Summary saved to:                  {}".format(summary_path))
#     print("Predictions saved under:           {}".format(output_root))

# if __name__ == "__main__":
#     main()
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# import glob
# import os
# import sys
# sys.path.append('/path/to/aurora-root/Code/ip_basic')

# import time
# import csv
# import cv2
# import numpy as np

# from ip_basic import depth_map_utils_original

# PY2 = sys.version_info[0] == 2

# def now_time():
#     # 高精度计时，兼容老版本
#     try:
#         return time.perf_counter()
#     except AttributeError:
#         return time.time()

# def compute_metrics(pred, gt):
#     """
#     pred, gt: float32 depth in meters (same shape). Only evaluate where gt > 0.
#     Returns: rmse, mae, irmse, imae
#     """
#     mask = gt > 0
#     if not np.any(mask):
#         return np.nan, np.nan, np.nan, np.nan

#     p = pred[mask]
#     g = gt[mask]

#     diff = p - g
#     rmse = float(np.sqrt(np.mean(diff ** 2)))
#     mae  = float(np.mean(np.abs(diff)))

#     # inverse depth metrics (m^-1); add small epsilon to avoid division by zero
#     eps = 1e-6
#     inv_p = 1.0 / np.maximum(p, eps)
#     inv_g = 1.0 / np.maximum(g, eps)
#     inv_diff = inv_p - inv_g
#     irmse = float(np.sqrt(np.mean(inv_diff ** 2)))
#     imae  = float(np.mean(np.abs(inv_diff)))

#     return rmse, mae, irmse, imae

# def ensure_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def open_csv_writer(path):
#     # Python 2 需要 'wb'；Python 3 用 newline='' 防止空行
#     if PY2:
#         return open(path, 'wb')
#     else:
#         return open(path, 'w', newline='')

# def build_gt_path(pred_or_input_basename, gt_root_dir):
#     """
#     仅基于文件名构造 GT 路径：
#     2011_..._velodyne_raw_XXXXXXXXXX_image_0Y.png
#     -> 2011_..._groundtruth_depth_XXXXXXXXXX_image_0Y.png
#     """
#     gt_name = pred_or_input_basename.replace('_velodyne_raw_', '_groundtruth_depth_')
#     return os.path.join(gt_root_dir, gt_name)

# def main():
#     """
#     推理 + 保存预测 + 评估（RMSE/MAE/iRMSE/iMAE）+ 每图耗时统计
#     """

#     ##############################
#     # 路径设置（按你的要求）
#     ##############################
#     # 输入：扁平化的 velodyne_raw 图像目录
#     input_depth_dir = '/path/to/aurora-root/Dataset/AuroraKITTI/data/depth_selection/val_selection_cropped/velodyne_raw'
#     # 对应 GT 根目录（与输入不同目录、不同命名关键字）
#     gt_root_dir = '/path/to/aurora-root/Dataset/AuroraKITTI/data/depth_selection/val_selection_cropped/groundtruth_depth'
#     # 输出预测保存目录
#     output_root = '/path/to/aurora-root/Code/ip_basic/results/val'

#     ##############################
#     # 方法参数（与你当前配置一致）
#     ##############################
#     fill_type = 'multiscale'     # 'fast' 或 'multiscale'
#     extrapolate = False
#     blur_type = 'bilateral'
#     show_process = False         # 此脚本默认不弹可视化窗口
#     save_depth_maps = True

#     ensure_dir(output_root)

#     # 枚举所有 PNG（输入）
#     images_to_use = sorted(glob.glob(os.path.join(input_depth_dir, '*.png')))
#     if not images_to_use:
#         print('No images found under: {}'.format(input_depth_dir))
#         return

#     # 评估输出文件
#     csv_path = os.path.join(output_root, 'metrics_val.csv')
#     summary_path = os.path.join(output_root, 'metrics_summary.txt')

#     # 时间统计（滑动平均，仅用于进度显示）
#     avg_time_arr_length = 10
#     last_fill_times = np.repeat([1.0], avg_time_arr_length)
#     last_total_times = np.repeat([1.0], avg_time_arr_length)

#     # 汇总（两种口径：按像素“微平均”和按图“宏平均”）
#     sum_sq_err = 0.0
#     sum_abs_err = 0.0
#     sum_sq_inv_err = 0.0
#     sum_abs_inv_err = 0.0
#     global_valid_count = 0

#     # 宏平均：逐图指标的平均
#     per_img_metrics = []

#     # 耗时统计：逐图 & 平均
#     per_img_fill_times = []
#     per_img_total_times = []

#     num_images = len(images_to_use)
#     print("##################################################")
#     print("Found {} images".format(num_images))

#     # 打开 CSV（逐图）
#     fcsv = open_csv_writer(csv_path)
#     try:
#         writer = csv.writer(fcsv)
#         writer.writerow([
#             'filename', 'rmse', 'mae', 'irmse', 'imae', 'valid_pixels',
#             'fill_time_s', 'total_time_s'
#         ])

#         for i, depth_image_path in enumerate(images_to_use):
#             # 平均时间（用于进度显示）
#             avg_fill_time = float(np.mean(last_fill_times))
#             avg_total_time = float(np.mean(last_total_times))

#             # 进度条
#             et_remain = max(0.0, avg_total_time * (num_images - (i + 1)))
#             sys.stdout.write('\rProcessing {} / {}, Avg Fill: {:.5f}s, Avg Total: {:.5f}s, ETRem: {:.3f}s'.format(
#                 i + 1, num_images, avg_fill_time, avg_total_time, et_remain))
#             sys.stdout.flush()

#             start_total_time = now_time()

#             # 读取预测输入（uint16 → meters）
#             depth_u16 = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
#             if depth_u16 is None:
#                 sys.stdout.write("\n[WARN] Failed to read: {}\n".format(depth_image_path))
#                 writer.writerow([os.path.basename(depth_image_path),
#                                  np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan])
#                 per_img_fill_times.append(np.nan)
#                 per_img_total_times.append(np.nan)
#                 # 更新时间统计（保持滑动）
#                 last_fill_times = np.roll(last_fill_times, -1); last_fill_times[-1] = 0.0
#                 last_total_times = np.roll(last_total_times, -1); last_total_times[-1] = 0.0
#                 continue

#             projected_depths = np.float32(depth_u16) / 256.0  # meters

#             # 推理
#             start_fill_time = now_time()
#             if fill_type == 'fast':
#                 final_depths = depth_map_utils_original.fill_in_fast(
#                     projected_depths, extrapolate=extrapolate, blur_type=blur_type)
#                 process_dict = None
#             elif fill_type == 'multiscale':
#                 final_depths, process_dict = depth_map_utils_original.fill_in_multiscale(
#                     projected_depths, extrapolate=extrapolate, blur_type=blur_type, show_process=show_process)
#             else:
#                 raise ValueError('Invalid fill_type {}'.format(fill_type))
#             end_fill_time = now_time()
#             fill_time = end_fill_time - start_fill_time

#             # 保存预测（uint16 PNG，×256）
#             if save_depth_maps:
#                 out_basename = os.path.basename(depth_image_path)  # 与输入同名
#                 out_path = os.path.join(output_root, out_basename)
#                 pred_u16 = (np.clip(final_depths, 0, np.finfo(np.float32).max) * 256.0).astype(np.uint16)
#                 ok = cv2.imwrite(out_path, pred_u16)
#                 if not ok:
#                     sys.stdout.write("\n[WARN] Failed to write prediction: {}\n".format(out_path))

#             # === 基于“文件名”构造 GT 路径（关键改动） ===
#             in_basename = os.path.basename(depth_image_path)
#             gt_path = build_gt_path(in_basename, gt_root_dir)

#             gt_u16 = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH)
#             if gt_u16 is None:
#                 sys.stdout.write("\n[WARN] GT not found for {} at {}\n".format(in_basename, gt_path))
#                 rmse = mae = irmse = imae = np.nan
#                 valid_count = 0
#             else:
#                 gt = np.float32(gt_u16) / 256.0  # meters

#                 # 尺寸不一致时，调整预测尺寸到 GT
#                 pred = final_depths
#                 if pred.shape != gt.shape:
#                     pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

#                 rmse, mae, irmse, imae = compute_metrics(pred, gt)

#                 # 统计微平均
#                 mask = gt > 0
#                 valid_count = int(mask.sum())
#                 if valid_count > 0:
#                     p = pred[mask]; g = gt[mask]
#                     d = p - g
#                     sum_sq_err += float(np.sum(d ** 2))
#                     sum_abs_err += float(np.sum(np.abs(d)))

#                     eps = 1e-6
#                     inv_p = 1.0 / np.maximum(p, eps)
#                     inv_g = 1.0 / np.maximum(g, eps)
#                     inv_d = inv_p - inv_g
#                     sum_sq_inv_err += float(np.sum(inv_d ** 2))
#                     sum_abs_inv_err += float(np.sum(np.abs(inv_d)))
#                     global_valid_count += valid_count
#                 else:
#                     valid_count = 0

#             end_total_time = now_time()
#             total_time = end_total_time - start_total_time

#             # 写入逐图指标 + 耗时
#             writer.writerow([
#                 os.path.basename(depth_image_path),
#                 rmse, mae, irmse, imae, valid_count,
#                 fill_time, total_time
#             ])
#             per_img_metrics.append((rmse, mae, irmse, imae))
#             per_img_fill_times.append(fill_time)
#             per_img_total_times.append(total_time)

#             # 更新时间统计（进度显示）
#             last_fill_times = np.roll(last_fill_times, -1); last_fill_times[-1] = fill_time
#             last_total_times = np.roll(last_total_times, -1); last_total_times[-1] = total_time
#     finally:
#         fcsv.close()

#     # 汇总：宏平均（逐图指标的平均，忽略 nan）
#     per_img_arr = np.array(per_img_metrics, dtype=np.float64) if per_img_metrics else np.empty((0,4))
#     with np.errstate(invalid='ignore'):
#         macro_rmse  = float(np.nanmean(per_img_arr[:, 0])) if per_img_arr.size else np.nan
#         macro_mae   = float(np.nanmean(per_img_arr[:, 1])) if per_img_arr.size else np.nan
#         macro_irmse = float(np.nanmean(per_img_arr[:, 2])) if per_img_arr.size else np.nan
#         macro_imae  = float(np.nanmean(per_img_arr[:, 3])) if per_img_arr.size else np.nan

#     # 汇总：微平均（把所有有效像素合起来）
#     if global_valid_count > 0:
#         micro_rmse  = float(np.sqrt(sum_sq_err / global_valid_count))
#         micro_mae   = float(        sum_abs_err / global_valid_count)
#         micro_irmse = float(np.sqrt(sum_sq_inv_err / global_valid_count))
#         micro_imae  = float(        sum_abs_inv_err / global_valid_count)
#     else:
#         micro_rmse = micro_mae = micro_irmse = micro_imae = np.nan

#     # 耗时平均（忽略 nan）
#     avg_fill  = float(np.nanmean(per_img_fill_times)) if len(per_img_fill_times) else np.nan
#     avg_total = float(np.nanmean(per_img_total_times)) if len(per_img_total_times) else np.nan

#     # 保存 summary
#     summary_path = os.path.join(output_root, 'metrics_summary.txt')
#     with open(summary_path, 'w') as fs:
#         fs.write("===== Depth Completion Evaluation (val) =====\n")
#         fs.write("Images evaluated: {}\n".format(len(images_to_use)))
#         fs.write("Global valid pixels: {}\n\n".format(global_valid_count))

#         fs.write("[Macro Average per-image]\n")
#         fs.write("RMSE  (m): {}\n".format(macro_rmse))
#         fs.write("MAE   (m): {}\n".format(macro_mae))
#         fs.write("iRMSE(1/m): {}\n".format(macro_irmse))
#         fs.write("iMAE (1/m): {}\n\n".format(macro_imae))

#         fs.write("[Micro Average over all valid pixels]\n")
#         fs.write("RMSE  (m): {}\n".format(micro_rmse))
#         fs.write("MAE   (m): {}\n".format(micro_mae))
#         fs.write("iRMSE(1/m): {}\n".format(micro_irmse))
#         fs.write("iMAE (1/m): {}\n\n".format(micro_imae))

#         fs.write("[Timing]\n")
#         fs.write("Avg fill time per image (s):  {}\n".format(avg_fill))
#         fs.write("Avg total time per image (s): {}\n".format(avg_total))

#     print("\n\n===== Done =====")
#     print("Per-image metrics & timing saved to: {}".format(csv_path))
#     print("Summary saved to:                  {}".format(summary_path))
#     print("Predictions saved under:           {}".format(output_root))

# if __name__ == "__main__":
#     main()

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import glob
import os
import sys
sys.path.append('/path/to/aurora-root/Code/ip_basic')

import time
import csv
import cv2
import numpy as np

from ip_basic import depth_map_utils_original

PY2 = sys.version_info[0] == 2

def now_time():
    # 高精度计时，兼容老版本
    try:
        return time.perf_counter()
    except AttributeError:
        return time.time()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def open_csv_writer(path):
    # Python 2 用 'wb'；Python 3 用 newline='' 防止空行
    if PY2:
        return open(path, 'wb')
    else:
        return open(path, 'w', newline='')

def build_gt_path(pred_or_input_basename, gt_root_dir):
    """
    仅基于文件名构造 GT 路径：
    2011_..._velodyne_raw_XXXXXXXXXX_image_0Y.png
    -> 2011_..._groundtruth_depth_XXXXXXXXXX_image_0Y.png
    """
    gt_name = pred_or_input_basename.replace('_velodyne_raw_', '_groundtruth_depth_')
    return os.path.join(gt_root_dir, gt_name)

def compute_metrics_numpy(pred, gt, valid_min=0.1):
    """
    复刻你给的 PyTorch metrics（使用 NumPy 实现）
    评估仅在 gt > valid_min (m) 的像素上进行。
    返回: dict with keys:
      irmse(1/km), imae(1/km), mse(m^2), rmse(m), mae(m),
      absrel, squared_rel, lg10, delta1, delta2, delta3, silog
    """
    eps = 1e-6

    mask = gt > valid_min
    if not np.any(mask):
        return {
            'irmse': np.nan, 'imae': np.nan, 'mse': np.nan, 'rmse': np.nan, 'mae': np.nan,
            'absrel': np.nan, 'squared_rel': np.nan, 'lg10': np.nan,
            'delta1': 0.0, 'delta2': 0.0, 'delta3': 0.0, 'silog': np.nan
        }

    out = pred[mask]
    tgt = gt[mask]

    abs_diff = np.abs(out - tgt)

    mse = float(np.mean(abs_diff ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(abs_diff))
    absrel = float(np.mean(abs_diff / np.maximum(tgt, eps)))
    squared_rel = float(np.mean((abs_diff / np.maximum(tgt, eps)) ** 2))

    # lg10（避免 log(0)）
    lg10 = float(np.mean(np.abs(
        np.log10(np.maximum(out, eps)) - np.log10(np.maximum(tgt, eps))
    )))

    # delta 指标（避免除以 0）
    max_ratio = np.maximum(
        out / np.maximum(tgt, eps),
        tgt / np.maximum(out, eps)
    )
    delta1 = float(np.mean(max_ratio < 1.25))
    delta2 = float(np.mean(max_ratio < (1.25 ** 2)))
    delta3 = float(np.mean(max_ratio < (1.25 ** 3)))

    # silog（自然对数），*100
    err_log = np.log(np.maximum(tgt, eps)) - np.log(np.maximum(out, eps))
    normalized_squared_log = float(np.mean(err_log ** 2))
    log_mean = float(np.mean(err_log))
    silog = float(np.sqrt(max(0.0, normalized_squared_log - log_mean * log_mean)) * 100.0)

    # inverse depth metrics（与参考一致：pred>0 才参与；单位 1/km）
    inv_mask = out > 0
    if not np.any(inv_mask):
        irmse = np.nan
        imae = np.nan
    else:
        inv_out = 1000.0 / out[inv_mask]  # 1/km
        inv_tgt = 1000.0 / tgt[inv_mask]
        abs_inv_diff = np.abs(inv_out - inv_tgt)
        irmse = float(np.sqrt(np.mean(abs_inv_diff ** 2)))
        imae  = float(np.mean(abs_inv_diff))

    return {
        'irmse': irmse, 'imae': imae, 'mse': mse, 'rmse': rmse, 'mae': mae,
        'absrel': absrel, 'squared_rel': squared_rel, 'lg10': lg10,
        'delta1': delta1, 'delta2': delta2, 'delta3': delta3, 'silog': silog
    }

def main():
    """
    推理 + 保存预测 + 评估（按参考代码的 metrics）+ 每图耗时统计
    """
    ##############################
    # 路径设置
    ##############################
    input_depth_dir = '/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti_AURORA/val_selection_cropped_TEST/velodyne_raw'
    gt_root_dir     = '/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti_AURORA/val_selection_cropped_TEST//groundtruth_depth'
    output_root     = '/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti_AURORA/val_selection_cropped_TEST//dens'

    ##############################
    # 方法参数
    ##############################
    fill_type = 'multiscale'     # 'fast' or 'multiscale'
    extrapolate = False
    blur_type = 'bilateral'
    show_process = False
    save_depth_maps = True

    ensure_dir(output_root)

    images_to_use = sorted(glob.glob(os.path.join(input_depth_dir, '*.png')))
    if not images_to_use:
        print('No images found under: {}'.format(input_depth_dir))
        return

    csv_path = os.path.join(output_root, 'metrics_val.csv')
    summary_path = os.path.join(output_root, 'metrics_summary.txt')

    # 进度滑动平均
    avg_time_arr_length = 10
    last_fill_times = np.repeat([1.0], avg_time_arr_length)
    last_total_times = np.repeat([1.0], avg_time_arr_length)

    # 逐图指标收集（用于“按图平均”）
    metrics_list = []
    per_img_fill_times = []
    per_img_total_times = []

    num_images = len(images_to_use)
    print("##################################################")
    print("Found {} images".format(num_images))

    fcsv = open_csv_writer(csv_path)
    try:
        writer = csv.writer(fcsv)
        writer.writerow([
            'filename',
            'irmse(1_per_km)', 'imae(1_per_km)',
            'mse(m2)', 'rmse(m)', 'mae(m)',
            'absrel', 'squared_rel', 'lg10',
            'delta1', 'delta2', 'delta3',
            'silog', 'valid_pixels',
            'fill_time_s', 'total_time_s'
        ])

        for i, depth_image_path in enumerate(images_to_use):
            # 进度显示
            avg_fill_time = float(np.mean(last_fill_times))
            avg_total_time = float(np.mean(last_total_times))
            et_remain = max(0.0, avg_total_time * (num_images - (i + 1)))
            sys.stdout.write('\rProcessing {} / {}, Avg Fill: {:.5f}s, Avg Total: {:.5f}s, ETRem: {:.3f}s'.format(
                i + 1, num_images, avg_fill_time, avg_total_time, et_remain))
            sys.stdout.flush()

            start_total_time = now_time()

            # 读取输入（uint16 → meters）
            depth_u16 = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
            if depth_u16 is None:
                sys.stdout.write("\n[WARN] Failed to read: {}\n".format(depth_image_path))
                writer.writerow([os.path.basename(depth_image_path)] + [np.nan]*13 + [0, np.nan, np.nan])
                per_img_fill_times.append(np.nan)
                per_img_total_times.append(np.nan)
                last_fill_times = np.roll(last_fill_times, -1); last_fill_times[-1] = 0.0
                last_total_times = np.roll(last_total_times, -1); last_total_times[-1] = 0.0
                continue
            projected_depths = np.float32(depth_u16) / 256.0  # meters

            # 深度补全
            start_fill_time = now_time()
            if fill_type == 'fast':
                final_depths = depth_map_utils_original.fill_in_fast(
                    projected_depths, extrapolate=extrapolate, blur_type=blur_type)
            elif fill_type == 'multiscale':
                final_depths, _ = depth_map_utils_original.fill_in_multiscale(
                    projected_depths, extrapolate=extrapolate, blur_type=blur_type, show_process=show_process)
            else:
                raise ValueError('Invalid fill_type {}'.format(fill_type))
            end_fill_time = now_time()
            fill_time = end_fill_time - start_fill_time

            # 保存预测（uint16，×256）
            if save_depth_maps:
                out_basename = os.path.basename(depth_image_path)
                out_path = os.path.join(output_root, out_basename)
                pred_u16 = (np.clip(final_depths, 0, np.finfo(np.float32).max) * 256.0).astype(np.uint16)
                ok = cv2.imwrite(out_path, pred_u16)
                if not ok:
                    sys.stdout.write("\n[WARN] Failed to write prediction: {}\n".format(out_path))

            # 基于文件名映射 GT
            in_basename = os.path.basename(depth_image_path)
            gt_path = build_gt_path(in_basename, gt_root_dir)

            gt_u16 = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH)
            if gt_u16 is None:
                sys.stdout.write("\n[WARN] GT not found for {} at {}\n".format(in_basename, gt_path))
                metrics = {
                    'irmse': np.nan, 'imae': np.nan, 'mse': np.nan, 'rmse': np.nan, 'mae': np.nan,
                    'absrel': np.nan, 'squared_rel': np.nan, 'lg10': np.nan,
                    'delta1': 0.0, 'delta2': 0.0, 'delta3': 0.0, 'silog': np.nan
                }
                valid_count = 0
            else:
                gt = np.float32(gt_u16) / 256.0  # meters

                # 尺寸对齐
                pred = final_depths
                if pred.shape != gt.shape:
                    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

                # 评估（按参考实现）
                metrics = compute_metrics_numpy(pred, gt, valid_min=0.1)
                valid_count = int((gt > 0.1).sum())

            end_total_time = now_time()
            total_time = end_total_time - start_total_time

            # 写 CSV
            writer.writerow([
                in_basename,
                metrics['irmse'], metrics['imae'],
                metrics['mse'], metrics['rmse'], metrics['mae'],
                metrics['absrel'], metrics['squared_rel'], metrics['lg10'],
                metrics['delta1'], metrics['delta2'], metrics['delta3'],
                metrics['silog'], valid_count,
                fill_time, total_time
            ])

            metrics_list.append(metrics)
            per_img_fill_times.append(fill_time)
            per_img_total_times.append(total_time)

            # 进度滑动更新
            last_fill_times = np.roll(last_fill_times, -1); last_fill_times[-1] = fill_time
            last_total_times = np.roll(last_total_times, -1); last_total_times[-1] = total_time
    finally:
        fcsv.close()

    # ===== 按图平均（与参考 AverageMeter.average() 一致的口径）=====
    def avg_of(key):
        arr = np.array([m[key] for m in metrics_list], dtype=np.float64) if metrics_list else np.array([])
        return float(np.nanmean(arr)) if arr.size else np.nan

    avg_irmse = avg_of('irmse')
    avg_imae  = avg_of('imae')
    avg_mse   = avg_of('mse')
    avg_rmse  = avg_of('rmse')
    avg_mae   = avg_of('mae')
    avg_absrel= avg_of('absrel')
    avg_sqrel = avg_of('squared_rel')
    avg_lg10  = avg_of('lg10')
    avg_d1    = avg_of('delta1')
    avg_d2    = avg_of('delta2')
    avg_d3    = avg_of('delta3')
    avg_silog = avg_of('silog')

    avg_fill  = float(np.nanmean(per_img_fill_times)) if len(per_img_fill_times) else np.nan
    avg_total = float(np.nanmean(per_img_total_times)) if len(per_img_total_times) else np.nan

    # 写 summary
    with open(summary_path, 'w') as fs:
        fs.write("===== Depth Completion Evaluation (val) =====\n")
        fs.write("Images evaluated: {}\n\n".format(len(images_to_use)))

        fs.write("[Per-image Averages]\n")
        fs.write("iRMSE (1/km): {}\n".format(avg_irmse))
        fs.write("iMAE  (1/km): {}\n".format(avg_imae))
        fs.write("MSE    (m^2): {}\n".format(avg_mse))
        fs.write("RMSE     (m): {}\n".format(avg_rmse))
        fs.write("MAE      (m): {}\n".format(avg_mae))
        fs.write("AbsRel      : {}\n".format(avg_absrel))
        fs.write("SqRel       : {}\n".format(avg_sqrel))
        fs.write("Lg10        : {}\n".format(avg_lg10))
        fs.write("delta1      : {}\n".format(avg_d1))
        fs.write("delta2      : {}\n".format(avg_d2))
        fs.write("delta3      : {}\n".format(avg_d3))
        fs.write("SiLog       : {}\n\n".format(avg_silog))

        fs.write("[Timing]\n")
        fs.write("Avg fill time per image (s):  {}\n".format(avg_fill))
        fs.write("Avg total time per image (s): {}\n".format(avg_total))

    print("\n\n===== Done =====")
    print("Per-image metrics & timing saved to: {}".format(csv_path))
    print("Summary saved to:                  {}".format(summary_path))
    print("Predictions saved under:           {}".format(output_root))

if __name__ == "__main__":
    main()
