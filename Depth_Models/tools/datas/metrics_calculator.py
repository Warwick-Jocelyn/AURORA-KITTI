import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from scipy.ndimage import convolve

# 尝试导入 skimage 用于 SSIM
try:
    from skimage.metrics import structural_similarity as ssim_func
except ImportError:
    print("请安装 scikit-image: pip install scikit-image")
    exit()

# ================= 配置区域 =================

# 预测结果文件夹
PRED_FOLDER = "/path/to/aurora-root/Code/OMNI-DC/output_DENSE_Original"
# Ground Truth 文件夹
GT_FOLDER = "/path/to/aurora-root/Dataset/PixelAccurateDepthBenchmark/datas/GT_renamed"

# 有效深度范围 (KITTI 标准通常是 0m - 80m)
MIN_DEPTH = 1e-5
MAX_DEPTH = 80.0
print(PRED_FOLDER)
# 缩放因子：将 16bit 像素值转换为米 (Meters)
PRED_SCALE_FACTOR = 256.0
GT_SCALE_FACTOR = 256.0

# 逆深度的缩放：
#   - 如果想和你当前脚本一样，用 1/km，则 INV_SCALE=1000.0
#   - 如果想和 MetricALL 里完全一致（1/m），改成 INV_SCALE=1.0
INV_SCALE = 1000.0   # 1000.0 -> 1/km, 1.0 -> 1/m

# ===========================================

def get_args():
    parser = argparse.ArgumentParser(description="Depth Evaluation Script")
    parser.add_argument('--mode', type=str, default='MetricsAll', choices=['Errors', 'IQA', 'MetricsAll'],
                        help="选择评估模式: 'Errors' (仅误差指标), 'IQA' (SSIM/PSNR), 'MetricsAll' (全部)")
    return parser.parse_args()

def get_channel_and_bits(img):
    if img.ndim == 2:
        channels = 1
    else:
        channels = img.shape[2]
    dtype = img.dtype
    if np.issubdtype(dtype, np.integer):
        bits = np.iinfo(dtype).bits
    elif np.issubdtype(dtype, np.floating):
        bits = "float"
    else:
        bits = "unknown"
    return channels, bits, dtype

# ================= MS-SSIM NumPy 实现 =================

def matlab_style_gauss2D(shape=(11, 11), sigma=1.5):
    """
    生成类似 Matlab fspecial('gaussian') 的 2D 高斯核
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(img, fltr, mode='reflect'):
    """
    二维卷积，输出大小与输入一致
    """
    return convolve(img, fltr, mode=mode)

def _ssim_for_ms(img1, img2,
                 K1=0.01, K2=0.03,
                 max_val=80.0,
                 window=None):
    """
    单尺度 SSIM 计算，用于 MS-SSIM：
    返回:
        l_mean  : 亮度项 l 的均值
        cs_mean : 对比度*结构项 (c*s) 的均值（通常简称 cs）
    """
    if window is None:
        window = matlab_style_gauss2D((11, 11), 1.5)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 均值
    mu1 = filter2(img1, window, mode='reflect')
    mu2 = filter2(img2, window, mode='reflect')

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # 方差 / 协方差
    sigma1_sq = filter2(img1 * img1, window, mode='reflect') - mu1_sq
    sigma2_sq = filter2(img2 * img2, window, mode='reflect') - mu2_sq
    sigma12   = filter2(img1 * img2, window, mode='reflect') - mu1_mu2

    # 数值安全
    sigma1_sq = np.maximum(sigma1_sq, 0.0)
    sigma2_sq = np.maximum(sigma2_sq, 0.0)

    # 常数
    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2

    # 亮度项 l
    l = (2.0 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)

    # 对比度+结构项 cs
    cs = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    l_mean  = l.mean()
    cs_mean = cs.mean()

    return l_mean, cs_mean

def _downsample(img):
    """
    2x2 平均池化下采样一倍（高度和宽度各除以 2）
    """
    h, w = img.shape
    h -= h % 2
    w -= w % 2
    img = img[:h, :w]

    return (img[0:h:2, 0:w:2] + img[1:h:2, 0:w:2] +
            img[0:h:2, 1:w:2] + img[1:h:2, 1:w:2]) / 4.0

def compute_ms_ssim(img1, img2, max_val=80.0):
    """
    计算 MS-SSIM (Multi-Scale Structural Similarity)
    img1, img2: 2D numpy arrays (float), 单通道 depth
    max_val    : 数据范围上界（例如 80.0，对应 MAX_DEPTH）
    返回:
        一个标量，范围大致在 [0,1]（两图越相似越接近 1）
    """
    # Wang et al. 2003 中常用的一组权重
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float64)
    levels = weights.size

    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)

    window = matlab_style_gauss2D((11, 11), 1.5)

    mssim = []
    mcs   = []

    for _ in range(levels):
        # l_mean: 亮度; cs_mean: 对比度*结构
        l_mean, cs_mean = _ssim_for_ms(im1, im2, max_val=max_val, window=window)
        # 单尺度 SSIM = l * cs
        mssim.append(l_mean * cs_mean)
        mcs.append(cs_mean)

        # 除了最后一层都要下采样
        im1 = _downsample(im1)
        im2 = _downsample(im2)

    mssim = np.array(mssim)
    mcs   = np.array(mcs)

    # 标准 MS-SSIM：
    #   MS-SSIM = ( Π_{j=1}^{M-1} mcs_j^{w_j} ) * (mssim_M^{w_M})
    return (np.prod(mcs[:-1] ** weights[:-1]) *
            (mssim[-1] ** weights[-1]))

# =======================================================

def compute_metrics(pred_path, gt_path, mode):
    # 1. 读取图片
    pred_img = cv2.imread(pred_path, -1)
    gt_img   = cv2.imread(gt_path, -1)

    if pred_img is None:
        return None
    if gt_img is None:
        return None

    # 2. 对齐尺寸
    if pred_img.shape != gt_img.shape:
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    # 3. 转换为米 (假设输入是 16bit 单通道 depth)
    depth_pred = pred_img.astype(np.float32) / PRED_SCALE_FACTOR
    depth_gt   = gt_img.astype(np.float32) / GT_SCALE_FACTOR

    # 4. 有效 Mask (用于 Errors 计算)
    mask = (depth_gt > MIN_DEPTH) & (depth_gt < MAX_DEPTH)

    # 结果字典
    results = {}

    # === 计算 ERROR 指标 ===
    if mode in ['Errors', 'MetricsAll']:
        if not np.any(mask):
            return None  # 如果没有有效像素，跳过

        valid_pred = depth_pred[mask]
        valid_gt   = depth_gt[mask]

        eps = 1e-8
        diff     = valid_pred - valid_gt
        diff_abs = np.abs(diff)
        diff_sqr = diff ** 2

        results["MAE"]  = diff_abs.mean()
        results["RMSE"] = np.sqrt(diff_sqr.mean())
        results["REL"]  = (diff_abs / (valid_gt + eps)).mean()

        # 逆深度
        inv_pred = INV_SCALE / (valid_pred + eps)
        inv_gt   = INV_SCALE / (valid_gt + eps)
        diff_inv = inv_pred - inv_gt
        results["iMAE"]  = np.mean(np.abs(diff_inv))
        results["iRMSE"] = np.sqrt(np.mean(diff_inv ** 2))

        # Deltas
        r1 = valid_gt / (valid_pred + eps)
        r2 = valid_pred / (valid_gt + eps)
        ratio = np.maximum(r1, r2)
        results["D1"]   = np.mean(ratio < 1.25)
        results["D2"]   = np.mean(ratio < 1.25 ** 2)
        results["D3"]   = np.mean(ratio < 1.25 ** 3)
        results["D102"] = np.mean(ratio < 1.02)
        results["D105"] = np.mean(ratio < 1.05)
        results["D110"] = np.mean(ratio < 1.10)

    # === 计算 IQA 指标 (SSIM, MS-SSIM, PSNR) ===
    if mode in ['IQA', 'MetricsAll']:
        # PSNR 基于有效像素计算 MSE
        if np.any(mask):
            valid_mse = np.mean((depth_pred[mask] - depth_gt[mask]) ** 2)
            if valid_mse == 0:
                results["PSNR"] = 100.0  # 完美匹配
            else:
                # 峰值取 MAX_DEPTH (80m)
                results["PSNR"] = 20 * np.log10(MAX_DEPTH / np.sqrt(valid_mse))
        else:
            results["PSNR"] = 0.0

        # 深度图中的无效值通常很小或很大，这里我们限制在 [0, MAX_DEPTH]
        img_pred_safe = np.clip(depth_pred, 0, MAX_DEPTH)
        img_gt_safe   = np.clip(depth_gt, 0, MAX_DEPTH)

        # SSIM (skimage，单尺度)
        try:
            val_ssim = ssim_func(img_gt_safe, img_pred_safe, data_range=MAX_DEPTH)
        except ValueError:
            # 如果图像太小 (小于7x7)，SSIM 可能会报错
            val_ssim = 0.0
        results["SSIM"] = val_ssim

        # MS-SSIM (多尺度)
        try:
            val_msssim = compute_ms_ssim(img_gt_safe, img_pred_safe, max_val=MAX_DEPTH)
        except Exception as e:
            print(f"MS-SSIM 计算出错: {e}")
            val_msssim = 0.0
        results["MS-SSIM"] = val_msssim

    return results

def main():
    args = get_args()
    mode = args.mode

    # 获取文件列表并排序
    pred_files = sorted(os.listdir(PRED_FOLDER))
    gt_files   = sorted(os.listdir(GT_FOLDER))

    pred_files = [f for f in pred_files if f.endswith(('.png', '.jpg', '.tif'))]
    gt_files   = [f for f in gt_files if f.endswith(('.png', '.jpg', '.tif'))]

    if len(pred_files) != len(gt_files):
        print("Warning: Files mismatch! Using intersection...")
        num_files = min(len(pred_files), len(gt_files))
    else:
        num_files = len(pred_files)

    print(f"Found {num_files} pairs. Mode: {mode}")

    # 收集 Metrics
    all_metrics = {}

    valid_count = 0

    print("Starting evaluation...")
    for i in tqdm(range(num_files)):
        p_path = os.path.join(PRED_FOLDER, pred_files[i])
        g_path = os.path.join(GT_FOLDER, gt_files[i])

        res = compute_metrics(p_path, g_path, mode)
        if res is None:
            continue

        valid_count += 1

        # 动态初始化 list
        for k, v in res.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)

    if valid_count == 0:
        print("No valid data found.")
        return

    # 计算平均值
    avg = {k: float(np.mean(v)) for k, v in all_metrics.items()}

    print("\n" + "=" * 40)
    print(f"Evaluation Results ({valid_count} images) | Mode: {mode}")
    print("=" * 40)

    # === 输出 Errors ===
    if mode in ['Errors', 'MetricsAll']:
        print("--- Depth Errors ---")
        print(f"RMSE (mm)     : {avg['RMSE']:.5f}")
        print(f"MAE  (mm)     : {avg['MAE']:.5f}")
        unit_inv = "1/km" if INV_SCALE == 1000.0 else "1/m"
        print(f"iRMSE ({unit_inv}): {avg['iRMSE']:.4f}")
        print(f"iMAE  ({unit_inv}): {avg['iMAE']:.4f}")
        print(f"REL           : {avg['REL']:.4f}")
        print("-" * 20)
        print(f"D^1   (<1.25)   : {avg['D1'] * 100:.2f}%")
        print(f"D^2   (<1.25^2) : {avg['D2'] * 100:.2f}%")
        print(f"D^3   (<1.25^3) : {avg['D3'] * 100:.2f}%")
        print(f"D102 (<1.02)    : {avg['D102'] * 100:.2f}%")
        print(f"D105 (<1.05)    : {avg['D105'] * 100:.2f}%")
        print(f"D110 (<1.10)    : {avg['D110'] * 100:.2f}%")
        print("\n")

    # === 输出 IQA ===
    if mode in ['IQA', 'MetricsAll']:
        print("--- Image Quality (IQA) ---")
        print(f"SSIM          : {avg['SSIM']:.4f}")
        print(f"MS-SSIM       : {avg['MS-SSIM']:.4f}")
        print(f"PSNR (dB)     : {avg['PSNR']:.4f}")

    print("=" * 40)
    print(f"Note: Scaling used -> Pred: /{PRED_SCALE_FACTOR}, GT: /{GT_SCALE_FACTOR}")
    if mode in ['Errors', 'MetricsAll']:
        unit_inv = "1/km" if INV_SCALE == 1000.0 else "1/m"
        print(f"      Inverse depth scale INV_SCALE = {INV_SCALE} (unit: {unit_inv})")

if __name__ == "__main__":
    main()
