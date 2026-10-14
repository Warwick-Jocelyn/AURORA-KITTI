import os
import cv2
import numpy as np
import pandas as pd
from typing import Optional, Tuple

def depth_to_color(depth, max_depth=None, colormap=cv2.COLORMAP_JET):
    """
    将 16bit 单通道深度图转为伪彩色 (BGR)。

    参数:
      depth      -- np.ndarray, dtype=np.uint16, shape=(H, W)
      max_depth  -- 用于归一化的最大深度值；若为 None 则取 depth.max()
      colormap   -- OpenCV 的 colormap 枚举

    返回:
      depth_color -- np.ndarray, dtype=np.uint8, shape=(H, W, 3)
    """
    if max_depth is None:
        max_depth = float(depth.max())
    # 归一化到 [0,255]
    depth_norm = np.clip(depth.astype(np.float32) / max_depth * 255.0, 0, 255).astype(np.uint8)
    # 生成伪彩色
    depth_color = cv2.applyColorMap(depth_norm, colormap)
    return depth_color

def overlay_depth(image, depth_color, alpha=0.6):
    """
    将伪彩色深度图叠加到原始 BGR 图像上。

    参数:
      image        -- np.ndarray, dtype=uint8, shape=(H, W, 3)
      depth_color  -- 同上, shape=(H, W, 3)
      alpha        -- 深度图的权重 (0.0 - 1.0)
    """
    return cv2.addWeighted(image, 1 - alpha, depth_color, alpha, 0)


def depth_points_hsv(depth, image=None, max_depth=None, point_size=1):
    """
    把深度图当作一堆点，用 HSV 的 H 通道表示深度，投影到一张底图（可选）或纯黑背景上。

    参数:
      depth      -- np.ndarray, uint16, shape=(H, W)
      image      -- 可选底图 (BGR)，若为 None 则用黑底
      max_depth  -- 归一化最大深度
      point_size -- 绘制点的半径
    返回:
      bgr_output -- np.ndarray, uint8, shape=(H, W, 3)
    """
    H, W = depth.shape
    if max_depth is None:
        max_depth = float(depth.max())
    # 准备 HSV 图
    if image is None:
        hsv = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 归一化深度到 [0,179]，作为 hue 值
    hue = np.clip(depth.astype(np.float32) / max_depth * 179.0, 0, 179).astype(np.uint8)
    # 只处理 depth>0 的点
    ys, xs = np.where(depth > 0)
    for y, x in zip(ys, xs):
        h = int(hue[y, x])
        # (H, S, V) = (hue, max, max)
        cv2.circle(hsv, (x, y), point_size, (h, 255, 255), -1)
    # 转回 BGR
    bgr_output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr_output

def ensure_match_size(depth_u16: np.ndarray, target_hw: tuple) -> np.ndarray:
    th, tw = target_hw
    if depth_u16.shape[:2] == (th, tw):
        return depth_u16
    return cv2.resize(depth_u16, (tw, th), interpolation=cv2.INTER_NEAREST)

# ================= 用户可修改的配置 =================
BASE_PATH = "/path/to/aurora-root/Dataset/AuroraKITTI_Tools/Datasets/Split_Flattened_Dataset"
SPLIT = "val"   # train / val / test

# 输入目录
DEPTH_DIR = os.path.join(BASE_PATH, "16bit_proj_depth", SPLIT)
FRAME_DIR = os.path.join(BASE_PATH, "frame_image", SPLIT)
# 输出目录
OUT_DIR = os.path.join(BASE_PATH, "visualised_depth", SPLIT)

# 可视化方法: "overlay" 或 "hsvpoints"
METHOD = "overlay"
ALPHA = 0.5      # overlay 叠加权重
POINT_SIZE = 1   # hsvpoints 点大小
MAX_DEPTH = None # 归一化上限；None 自动取非零最大值
COLORMAP = cv2.COLORMAP_JET
# ===================================================

# ================= CSV 可视化模式（新） =================
USE_CSV_MODE   = True  # 想按 CSV + weather 的前 N 张图来可视化，就改 True
CSV_PATH       = f"/path/to/aurora-root/Dataset/AuroraKITTI_Tools/Datasets/{SPLIT}.csv"
WEATHER_FILTER = "Rain"  # "Clear"/"Fog"/"Snow"/"Rain"；设 None 不过滤
CSV_LIMIT      = 20     # 只取前 N 张
# ===================================================

def image_name_from_depth(depth_fname: str) -> str:
    return depth_fname.replace("proj_depth", "image")

def depth_name_from_image(img_fname_no_ext: str) -> str:
    return img_fname_no_ext.replace("image_", "proj_depth_") + ".png"

# ========= 遍历 16bit_proj_depth/{SPLIT} =========
def visualize_all_from_depthdir():
    os.makedirs(OUT_DIR, exist_ok=True)
    depth_files = sorted([f for f in os.listdir(DEPTH_DIR) if f.endswith(".png")])
    print(f"Found {len(depth_files)} depth files in: {DEPTH_DIR}")

    processed, missing = 0, 0
    for fname in depth_files:
        depth_path = os.path.join(DEPTH_DIR, fname)
        img_name = image_name_from_depth(fname)
        img_path = os.path.join(FRAME_DIR, img_name)

        if not (os.path.isfile(depth_path) and os.path.isfile(img_path)):
            print(f"[Skip] Missing pair for {fname}")
            missing += 1
            continue

        depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if depth_u16 is None or image_bgr is None:
            print(f"[Skip] Failed to read {fname}")
            missing += 1
            continue
        depth_u16 = ensure_match_size(depth_u16, image_bgr.shape[:2])

        if METHOD == "overlay":
            depth_color = depth_to_color(depth_u16, max_depth=MAX_DEPTH, colormap=COLORMAP)
        #     vis = overlay_depth(image_bgr, depth_color, alpha=ALPHA)
        # else:
            vis = depth_points_hsv(depth_u16, image=image_bgr, point_size=POINT_SIZE)
            # vis = depth_points_hsv(depth_u16, image=image_bgr, max_depth=MAX_DEPTH, point_size=POINT_SIZE)

        out_name = fname.replace("proj_depth", "visualised_depth")
        out_path = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(out_path, vis)
        processed += 1
        print(f"✅ Saved: {out_name}")

    print(f"\nDone. Processed: {processed}, Missing/Skipped: {missing}")

# ========= 新增：按 CSV + weather =========
def visualize_from_csv(csv_path: str,
                       weather: Optional[str] = None,
                       limit: int = 20) -> None:
    assert os.path.isfile(csv_path), f"CSV not found: {csv_path}"
    df = pd.read_csv(csv_path)
    if weather is not None and "weather" in df.columns:
        df = df[df["weather"] == weather]
    df = df[df["name"].astype(str).str.len() > 0].copy()
    if limit is not None and limit > 0:
        df = df.head(limit)

    os.makedirs(OUT_DIR, exist_ok=True)
    processed, missing = 0, 0
    for _, row in df.iterrows():
        name = str(row["name"])
        depth_fname = depth_name_from_image(name)
        img_fname   = name + ".png"
        depth_path = os.path.join(DEPTH_DIR, depth_fname)
        img_path   = os.path.join(FRAME_DIR, img_fname)

        if not (os.path.isfile(depth_path) and os.path.isfile(img_path)):
            print(f"[Skip CSV] Missing for {name}")
            missing += 1
            continue

        depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if depth_u16 is None or image_bgr is None:
            print(f"[Skip CSV] Failed read for {name}")
            missing += 1
            continue
        depth_u16 = ensure_match_size(depth_u16, image_bgr.shape[:2])

        if METHOD == "overlay":
            depth_color = depth_to_color(depth_u16)
        #     vis = overlay_depth(image_bgr, depth_color, alpha=ALPHA)
        # else:
        vis = depth_points_hsv(depth_u16, image=image_bgr, point_size=POINT_SIZE)

        out_name = depth_fname.replace("proj_depth", "visualised_depth")
        out_path = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(out_path, vis)
        processed += 1
        print(f"✅ Saved (CSV): {out_name}")

    print(f"\n[CSV mode] Done. Processed: {processed}, Missing/Skipped: {missing}")

def main():
    if USE_CSV_MODE:
        visualize_from_csv(CSV_PATH, weather=WEATHER_FILTER, limit=CSV_LIMIT)
    else:
        visualize_all_from_depthdir()

if __name__ == "__main__":
    main()
