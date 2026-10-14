import os
import cv2
import numpy as np
import pandas as pd
from typing import Optional, Tuple

# ✅ 新增功能：

# add_colorbar 函数在可视化图右侧添加颜色条。

# overlay 可视化时颜色条也会显示。

# 保留原有 CSV 模式和目录遍历模式。

# 这样，你生成的 overlay 图旁边就会有对应深度的颜色条，便于分析和展示深度分布。

# -------------------- 可视化核心函数 --------------------
MAX_DEPTH_FIXED = 85 * 256  # 固定深度上限

def depth_to_color(depth, max_depth=MAX_DEPTH_FIXED, colormap=cv2.COLORMAP_JET):
    """
    将 16bit 单通道深度图转为伪彩色 (BGR)，颜色映射固定最大深度。
    """
    depth_norm = np.clip(depth.astype(np.float32) / max_depth * 255.0, 0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, colormap)
    return depth_color

def overlay_depth(image, depth_color, alpha=0.6):
    return cv2.addWeighted(image, 1 - alpha, depth_color, alpha, 0)

def depth_points_hsv(depth, image=None, max_depth=MAX_DEPTH_FIXED, point_size=1):
    """
    将深度点映射到 HSV 色相，绘制在底图上（或黑底）。
    """
    H, W = depth.shape
    if image is None:
        hsv = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = np.clip(depth.astype(np.float32) / max_depth * 179.0, 0, 179).astype(np.uint8)
    ys, xs = np.where(depth > 0)
    for y, x in zip(ys, xs):
        h = int(hue[y, x])
        cv2.circle(hsv, (x, y), point_size, (h, 255, 255), -1)
    bgr_output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr_output

def ensure_match_size(depth_u16: np.ndarray, target_hw: tuple) -> np.ndarray:
    th, tw = target_hw
    if depth_u16.shape[:2] == (th, tw):
        return depth_u16
    return cv2.resize(depth_u16, (tw, th), interpolation=cv2.INTER_NEAREST)

# -------------------- 新增函数：添加 Colorbar --------------------
def add_colorbar(vis_image, max_depth=MAX_DEPTH_FIXED, colormap=cv2.COLORMAP_JET, width=50):
    """
    在可视化图像右侧增加一个颜色条
    """
    h = vis_image.shape[0]
    gradient = np.linspace(max_depth, 0, h).astype(np.float32)  # 从上到下深度值
    gradient = gradient[:, np.newaxis]  # shape=(H,1)
    gradient_color = depth_to_color(gradient, max_depth=max_depth, colormap=colormap)
    gradient_color = cv2.resize(gradient_color, (width, h), interpolation=cv2.INTER_NEAREST)
    # 拼接图像
    vis_with_colorbar = np.concatenate([vis_image, gradient_color], axis=1)
    return vis_with_colorbar

# -------------------- 用户配置 --------------------
BASE_PATH = "/path/to/aurora-root/Dataset/AuroraKITTI_Tools/Datasets/Split_Flattened_Dataset"
SPLIT = "val"  # train / val / test

DEPTH_DIR = os.path.join(BASE_PATH, "16bit_proj_depth", SPLIT)
FRAME_DIR = os.path.join(BASE_PATH, "frame_image", SPLIT)
OUT_DIR = os.path.join(BASE_PATH, "visualised_depth", SPLIT)

METHOD = "overlay"  # "overlay" 或 "hsvpoints"
ALPHA = 0.5
POINT_SIZE = 1
COLORMAP = cv2.COLORMAP_JET

USE_CSV_MODE = True
CSV_PATH = os.path.join(BASE_PATH, f"{SPLIT}.csv")
WEATHER_FILTER = "Rain"
CSV_LIMIT = 20

# -------------------- 辅助函数 --------------------
def image_name_from_depth(depth_fname: str) -> str:
    return depth_fname.replace("proj_depth", "image")

def depth_name_from_image(img_fname_no_ext: str) -> str:
    return img_fname_no_ext.replace("image_", "proj_depth_") + ".png"

# -------------------- 遍历目录可视化 --------------------
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

        # if METHOD == "overlay":
        #     depth_color = depth_to_color(depth_u16, max_depth=MAX_DEPTH_FIXED, colormap=COLORMAP)
        #     vis = overlay_depth(image_bgr, depth_color, alpha=ALPHA)
        #     vis = add_colorbar(vis, max_depth=MAX_DEPTH_FIXED, colormap=COLORMAP)
        # else:
        vis = depth_points_hsv(depth_u16, image=image_bgr, max_depth=MAX_DEPTH_FIXED, point_size=POINT_SIZE)

        out_name = fname.replace("proj_depth", "visualised_depth")
        out_path = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(out_path, vis)
        processed += 1
        print(f"✅ Saved: {out_name}")

    print(f"\nDone. Processed: {processed}, Missing/Skipped: {missing}")

# -------------------- CSV 可视化 --------------------
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
        img_fname = name + ".png"
        depth_path = os.path.join(DEPTH_DIR, depth_fname)
        img_path = os.path.join(FRAME_DIR, img_fname)

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

        # if METHOD == "overlay":
        #     depth_color = depth_to_color(depth_u16, max_depth=MAX_DEPTH_FIXED, colormap=COLORMAP)
        #     vis = overlay_depth(image_bgr, depth_color, alpha=ALPHA)
        #     vis = add_colorbar(vis, max_depth=MAX_DEPTH_FIXED, colormap=COLORMAP)
        # else:
        vis = depth_points_hsv(depth_u16, image=image_bgr, max_depth=MAX_DEPTH_FIXED, point_size=POINT_SIZE)

        out_name = depth_fname.replace("proj_depth", "visualised_depth")
        out_path = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(out_path, vis)
        processed += 1
        print(f"✅ Saved (CSV): {out_name}")

    print(f"\n[CSV mode] Done. Processed: {processed}, Missing/Skipped: {missing}")

# -------------------- 主函数 --------------------
def main():
    if USE_CSV_MODE:
        visualize_from_csv(CSV_PATH, weather=WEATHER_FILTER, limit=CSV_LIMIT)
    else:
        visualize_all_from_depthdir()

if __name__ == "__main__":
    main()
