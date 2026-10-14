import os
import cv2
import numpy as np
import pandas as pd
import importlib
from tqdm import tqdm
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004

class WeatherPathReader:
    def __init__(self, csv_path, root_dir, split):
        self.df    = pd.read_csv(csv_path)
        self.root  = root_dir.rstrip('/')  # 用于后面 relativize
        self.split = split
        self.modality_map = {
            'image':     ('frame_image',      '.png'),
            '8bitdepth': ('8bit_clean_depth','.png'),
        }

    def build_image_path(self, name):
        folder, ext = self.modality_map['image']
        return os.path.join(self.root, folder, self.split, f"{name}{ext}")

    def build_depth_path(self, name):
        # name format: seq_sync_image_XYZ.png base without extension
        seq, tail = name.split('_sync_')
        # replace only the first occurrence of 'image' with '8bitdepth'
        depth_tail = tail.replace('image', '8bitdepth', 1)
        filename = f"{seq}_sync_{depth_tail}.png"
        folder, ext = self.modality_map['8bitdepth']
        return os.path.join(self.root, folder, self.split, filename)

if __name__ == '__main__':
    # —— 配置区 —— 
    SPLIT      = 'val'    # train / val / test
    OUTPUT_DIR = '/path/to/aurora-root/Code/Cam_Weather_perf-optimization_numba/output_snow_light'  # 新的输出根目录
    MAX_DEPTH  = 80.0    # 8-bit 深度值 [0,255] 映射到 [0, MAX_DEPTH] 米

    CSV_PATH = f"/path/to/aurora-root/Dataset/AuroraDepth/Datasets/{SPLIT}.csv"
    ROOT_DIR = "/path/to/aurora-root/Dataset/AuroraDepth/Datasets/Split_Flattened_Dataset"

    # 读取 CSV 并过滤 moderate–Snow 条目
    df = pd.read_csv(CSV_PATH)
    df_mod = df[(df.weather=='Snow') & (df.severity=='light')]

    total = len(df_mod)
    print(f"🔍 CSV 中 moderate–Snow 总条目数：{total}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generated = 0

    for name in tqdm(df_mod['name'], desc="Generating snowfall", unit="img"):
        # 1) 彩色图路径
        img_path = WeatherPathReader(CSV_PATH, ROOT_DIR, SPLIT).build_image_path(name)
        colour   = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if colour is None:
            print(f"[跳过] 找不到彩色图：{img_path}")
            continue

        # 2) 深度图路径（替换 image -> 8bitdepth）
        depth_path = WeatherPathReader(CSV_PATH, ROOT_DIR, SPLIT).build_depth_path(name)
        norm_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if norm_depth is None:
            print(f"[跳过] 找不到深度图：{depth_path}")
            continue
        if norm_depth.ndim == 3:
            norm_depth = cv2.cvtColor(norm_depth, cv2.COLOR_BGR2GRAY)
        depth = norm_depth.astype(np.float32) / 255.0 * MAX_DEPTH

        # 3) Bayer 转换与落雪合成
        B, G, R = cv2.split(colour)
        h, w = colour.shape[:2]
        bay = np.empty((h, w), np.uint8)
        bay[0::2,0::2] = R[0::2,0::2]
        bay[1::2,0::2] = G[1::2,0::2]
        bay[0::2,1::2] = G[0::2,1::2]
        bay[1::2,1::2] = B[1::2,1::2]

        weather_class = getattr(importlib.import_module("noise_class"), 'cam_snow')
        noise_class   = getattr(importlib.import_module("noise_class"), 'cam_wind')
        noise = noise_class(res=np.array([w,h]), weather_rate=1)
        out_b = noise.application(bay, depth, 'cam_snow')
        rgb = demosaicing_CFA_Bayer_Malvar2004(out_b)
        out_c = cv2.cvtColor(np.clip(rgb,0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # 4) 保存到 OUTPUT_DIR，保持相对路径结构
        rel = os.path.relpath(img_path, start=ROOT_DIR)
        out_path = os.path.join(OUTPUT_DIR, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, out_c)
        generated += 1

    # 结果校验
    print(f"🎉 生成完毕：CSV 条目={total}, 实际生成={generated}")
    if generated == total:
        print("✅ 数量一致，未遗漏！")
    else:
        print("⚠️ 数量不符，请检查跳过的条目。")
