import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def process_and_save_images(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # 遍历所有子目录及子子目录下的 PNG 文件
    for img_path in input_path.rglob("*.png"):
        # 读取16bit单通道图像
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        height, width = image.shape  # 获取原始图像尺寸
        
        # 进行膨胀操作增强图像
        kernel = np.ones((3, 3), np.uint8)
        enhanced_image = cv2.dilate(image, kernel, iterations=1)
        
        # 颜色范围最大值设为 51200
        max_val = 51200
        
        # 可视化增强后的图像，保持原始比例
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        y, x = np.where(enhanced_image > 0)
        colors = enhanced_image[enhanced_image > 0]
        sc1 = ax.scatter(x, -y, c=colors, cmap='jet', s=1, vmin=0, vmax=max_val)
        ax.set_xlim(0, width)
        ax.set_ylim(-height, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        cbar = fig.colorbar(sc1, fraction=0.046, pad=0.04)
        cbar.set_label('Depth Value')
        
        # 生成保存路径
        relative_path = img_path.relative_to(input_path)
        save_dir = output_path / relative_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / (img_path.stem + "_processed.png")
        
        # 保存可视化结果
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Processed and saved: {save_path}")

# 示例用法
input_path = "/path/to/aurora-root/Code/MUSES/Dataset/Sample_images/input"
output_path = "/path/to/aurora-root/Code/MUSES/Dataset/Sample_images/visualized"
process_and_save_images(input_path, output_path)