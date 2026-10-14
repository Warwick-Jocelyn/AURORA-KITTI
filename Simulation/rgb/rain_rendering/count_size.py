import os
from collections import Counter
from PIL import Image

def count_image_resolutions(folder_path):
    """
    遍历 folder_path 下所有图片文件，统计分辨率（宽×高）出现的次数，并记录每种分辨率的一个示例文件名。
    支持常见图片格式：.jpg, .jpeg, .png, .gif, .bmp, .tiff
    """
    resolutions = []
    examples = {}

    for root, _dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
                file_path = os.path.join(root, filename)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        res = f"{width}×{height}"
                        resolutions.append(res)
                        # 记录第一个遇到的示例文件名
                        if res not in examples:
                            examples[res] = filename
                except Exception as e:
                    print(f"无法处理图片 {file_path}：{e}")

    return Counter(resolutions), examples

if __name__ == "__main__":
    # 将下面的路径替换成你的图片文件夹路径
    folder = "/path/to/aurora-root/Code/rain-rendering/data/source/kitti/data_object/training/depth"
    counts, examples = count_image_resolutions(folder)
    
    print("图片分辨率统计结果：")
    for res, cnt in counts.most_common():
        example_name = examples.get(res, "")
        print(f"{res}：{cnt} 张，示例：{example_name}")
