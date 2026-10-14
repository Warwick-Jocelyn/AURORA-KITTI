import os
if not ("DISPLAY" in os.environ):
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

cmap = plt.cm.jet

import numpy as np

def _ensure_3ch(img):
    # img: numpy array, shape (H,W) or (H,W,C)
    if img.ndim == 2:
        # 灰度 -> 3通道
        img = np.repeat(img[..., None], 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 4:
        # RGBA -> RGB
        img = img[..., :3]
    return img

def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth.astype('uint8')


# def merge_into_row(ele, pred):
#     def preprocess_depth(x):
#         y = np.squeeze(x.data.cpu().numpy())
#         return depth_colorize(y)

#     # if is gray, transforms to rgb
#     # img_list = []
#     img_list = [_ensure_3ch(x) for x in img_list]
#     if 'rgb' in ele:
#         rgb = np.squeeze(ele['rgb'][0, ...].data.cpu().numpy())
#         rgb = np.transpose(rgb, (1, 2, 0))
#         img_list.append(rgb)
#     elif 'g' in ele:
#         g = np.squeeze(ele['g'][0, ...].data.cpu().numpy())
#         g = np.array(Image.fromarray(g).convert('RGB'))
#         img_list.append(g)
#     if 'd' in ele:
#         img_list.append(preprocess_depth(ele['d'][0, ...]))
#     img_list.append(preprocess_depth(pred[0, ...]))
#     if 'gt' in ele:
#         img_list.append(preprocess_depth(ele['gt'][0, ...]))

#     img_merge = np.hstack(img_list)
#     # return img_merge.astype('uint8')
#     return img_merge

# vis_utils.py 修正版
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

cmap = plt.cm.jet

def depth_colorize(depth):
    """
    将深度图标准化并映射到彩色
    depth: 2D numpy array (H, W)
    return: 3通道RGB图 (H, W, 3)
    """
    if np.max(depth) == np.min(depth):
        depth_normalized = np.zeros_like(depth)
    else:
        depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth_rgb = (255 * cmap(depth_normalized)[:, :, :3]).astype(np.uint8)
    return depth_rgb

def to_rgb(img):
    """
    强制将图像转换为3通道RGB
    img: numpy array, 2D or 3D
    """
    if img is None:
        return None
    img = np.squeeze(img)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    return img

def merge_into_row(ele, pred):
    """
    将 RGB/灰度/Depth/Pred/GT 合并成一行图像用于可视化
    ele: dict, 可能包含 'rgb', 'g', 'd', 'gt'
    pred: tensor, 模型预测 depth
    """
    img_list = []

    # RGB 或灰度
    if 'rgb' in ele and ele['rgb'] is not None:
        rgb = to_rgb(np.transpose(np.squeeze(ele['rgb'][0, ...].data.cpu().numpy()), (1, 2, 0)))
        img_list.append(rgb)
    elif 'g' in ele and ele['g'] is not None:
        g = np.array(Image.fromarray(np.squeeze(ele['g'][0, ...].data.cpu().numpy())).convert('RGB'))
        g = to_rgb(g)
        img_list.append(g)

    # Sparse depth
    if 'd' in ele and ele['d'] is not None:
        d = np.squeeze(ele['d'][0, ...].data.cpu().numpy())
        if np.any(d > 0):
            d_colored = depth_colorize(d)
            d_colored = to_rgb(d_colored)
            img_list.append(d_colored)

    # Model prediction
    if pred is not None:
        p = np.squeeze(pred[0, ...].data.cpu().numpy())
        if np.any(p > 0):
            p_colored = depth_colorize(p)
            p_colored = to_rgb(p_colored)
            img_list.append(p_colored)

    # Ground truth
    if 'gt' in ele and ele['gt'] is not None:
        gt = np.squeeze(ele['gt'][0, ...].data.cpu().numpy())
        if np.any(gt > 0):
            gt_colored = depth_colorize(gt)
            gt_colored = to_rgb(gt_colored)
            img_list.append(gt_colored)

    if len(img_list) == 0:
        # 防止空列表 crash
        return np.zeros((10, 10, 3), dtype=np.uint8)

    # 合并成一行
    img_merge = np.hstack(img_list)
    return img_merge

def add_row(img_merge, row):
    return np.vstack([img_merge, row])

def save_image(img_merge, filename):
    image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)

def save_depth_as_uint16png(img, filename):
    img = (img * 256).astype('uint16')
    cv2.imwrite(filename, img)



def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)


def save_depth_as_uint16png(img, filename):
    img = (img * 256).astype('uint16')
    cv2.imwrite(filename, img)


if ("DISPLAY" in os.environ):
    f, axarr = plt.subplots(4, 1)
    plt.tight_layout()
    plt.ion()


def display_warping(rgb_tgt, pred_tgt, warped):
    def preprocess(rgb_tgt, pred_tgt, warped):
        rgb_tgt = 255 * np.transpose(np.squeeze(rgb_tgt.data.cpu().numpy()),
                                     (1, 2, 0))  # H, W, C
        # depth = np.squeeze(depth.cpu().numpy())
        # depth = depth_colorize(depth)

        # convert to log-scale
        pred_tgt = np.squeeze(pred_tgt.data.cpu().numpy())
        # pred_tgt[pred_tgt<=0] = 0.9 # remove negative predictions
        # pred_tgt = np.log10(pred_tgt)

        pred_tgt = depth_colorize(pred_tgt)

        warped = 255 * np.transpose(np.squeeze(warped.data.cpu().numpy()),
                                    (1, 2, 0))  # H, W, C
        recon_err = np.absolute(
            warped.astype('float') - rgb_tgt.astype('float')) * (warped > 0)
        recon_err = recon_err[:, :, 0] + recon_err[:, :, 1] + recon_err[:, :, 2]
        recon_err = depth_colorize(recon_err)
        return rgb_tgt.astype('uint8'), warped.astype(
            'uint8'), recon_err, pred_tgt

    rgb_tgt, warped, recon_err, pred_tgt = preprocess(rgb_tgt, pred_tgt,
                                                      warped)

    # 1st column
    column = 0
    axarr[0].imshow(rgb_tgt)
    axarr[0].axis('off')
    axarr[0].axis('equal')
    # axarr[0, column].set_title('rgb_tgt')

    axarr[1].imshow(warped)
    axarr[1].axis('off')
    axarr[1].axis('equal')
    # axarr[1, column].set_title('warped')

    axarr[2].imshow(recon_err, 'hot')
    axarr[2].axis('off')
    axarr[2].axis('equal')
    # axarr[2, column].set_title('recon_err error')

    axarr[3].imshow(pred_tgt, 'hot')
    axarr[3].axis('off')
    axarr[3].axis('equal')
    # axarr[3, column].set_title('pred_tgt')

    # plt.show()
    plt.pause(0.001)
