"""
Thin zero-shot inference for CompletionFormer on AURORA(TEST)/DENSE crops.
Points directly at image/ velodyne_raw/ intrinsics/ folders (no KITTI json layout).
Output: uint16 PNG, value=round(metres*256), shape 352x1216, named like the GT file.
"""
import os, sys, argparse, glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from model.completionformer import CompletionFormer

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class A:  # minimal args namespace for model construction
    model_name = 'CompletionFormer'
    prop_time = 6
    prop_kernel = 3
    conf_prop = True
    affinity = 'TGASS'
    affinity_gamma = 0.5
    preserve_input = False
    legacy = False


def names_for(gt_name, split):
    """Return (image_name, velo_name, intr_name) for a given GT basename."""
    if split == 'TEST':
        img = gt_name.replace('groundtruth_depth', 'image')
        velo = gt_name.replace('groundtruth_depth', 'velodyne_raw')
        intr = img.replace('.png', '.txt')
    else:  # DENSE: identical basenames
        img = gt_name
        velo = gt_name
        intr = gt_name.replace('.png', '.txt')
    return img, velo, intr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['TEST', 'DENSE'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pretrain', required=True)
    args = ap.parse_args()

    gt_dir = os.path.join(args.root, 'groundtruth_depth')
    img_dir = os.path.join(args.root, 'image')
    velo_dir = os.path.join(args.root, 'velodyne_raw')
    intr_dir = os.path.join(args.root, 'intrinsics')
    os.makedirs(args.out, exist_ok=True)

    gt_names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(gt_dir, '*.png')))
    print('frames:', len(gt_names))

    model = CompletionFormer(A())
    ck = torch.load(args.pretrain, map_location='cpu')
    key_m, key_u = model.load_state_dict(ck['net'], strict=False)
    print('missing keys:', len(key_m), 'unexpected:', len(key_u))
    if key_m:
        print('  sample missing:', key_m[:5])
    model = model.cuda().eval()

    done = 0
    for gt_name in gt_names:
        img_n, velo_n, _ = names_for(gt_name, args.split)
        rgb = Image.open(os.path.join(img_dir, img_n)).convert('RGB')
        rgb = TF.to_tensor(rgb)
        rgb = TF.normalize(rgb, MEAN, STD)[None].cuda()

        velo = np.array(Image.open(os.path.join(velo_dir, velo_n))).astype(np.float32) / 256.0
        dep = torch.from_numpy(velo)[None, None].cuda()

        H, W = velo.shape
        assert (H, W) == (352, 1216), (gt_name, H, W)

        with torch.no_grad():
            out = model({'rgb': rgb, 'dep': dep})
        pred = out['pred'][0, 0].clamp(min=0).cpu().numpy()  # metres

        u16 = np.clip(np.round(pred * 256.0), 0, 65535).astype(np.uint16)
        Image.fromarray(u16, mode='I;16').save(os.path.join(args.out, gt_name))
        done += 1
        if done % 200 == 0:
            print(f'{done}/{len(gt_names)}')
    print('DONE', done)


if __name__ == '__main__':
    main()
