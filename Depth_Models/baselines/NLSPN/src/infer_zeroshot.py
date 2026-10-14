"""Thin zero-shot inference for NLSPN (KITTI-DC pretrained) on our cropped splits.

Reads image/ + velodyne_raw/ directly (no KITTI dataset class). Output:
uint16 PNG = round(metres*256), 352x1216, named EXACTLY like the GT file.
Only rgb + sparse dep are used by the model forward (K is unused).

Fast IO: raw open().read() + cv2.imdecode (bypasses on-access AV scanner slowdown),
prefetched with a ThreadPoolExecutor; raw-write the encoded PNG too.
"""
import os
import sys
import glob
import argparse
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.nlspnmodel import NLSPNModel

cv2.setNumThreads(0)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_margs():
    a = argparse.Namespace()
    a.network = 'resnet34'
    a.from_scratch = False
    a.prop_time = 18
    a.prop_kernel = 3
    a.preserve_input = True   # README KITTI eval config
    a.affinity = 'TGASS'
    a.affinity_gamma = 0.5
    a.conf_prop = True
    a.legacy = True           # README KITTI eval config
    a.lr = 0.001
    return a


def fast_read_bytes(path):
    with open(path, 'rb') as f:
        return f.read()


def load_sample(img_path, velo_path):
    img_buf = fast_read_bytes(img_path)
    velo_buf = fast_read_bytes(velo_path)
    bgr = cv2.imdecode(np.frombuffer(img_buf, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD                      # HWC
    rgb = np.transpose(rgb, (2, 0, 1))            # CHW
    velo = cv2.imdecode(np.frombuffer(velo_buf, np.uint8), cv2.IMREAD_UNCHANGED)
    dep = (velo.astype(np.float32) / 256.0)[None]  # 1HW metres
    return rgb, dep


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gt_dir', required=True)
    p.add_argument('--image_dir', required=True)
    p.add_argument('--velo_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    net = NLSPNModel(build_margs()).cuda().eval()
    ck = torch.load(args.ckpt, map_location='cpu')
    sd = ck['net'] if 'net' in ck else ck
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"loaded ckpt. missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print("  missing:", missing[:8], flush=True)
    if unexpected:
        print("  unexpected:", unexpected[:8], flush=True)

    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, '*.png')))
    n = len(gt_files)
    print(f"{n} GT files -> {args.out_dir}", flush=True)

    jobs = []
    for gt_path in gt_files:
        gt_name = os.path.basename(gt_path)
        img_name = gt_name.replace('groundtruth_depth', 'image')
        velo_name = gt_name.replace('groundtruth_depth', 'velodyne_raw')
        img_path = os.path.join(args.image_dir, img_name)
        velo_path = os.path.join(args.velo_dir, velo_name)
        assert os.path.exists(img_path), img_path
        assert os.path.exists(velo_path), velo_path
        jobs.append((gt_name, img_path, velo_path))

    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = {}
    PREFETCH = args.workers * 2

    def submit(i):
        _, ip, vp = jobs[i]
        futures[i] = pool.submit(load_sample, ip, vp)

    for i in range(min(PREFETCH, n)):
        submit(i)

    write_pool = ThreadPoolExecutor(max_workers=8)

    def write_out(out_name, arr16):
        ok, enc = cv2.imencode('.png', arr16)
        with open(os.path.join(args.out_dir, out_name), 'wb') as f:
            f.write(enc.tobytes())

    done = 0
    for i in range(n):
        rgb, dep = futures.pop(i).result()
        nxt = i + PREFETCH
        if nxt < n:
            submit(nxt)
        rgb_t = torch.from_numpy(rgb).unsqueeze(0).cuda(non_blocking=True)
        dep_t = torch.from_numpy(dep).unsqueeze(0).cuda(non_blocking=True)
        with torch.no_grad():
            out = net({'rgb': rgb_t, 'dep': dep_t})
        pred = out['pred'].squeeze().detach().float().cpu().numpy()
        pred = np.clip(pred, 0, None)
        out16 = np.round(pred * 256.0).astype(np.uint16)
        write_pool.submit(write_out, jobs[i][0], out16)
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{n}", flush=True)

    pool.shutdown(wait=True)
    write_pool.shutdown(wait=True)
    print(f"DONE {done} -> {args.out_dir}", flush=True)


if __name__ == '__main__':
    main()
