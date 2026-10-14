import os
import re
import glob
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import imageio.v2 as imageio

import torch
import torch.nn.functional as F
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
DATE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}$")


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    paths = []
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext in IMG_EXTS:
            paths.append(p)
    return sorted(paths)


def has_complete_depth(data_path, out_path):
    imgs = list_images(data_path)
    if not imgs:
        return False
    img_stems = {Path(p).stem for p in imgs}
    out_files = glob.glob(os.path.join(out_path, "*"))
    out_stems = {Path(p).stem for p in out_files}
    missing = img_stems - out_stems
    if missing:
        print(f"ℹ️  Incomplete: {out_path} missing {len(missing)} files.")
        return False
    return True


def parse_kitti_calib_cam_to_cam(calib_file):
    data = {}
    with open(calib_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            vals = np.fromstring(val.strip(), sep=" ", dtype=np.float64)
            data[key.strip()] = vals
    return data


def get_K_from_calib(calib_dict, cam_id):
    p_key = f"P_rect_{cam_id}"
    k_key = f"K_{cam_id}"

    if p_key in calib_dict and calib_dict[p_key].size == 12:
        P = calib_dict[p_key].reshape(3, 4)
        fx, fy, cx, cy = P[0, 0], P[1, 1], P[0, 2], P[1, 2]
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        return K

    if k_key in calib_dict and calib_dict[k_key].size == 9:
        return calib_dict[k_key].reshape(3, 3).astype(np.float32)

    raise KeyError(f"Cannot find {p_key} (12) or {k_key} (9) in calib_cam_to_cam.txt")


def chunker(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def find_date_root(root):
    """
    Return (date_root, date_dirs) where date_root contains YYYY_MM_DD folders.
    - If root itself has date folders, use root.
    - Else if root/raw has date folders, use root/raw.
    """
    root = os.path.abspath(root)

    def date_dirs_in(p):
        if not os.path.isdir(p):
            return []
        ds = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d)) and DATE_RE.match(d)]
        return sorted(ds)

    ds0 = date_dirs_in(root)
    if ds0:
        return root, ds0

    raw_candidate = os.path.join(root, "raw")
    ds1 = date_dirs_in(raw_candidate)
    if ds1:
        return raw_candidate, ds1

    # 你也可以在这里加更多候选，如 raw_used / raw_used_train 等
    for cand in ["raw_used", "raw_used_train", "raw_train"]:
        p = os.path.join(root, cand)
        ds = date_dirs_in(p)
        if ds:
            return p, ds

    return root, []


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("DA3 metric depth batch for KITTI raw -> /da3/ (auto root)")

    parser.add_argument("--root", type=str, required=True,
                        help="Dataset root. Can be a raw root (date folders) or a package root containing raw/")
    parser.add_argument("--calib-root", type=str, required=True,
                        help="Calib root containing folders like 2011_09_26_calib/calib_cam_to_cam.txt")
    parser.add_argument("--ckpt", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")

    parser.add_argument("--out-folder-name", type=str, default="da3",
                        help='Output folder under each camera folder (default "da3").')

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=0)

    parser.add_argument("--process-res-method", type=str, default="upper_bound_resize")
    parser.add_argument("--process-res", type=int, default=0,
                        help="0 => auto per batch max_side; >0 => fixed")
    parser.add_argument("--cap", type=int, default=0,
                        help="If >0 and process_res=0, process_res=min(max_side, cap)")

    parser.add_argument("--png-scale", type=float, default=256.0)
    parser.add_argument("--max-depth-m", type=float, default=255.0)

    parser.add_argument("--align-back-to-input", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-skip-if-exists", action="store_true")
    parser.add_argument("--verbose-shape", action="store_true")

    args = parser.parse_args()
    skip_if_exists = not args.no_skip_if_exists

    if args.out_folder_name.lower() in {"disp", "dis"}:
        raise ValueError('out-folder-name="disp/dis" may overwrite your DA2 results. Use --out-folder-name da3')

    # Auto-detect date root
    date_root, date_dirs = find_date_root(args.root)
    print(f"[INFO] root={os.path.abspath(args.root)}")
    print(f"[INFO] detected date_root={date_root}")
    print(f"[INFO] found {len(date_dirs)} date dirs")

    if not date_dirs:
        raise RuntimeError(
            f"No date folders (YYYY_MM_DD) found under {args.root} or its raw/ candidates. "
            f"Your root structure may differ."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(args.ckpt).to(device=device).eval()
    print(f"[DA3] loaded {args.ckpt} on {device}")

    calib_cache = {}

    for date_dir in date_dirs:
        date_path = os.path.join(date_root, date_dir)

        calib_folder = os.path.join(args.calib_root, f"{date_dir}_calib")
        calib_file = os.path.join(calib_folder, "calib_cam_to_cam.txt")
        if not os.path.isfile(calib_file):
            print(f"[skip] missing calib_cam_to_cam.txt for date {date_dir}: {calib_file}")
            continue

        if date_dir not in calib_cache:
            calib_cache[date_dir] = parse_kitti_calib_cam_to_cam(calib_file)

        drives = [d for d in sorted(os.listdir(date_path))
                  if d.endswith("_sync") and os.path.isdir(os.path.join(date_path, d))]

        for drive_dir in drives:
            drive_path = os.path.join(date_path, drive_dir)

            for cam in ["image_02", "image_03"]:
                cam_id = "02" if cam == "image_02" else "03"
                data_path = os.path.join(drive_path, cam, "data")
                if not os.path.isdir(data_path):
                    continue

                out_path = os.path.join(drive_path, cam, args.out_folder_name)
                os.makedirs(out_path, exist_ok=True)

                if skip_if_exists and has_complete_depth(data_path, out_path):
                    print(f"⏭️  Skip (already complete): {data_path} -> {out_path}")
                    continue

                img_paths = list_images(data_path)
                if args.max_images and args.max_images > 0:
                    img_paths = img_paths[:args.max_images]
                if not img_paths:
                    continue

                K_cam = get_K_from_calib(calib_cache[date_dir], cam_id)

                pbar = tqdm(total=len(img_paths), desc=f"{date_dir}/{drive_dir}/{cam}")

                for batch_paths in chunker(img_paths, args.batch_size):
                    images, orig_hw = [], []
                    max_side = 0

                    for p in batch_paths:
                        img = Image.open(p).convert("RGB")
                        w, h = img.size
                        orig_hw.append((h, w))
                        images.append(img)
                        max_side = max(max_side, h, w)

                    if args.process_res > 0:
                        process_res = int(args.process_res)
                    else:
                        process_res = int(max_side)
                        if args.cap and args.cap > 0:
                            process_res = min(process_res, int(args.cap))

                    intrinsics_np = np.repeat(K_cam[None, :, :], repeats=len(batch_paths), axis=0).astype(np.float32)

                    if device.type == "cuda" and args.amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            pred = model.inference(
                                image=images,
                                intrinsics=intrinsics_np,
                                process_res=process_res,
                                process_res_method=args.process_res_method,
                            )
                    else:
                        pred = model.inference(
                            image=images,
                            intrinsics=intrinsics_np,
                            process_res=process_res,
                            process_res_method=args.process_res_method,
                        )

                    depth = pred.depth
                    if isinstance(depth, np.ndarray):
                        depth_t = torch.from_numpy(depth).to(device=device, dtype=torch.float32)
                    else:
                        depth_t = depth.to(device=device, dtype=torch.float32)

                    if args.verbose_shape:
                        print(f"  process_res={process_res}, depth.shape={tuple(depth_t.shape)}")

                    out_depth_list = []
                    for i in range(depth_t.shape[0]):
                        h0, w0 = orig_hw[i]
                        di = depth_t[i:i+1, None]
                        if args.align_back_to_input and (di.shape[-2] != h0 or di.shape[-1] != w0):
                            di = F.interpolate(di, size=(h0, w0), mode="bilinear", align_corners=False)
                        out_depth_list.append(di[0, 0])

                    out_depth = torch.stack(out_depth_list, dim=0)

                    for i, p in enumerate(batch_paths):
                        name = Path(p).name
                        out_png = os.path.join(out_path, name)

                        dm = out_depth[i].detach().cpu().numpy().astype(np.float32)
                        dm = np.clip(dm, 0.0, args.max_depth_m)
                        u16 = np.clip(dm * args.png_scale, 0, 65535).astype(np.uint16)
                        imageio.imwrite(out_png, u16)

                    pbar.update(len(batch_paths))

                pbar.close()

    print("🎉 All done.")


if __name__ == "__main__":
    main()
