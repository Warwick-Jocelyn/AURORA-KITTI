# import torch

# import os, cv2
# from PIL import Image
# import numpy as np

# from depth_anything_3.api import DepthAnything3
# # from depth_anything_3.utils import visualize_depth

# from depth_anything_3.utils.visualize import visualize_depth

# # Load model from Hugging Face Hub
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = DepthAnything3.from_pretrained("depth-anything/da3nested-giant-large")

# model = model.to(device)
# model.eval()
# print(f"Model loaded on {device}!")

# model = model.to(device=device)

# # Run inference on images
# image_path = ["./examples/scene1_night_clear_1.png", "./examples/scene1_day_clear_0.png"]  # List of image paths, PIL Images, or numpy arrays
# prediction = model.inference(
#     image = image_path,
#     process_res=504,
#     process_res_method = "upper_bound_resize",
#     export_dir="./output",
#     export_format="glb"  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
# )

# # Access results
# print(prediction.depth.shape)        # Depth maps: [N, H, W] float32
# print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
# print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
# print(prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32

# demo.py
import os
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


def read_k33(txt_path: str) -> np.ndarray:
    """Read 3x3 intrinsics matrix from txt (9 floats)."""
    k = np.loadtxt(txt_path, dtype=np.float32).reshape(3, 3)
    return k


def list_images(image_dir: str, exts=("png", "jpg", "jpeg", "webp")):
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(image_dir, f"*.{e}"))
        paths += glob.glob(os.path.join(image_dir, f"*.{e.upper()}"))
    return sorted(paths)


def chunker(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("DA3 monocular metric depth -> uint16 PNG (meters*256)")

    # input/output
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--intrinsics-dir", type=str, required=True,
                        help="Folder containing per-image 3x3 intrinsics txt (same stem as image).")
    parser.add_argument("--out-dir", type=str, required=True)

    # model
    parser.add_argument("--ckpt", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp", action="store_true",
                        help="Use autocast mixed precision on CUDA (faster / less memory).")

    # resolution (the key you care about)
    parser.add_argument("--process-res", type=int, default=1216,
                        help="Internal processing resolution (e.g. KITTI width 1216).")
    parser.add_argument("--process-res-method", type=str, default="upper_bound_resize")

    # batching
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=0, help="<=0 means all")

    # output encoding
    parser.add_argument("--png-scale", type=float, default=256.0, help="meters -> uint16 scale")
    parser.add_argument("--max-depth-m", type=float, default=255.0)

    # alignment / debug
    parser.add_argument("--align-back-to-input", action="store_true",
                        help="If depth != input size, interpolate back to (H,W).")
    parser.add_argument("--save-used-intrinsics", action="store_true",
                        help="Save model-used intrinsics (pred.intrinsics) into out_dir/_intrinsics_used.")
    parser.add_argument("--verbose-shape", action="store_true",
                        help="Print processed depth shapes to confirm process_res is effective.")

    args = parser.parse_args()

    img_paths = list_images(args.image_dir)
    if not img_paths:
        raise SystemExit(f"No images found in {args.image_dir}")

    if args.max_images and args.max_images > 0:
        img_paths = img_paths[:args.max_images]

    os.makedirs(args.out_dir, exist_ok=True)
    used_k_dir = os.path.join(args.out_dir, "_intrinsics_used")
    if args.save_used_intrinsics:
        os.makedirs(used_k_dir, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Load model
    model = DepthAnything3.from_pretrained(args.ckpt).to(device=device).eval()

    pbar = tqdm(total=len(img_paths), desc="DA3 inference")

    for batch_paths in chunker(img_paths, args.batch_size):
        images = []
        intrinsics_list = []
        orig_hw = []

        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            w, h = img.size
            orig_hw.append((h, w))
            images.append(img)

            stem = Path(p).stem
            k_path = os.path.join(args.intrinsics_dir, stem + ".txt")
            if not os.path.isfile(k_path):
                raise FileNotFoundError(f"Missing intrinsics txt: {k_path}")
            intrinsics_list.append(read_k33(k_path))

        intrinsics_np = np.stack(intrinsics_list, axis=0)  # [N,3,3]

        # forward (IMPORTANT: process_res affects the internal pre-processing)
        if device.type == "cuda" and args.amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model.inference(
                    image=images,
                    intrinsics=intrinsics_np,
                    process_res=args.process_res,
                    process_res_method=args.process_res_method,
                    # export_dir=None,
                    # export_format=None,
                )
        else:
            pred = model.inference(
                image=images,
                intrinsics=intrinsics_np,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
            )
            # pred = model.inference(
            #     image=images,
            #     intrinsics=intrinsics_np,
            #     process_res=args.process_res,
            #     process_res_method=args.process_res_method,
            #     export_dir=None,
            #     export_format=None,
            # )

        depth = pred.depth  # expected [N,Hproc,Wproc] float32 (metric for metric/nested models)

        if args.verbose_shape:
            print(f"[process_res={args.process_res}, method={args.process_res_method}] depth.shape = {getattr(depth, 'shape', None)}")

        # pred.intrinsics is often the "used" intrinsics after internal resize; save for sanity check
        if args.save_used_intrinsics and getattr(pred, "intrinsics", None) is not None:
            k_used = pred.intrinsics
            if torch.is_tensor(k_used):
                k_used = k_used.detach().cpu().numpy()
            for i, p in enumerate(batch_paths):
                stem = Path(p).stem
                np.savetxt(os.path.join(used_k_dir, stem + ".txt"), k_used[i].reshape(3, 3), fmt="%.8f")

        # ensure torch tensor for optional alignment
        if isinstance(depth, np.ndarray):
            depth_t = torch.from_numpy(depth).to(device=device, dtype=torch.float32)
        else:
            depth_t = depth.to(device=device, dtype=torch.float32)

        # align back to input size only when needed
        out_depth_list = []
        for i, p in enumerate(batch_paths):
            h0, w0 = orig_hw[i]
            di = depth_t[i:i+1, None]  # [1,1,H,W]
            if args.align_back_to_input and (di.shape[-2] != h0 or di.shape[-1] != w0):
                di = F.interpolate(di, size=(h0, w0), mode="bilinear", align_corners=False)
            out_depth_list.append(di[0, 0])  # [H,W] (aligned or native)

        out_depth = torch.stack(out_depth_list, dim=0)  # [N,H,W]

        # write uint16 png
        for i, p in enumerate(batch_paths):
            name = Path(p).name  # keep original name (xxx.png)
            out_png = os.path.join(args.out_dir, name)

            dm = out_depth[i].detach().cpu().numpy().astype(np.float32)
            dm = np.clip(dm, 0.0, args.max_depth_m)

            u16 = np.clip(dm * args.png_scale, 0, 65535).astype(np.uint16)
            imageio.imwrite(out_png, u16)

        pbar.update(len(batch_paths))

    pbar.close()
    print(f"Done. Saved uint16 depth PNGs to: {args.out_dir}")
    if args.save_used_intrinsics:
        print(f"Saved model-used intrinsics to: {used_k_dir}")


if __name__ == "__main__":
    main()
