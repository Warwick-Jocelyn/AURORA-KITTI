# demo_promptda_kitti.py
import os
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from tqdm import tqdm

from promptda.promptda import PromptDA
from promptda.utils.io_wrapper import load_image  # 官方提供的图像预处理

# -------------------------
# Utils
# -------------------------
def read_depth_png_to_meters(png_path: str, png_scale: float) -> torch.Tensor:
    """
    Read a single-channel uint16 depth PNG and convert to meters.
    Convention: depth_m = depth_u16 / png_scale
    Returns: torch tensor [1, 1, H, W] float32
    """
    d = imageio.imread(png_path)
    if d.ndim == 3:
        d = d[..., 0]
    d = d.astype(np.float32)
    # depth_m = d / float(png_scale)
    # t = torch.from_numpy(depth_m)[None, None, ...].float()
    t = torch.from_numpy(d)[None, None, ...].float()
    return t


def save_depth_m_to_png16(depth_m: np.ndarray, out_png: str, png_scale: float, max_depth_m: float):
    depth_m = np.clip(depth_m, 0.0, max_depth_m).astype(np.float32)
    u16 = np.clip(depth_m * png_scale, 0, 65535).astype(np.uint16)
    imageio.imwrite(out_png, u16)


def load_promptda_model(model_id: str, ckpt: str, device: torch.device):
    """
    Prefer HF model_id; if you want local checkpoint, try (best-effort) Lightning-style loading.
    """
    if model_id:
        model = PromptDA.from_pretrained(model_id)
        return model.to(device).eval()

    # Best-effort local ckpt load (may vary with your saved format)
    if ckpt:
        # Try Lightning-style API if available in this class
        if hasattr(PromptDA, "load_from_checkpoint"):
            model = PromptDA.load_from_checkpoint(ckpt, map_location="cpu")
            return model.to(device).eval()

        # Fallback: state_dict loading (only works if ckpt is a plain state_dict or contains one)
        obj = torch.load(ckpt, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj:
            sd = obj["state_dict"]
        elif isinstance(obj, dict):
            sd = obj
        else:
            raise RuntimeError("Unsupported checkpoint format.")

        model = PromptDA()  # may fail if constructor needs args
        model.load_state_dict(sd, strict=False)
        return model.to(device).eval()

    raise ValueError("Provide either --model-id or --ckpt.")


# -------------------------
# Main
# -------------------------
@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()

    

    # keep similar cmd style
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--intrinsics-dir", type=str, default="", help="(Not used by PromptDA demo; kept for cmd compatibility)")
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument(
    "--prompt-name-replace",
    type=str,
    default="_sync_image_=_sync_velodyne_raw_",
    help='Rename rule for prompt filename. Format: "OLD=NEW". '
         'Example: "_sync_image_=_sync_velodyne_raw_". Use "" to disable.'
)

    parser.add_argument("--max-images", type=int, default=0, help="<=0 means all")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--input-size", type=int, default=0,
                        help="Optional: resize RGB long-side (0 = keep original). PromptDA can handle high-res; keep 0 if you want.")
    # Prompt depth (required in practice)
    parser.add_argument("--prompt-depth-dir", type=str, required=True,
                        help="Folder of per-image prompt depth PNGs (same stem as image).")
    parser.add_argument("--prompt-size", type=int, nargs=2, default=[192, 256],
                        help="Resize prompt depth to this (H W). Default matches ARKit prompt (192x256).")
    parser.add_argument("--prompt-png-scale", type=float, default=256.0,
                        help="If prompt depth png is uint16, depth_m = depth_u16 / prompt_png_scale.")

    # Output
    parser.add_argument("--out-png-scale", type=float, default=256.0,
                        help="Save metric depth as uint16: depth_u16 = depth_m * out_png_scale.")
    parser.add_argument("--max-depth-m", type=float, default=255.0)

    # Model loading
    parser.add_argument("--model-id", type=str, default="depth-anything/prompt-depth-anything-vitl",
                        help="Recommended (HF).")
    parser.add_argument("--ckpt", type=str, default="",
                        help="Optional local checkpoint (.ckpt). If provided, set --model-id '' to use it.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    model = load_promptda_model(args.model_id, args.ckpt, device)

    img_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No png images found in {args.image_dir}")

    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(img_paths)

    if args.max_images and args.max_images > 0:
        img_paths = img_paths[: args.max_images]

    os.makedirs(args.out_dir, exist_ok=True)

    # batching helper
    def chunker(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    pbar = tqdm(total=len(img_paths), desc="PromptDA inference")

    for batch_paths in chunker(img_paths, args.batch_size):
        images = []
        prompts = []
        orig_hw = []

        for p in batch_paths:
            # RGB
            img = load_image(p).to(device)   # expected shape [1,3,H,W]
            h0, w0 = img.shape[-2], img.shape[-1]
            orig_hw.append((h0, w0))

            # Optional resize RGB (long-side)
            if args.input_size and args.input_size > 0:
                # keep aspect ratio
                long_side = max(h0, w0)
                scale = args.input_size / float(long_side)
                nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
                img = F.interpolate(img, size=(nh, nw), mode="bilinear", align_corners=False)

            images.append(img)

            # # Prompt depth (uint16 png -> meters)
            # stem = Path(p).stem
            # prompt_path = os.path.join(args.prompt_depth_dir, stem + ".png")
            # if not os.path.exists(prompt_path):
            #     raise FileNotFoundError(f"Missing prompt depth: {prompt_path}")

            stem = Path(p).stem

            # apply rename rule to match prompt files
            prompt_stem = stem
            if args.prompt_name_replace:
                if "=" not in args.prompt_name_replace:
                    raise ValueError('--prompt-name-replace must be like "OLD=NEW"')
                old, new = args.prompt_name_replace.split("=", 1)
                prompt_stem = prompt_stem.replace(old, new)

            prompt_path = os.path.join(args.prompt_depth_dir, prompt_stem + ".png")
            if not os.path.exists(prompt_path):
                raise FileNotFoundError(
                    f"Missing prompt depth: {prompt_path}\n"
                    f"  image stem : {stem}\n"
                    f"  prompt stem: {prompt_stem}\n"
                    f'  rule      : "{args.prompt_name_replace}"'
                )

            pd = read_depth_png_to_meters(prompt_path, args.prompt_png_scale).to(device)  # [1,1,H,W]

            # resize prompt depth to expected 192x256 (H,W), using nearest to preserve units
            ph, pw = args.prompt_size
            pd = F.interpolate(pd, size=(ph, pw), mode="nearest")
            prompts.append(pd)

        # stack batch
        image_b = torch.cat(images, dim=0)   # [N,3,H,W] (after optional resize)
        prompt_b = torch.cat(prompts, dim=0) # [N,1,ph,pw]

        # Predict: output depth in meters, shape [N,H,W] or [N,1,H,W] depending on implementation
        depth = model.predict(image_b, prompt_b)

        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth).to(device)

        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]  # [N,H,W]

        # If RGB was resized for inference, resize depth back to original input resolution per-image
        out_depths = []
        for i in range(depth.shape[0]):
            h0, w0 = orig_hw[i]
            di = depth[i:i+1, None, ...]  # [1,1,H,W]
            di = F.interpolate(di, size=(h0, w0), mode="bilinear", align_corners=False)
            out_depths.append(di[0, 0].detach().cpu().numpy())

        # save
        for i, p in enumerate(batch_paths):
            out_png = os.path.join(args.out_dir, Path(p).name)
            save_depth_m_to_png16(out_depths[i], out_png, args.out_png_scale, args.max_depth_m)

        pbar.update(len(batch_paths))

    pbar.close()
    print(f"Done. Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
