"""
Qualitative validation visualization for wandb during modality-ablation training.
Logs, each epoch, a fixed set of weather DMD3C-VAL frames (clear/fog/rain/snow) as
stacked panels: RGB | sparse(input) | prediction | GT  (depth colorized, turbo, 0-80m).
Lets you watch predictions improve over training. Decoding matches inference exactly
(rgb raw uint8 [0,255]; depth png/256).
"""
import os, csv
import numpy as np
import cv2
import torch
from PIL import Image

VAL_ROOT = ("/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/"
            "DMD3C/datas/kitti/val_selection_cropped_VAL")
CSV = os.path.join(VAL_ROOT, "val_small.csv")
OHEIGHT, OWIDTH = 352, 1216
MAXD = 80.0


def _pick_frames(per_weather=2):
    """Deterministic: first `per_weather` frames of each weather from val_small.csv."""
    rows = list(csv.DictReader(open(CSV)))
    by_w, picked = {}, []
    for r in rows:
        w = r["weather"].strip()
        by_w.setdefault(w, 0)
        if by_w[w] < per_weather:
            by_w[w] += 1
            picked.append(r)
    return picked  # each has 'name' = <drive>_image_<cam>_<frame>, 'weather'


def _paths(name):
    drive, rest = name.split("_image_"); cam, frame = rest.split("_")
    base = f"{drive}_{{}}_{frame}_image_{cam}.png"
    return (os.path.join(VAL_ROOT, "image", base.format("image")),
            os.path.join(VAL_ROOT, "velodyne_raw", base.format("velodyne_raw")),
            os.path.join(VAL_ROOT, "groundtruth_depth", base.format("groundtruth_depth")))


def _colorize(depth_m):
    d = np.clip(depth_m / MAXD, 0, 1)
    cm = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)  # BGR
    cm = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
    cm[depth_m <= 0] = 0
    return cv2.dilate(cm, np.ones((2, 2), np.uint8))  # fatten sparse points


def _label(img, txt):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(img, txt, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


@torch.no_grad()
def build_panels(model, args, device):
    """Returns list of (caption, HxWx3 uint8 RGB panel). Does not touch wandb."""
    model.eval()
    panels = []
    for r in _pick_frames():
        img_p, vel_p, gt_p = _paths(r["name"])
        if not (os.path.exists(img_p) and os.path.exists(vel_p) and os.path.exists(gt_p)):
            continue
        rgb = np.array(Image.open(img_p).convert("RGB"), dtype=np.uint8)
        sparse = np.array(Image.open(vel_p), dtype=np.float32) / 256.0
        gt = np.array(Image.open(gt_p), dtype=np.float32) / 256.0
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32))
        d_t = torch.from_numpy(sparse[None, :, :])
        g = np.array(Image.fromarray(rgb).convert("L"), dtype=np.float32)
        batch = {"rgb": rgb_t.unsqueeze(0).to(device),
                 "d": d_t.unsqueeze(0).to(device),
                 "g": torch.from_numpy(g[None, None]).to(device)}
        pred = model(batch).squeeze().detach().cpu().numpy()
        panel = np.vstack([
            _label(rgb, f"{r['weather']}/{r.get('severity','').strip()}  RGB"),
            _label(_colorize(sparse), "sparse input"),
            _label(_colorize(pred), "prediction"),
            _label(_colorize(gt), "GT"),
        ])
        panels.append((f"{r['weather']}_{r['name']}", panel))
    return panels


def log_val_viz(model, args, epoch, device, wandb_mod):
    """Build panels and log to wandb (no-op-safe if wandb missing)."""
    try:
        panels = build_panels(model, args, device)
        if panels and wandb_mod is not None:
            wandb_mod.log({"val_examples": [wandb_mod.Image(p, caption=c) for c, p in panels],
                           "epoch": epoch})
        return len(panels)
    except Exception as e:
        print(f"[wandb viz] skipped: {e}")
        return 0


if __name__ == "__main__":  # self-test: throwaway rgbd model -> save a panel, no wandb
    import types
    from model import DepthCompletionNet
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = DepthCompletionNet(types.SimpleNamespace(layers=34, input="rgbd", pretrained=False)).to(dev)
    ps = build_panels(m, types.SimpleNamespace(input="rgbd"), dev)
    print(f"built {len(ps)} panels")
    if ps:
        cv2.imwrite("/tmp/val_viz_selftest.png", cv2.cvtColor(ps[0][1], cv2.COLOR_RGB2BGR))
        print("saved /tmp/val_viz_selftest.png", ps[0][1].shape)
