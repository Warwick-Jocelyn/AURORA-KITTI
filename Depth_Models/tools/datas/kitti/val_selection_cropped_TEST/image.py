from pathlib import Path
import re
import cv2
from tqdm import tqdm


# -----------------------------
# 1) 解析文件名 -> AuroraKITTI 原图路径
# -----------------------------
# 例子：
# 2011_09_26_drive_0011_sync_image_0000000005_image_02.png
# -> drive_folder = 2011_09_26_drive_0011_sync
# -> frame_id     = 0000000005
# -> cam          = image_02
FNAME_RE = re.compile(
    r"^(?P<drive>\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)_image_(?P<frame>\d{10})_(?P<cam>image_\d{2})\.png$"
)


def build_aurora_path(aurora_root: Path, selection_png_name: str) -> Path:
    m = FNAME_RE.match(selection_png_name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {selection_png_name}")

    drive_folder = m.group("drive")
    frame_id = m.group("frame")
    cam = m.group("cam")

    # /test/<drive_folder>/<cam>/data/<frame_id>.png
    return aurora_root / drive_folder / cam / "data" / f"{frame_id}.png"


# -----------------------------
# 2) 裁剪逻辑：左右 + 上方
# -----------------------------
def crop_lr_and_top(img, out_w: int = 1216, out_h: int = 352):
    """
    左右裁剪：水平居中裁到 out_w
    上方裁剪：只裁上面（保留底部 out_h）
    若输入小于目标：先等比放大到 >= 目标，再裁剪，保证输出固定(out_w, out_h)
    """
    if img is None:
        return None

    h, w = img.shape[:2]

    # 若尺寸小于目标，先等比例放大到不小于目标
    if w < out_w or h < out_h:
        scale = max(out_w / w, out_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        h, w = img.shape[:2]

    # 左右裁剪（居中）
    if w > out_w:
        left = (w - out_w) // 2
        right = left + out_w
    else:
        left, right = 0, w

    # 上方裁剪：保留底部 out_h（即从上面裁掉）
    if h > out_h:
        top = h - out_h
        bottom = h
    else:
        top, bottom = 0, h

    cropped = img[top:bottom, left:right]

    # 双保险：如果因为取整导致尺寸不是精确目标，最后再 resize 一次
    if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
        cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)

    return cropped


# -----------------------------
# 3) 主处理：遍历 selection 文件名 -> 找 Aurora 原图 -> crop -> 保存
# -----------------------------
def crop_from_aurorakitti_by_selection_names(
    src_root: str,
    aurora_root: str,
    dst_root: str,
    out_size=(1216, 352),
    verbose_every: int = 0,
):
    src_root = Path(src_root)
    aurora_root = Path(aurora_root)
    dst_root = Path(dst_root)

    all_pngs = list(src_root.rglob("*.png"))
    missing = 0
    badname = 0
    readfail = 0
    saved = 0

    for i, sel_path in enumerate(tqdm(all_pngs, desc="Cropping from AuroraKITTI", unit="file")):
        rel_path = sel_path.relative_to(src_root)
        out_path = dst_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            aur_path = build_aurora_path(aurora_root, sel_path.name)
        except ValueError:
            badname += 1
            continue

        if not aur_path.exists():
            missing += 1
            continue

        img = cv2.imread(str(aur_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            readfail += 1
            continue

        cropped = crop_lr_and_top(img, out_w=out_size[0], out_h=out_size[1])
        if cropped is None:
            readfail += 1
            continue

        ok = cv2.imwrite(str(out_path), cropped)
        if ok:
            saved += 1

        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"[{i+1}/{len(all_pngs)}] saved={saved}, missing={missing}, badname={badname}, readfail={readfail}")

    print("\n✅ Done!")
    print(f"  Selection pngs: {len(all_pngs)}")
    print(f"  Saved:          {saved}")
    print(f"  Missing src:    {missing}  (AuroraKITTI path not found)")
    print(f"  Bad filename:   {badname}  (name not match pattern)")
    print(f"  Read failed:    {readfail}")
    print(f"  Output folder:  {dst_root}")


if __name__ == "__main__":
    SRC_ROOT = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped/image"
    AURORA_ROOT = "/path/to/aurora-root/Dataset/AuroraKITTI/data/data_rgb/test"
    DST_ROOT = "/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/kitti/val_selection_cropped/image_cropped"

    crop_from_aurorakitti_by_selection_names(
        src_root=SRC_ROOT,
        aurora_root=AURORA_ROOT,
        dst_root=DST_ROOT,
        out_size=(1216, 352),
        verbose_every=0,  # 想看过程日志可设成 100/500 等
    )
