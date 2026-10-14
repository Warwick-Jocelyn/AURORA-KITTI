#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
from pathlib import Path

FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

def read_numbers(txt: str):
    nums = [float(x) for x in FLOAT_RE.findall(txt)]
    return nums

def write_numbers(path: Path, nums, per_line: int, fmt: str):
    lines = []
    for i in range(0, len(nums), per_line):
        chunk = nums[i:i+per_line]
        lines.append(" ".join(fmt.format(v) for v in chunk))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def scale_matrix(nums, sx: float, sy: float):
    """
    支持两种常见格式：
    - 9 个数：3x3 K
    - 12 个数：3x4 P（投影矩阵）
    规则：第 1 行乘 sx，第 2 行乘 sy，其余不变
    """
    if len(nums) == 9:       # 3x3
        per_line = 3
        out = nums[:]
        # row 0
        out[0] *= sx; out[1] *= sx; out[2] *= sx
        # row 1
        out[3] *= sy; out[4] *= sy; out[5] *= sy
        # row 2 unchanged
        return out, per_line

    if len(nums) == 12:      # 3x4
        per_line = 4
        out = nums[:]
        # row 0 (0..3)
        for j in range(4):
            out[0*4 + j] *= sx
        # row 1 (4..7)
        for j in range(4):
            out[1*4 + j] *= sy
        # row 2 unchanged (8..11)
        return out, per_line

    raise ValueError(f"不支持的数字数量：{len(nums)}（期望 9 或 12）。")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="原始 calib txt 文件夹")
    ap.add_argument("--out_dir", required=True, help="输出新文件夹")
    ap.add_argument("--src_w", type=float, default=1920)
    ap.add_argument("--src_h", type=float, default=1024)
    ap.add_argument("--dst_w", type=float, default=1216)
    ap.add_argument("--dst_h", type=float, default=352)
    ap.add_argument("--glob", default="*.txt", help="匹配文件名，如 *.txt")
    ap.add_argument("--fmt", default="{:.6f}", help="输出浮点格式，默认保留6位小数")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sx = args.dst_w / args.src_w
    sy = args.dst_h / args.src_h

    files = sorted(in_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"在 {in_dir} 下没找到匹配 {args.glob} 的文件")

    for f in files:
        txt = f.read_text(encoding="utf-8", errors="ignore")
        nums = read_numbers(txt)
        new_nums, per_line = scale_matrix(nums, sx, sy)
        out_path = out_dir / f.name
        write_numbers(out_path, new_nums, per_line=per_line, fmt=args.fmt)

    print(f"Done. sx={sx:.8f}, sy={sy:.8f}, wrote {len(files)} files to {out_dir}")

if __name__ == "__main__":
    main()


# python resize_calib.py \
#   --in_dir ./intrinsics_original \
#   --out_dir ./intrinsics \
#   --src_w 1920 --src_h 1024 \
#   --dst_w 1216 --dst_h 352
