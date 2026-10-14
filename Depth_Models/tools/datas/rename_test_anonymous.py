"""
Rename Test set files to anonymous sequential names (00000.png, 00001.png, ...)
to prevent reverse-engineering of the original KITTI file structure.

Directories renamed (in sync):
  RGB_Foggy/Test/Day/                       (public)
  Test_set_private/Day_test_anotations_private/  (private GT)
  Test_clean_RGB__private/                  (private clean RGB)

A mapping CSV is saved privately so you can always trace back:
  Test_set_private/test_rename_mapping.csv
"""

import csv
import os
from pathlib import Path

BASE = Path("/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas/AURORA_Foggy_Set_Day")

DIRS = {
    "foggy":  BASE / "RGB_Foggy" / "Test" / "Day",
    "depth":  BASE / "Test_set_private" / "Day_test_anotations_private",
    "clean":  BASE / "Test_clean_RGB__private",
}
MAPPING_OUT = BASE / "Test_set_private" / "test_rename_mapping.csv"

def main():
    # Use RGB_Foggy as the master list (sorted alphabetically = default order)
    master_files = sorted(DIRS["foggy"].glob("*.png"), key=lambda p: p.name)
    n = len(master_files)
    pad = len(str(n - 1))  # e.g. 197 files → pad=2 gives 00..196, use 5 for safety

    print(f"Found {n} test files. Padding width: {pad} digits.\n")

    # Verify all dirs have matching filenames
    for key, d in DIRS.items():
        files_here = set(p.name for p in d.glob("*.png"))
        master_names = set(p.name for p in master_files)
        missing = master_names - files_here
        extra   = files_here - master_names
        if missing or extra:
            print(f"[WARN] {key} ({d.name}): missing={len(missing)}, extra={len(extra)}")
        else:
            print(f"[OK]   {key}: all {len(files_here)} files match master list")

    print()

    # Build mapping and rename
    mapping = []
    for idx, src_foggy in enumerate(master_files):
        new_name = f"{idx:05d}.png"
        original = src_foggy.name

        for key, d in DIRS.items():
            src = d / original
            dst = d / new_name
            if src.exists():
                os.rename(src, dst)
            else:
                print(f"  [MISSING] {key}/{original}")

        mapping.append({"index": f"{idx:05d}", "original_name": original})
        if idx % 50 == 0:
            print(f"  Renamed {idx+1}/{n} ...")

    # Save private mapping
    MAPPING_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "original_name"])
        writer.writeheader()
        writer.writerows(mapping)

    print(f"\nDone. {n} files renamed to 00000.png ~ {(n-1):05d}.png")
    print(f"Private mapping saved: {MAPPING_OUT}")
    print(f"\nFirst 5 mappings:")
    for row in mapping[:5]:
        print(f"  {row['index']}.png  ←  {row['original_name']}")


if __name__ == "__main__":
    main()
