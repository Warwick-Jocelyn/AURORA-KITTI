"""
Build AURORA_Foggy_Set_Day/RGB_Clean from existing foggy set filenames.

For each foggy image in AURORA_Foggy_Set/RGB/{Train,Val,Test}/{Day,Night}/,
parse the filename to locate the corresponding clean image in kitti_depth_DC/raw,
then save to AURORA_Foggy_Set_Day/RGB_Clean/{Train,Val,Test}/ (no Day/Night subdir).

Filename convention:
  {seq_id}_image_{frame_id}_{camera}.png
  e.g. 2011_09_26_drive_0001_sync_image_0000000005_image_02.png

Clean source:
  kitti_depth_DC/raw/{date}/{seq_id}/{camera}/data/{frame_id}.png
  e.g. kitti_depth_DC/raw/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png

Output:
  AURORA_Foggy_Set_Day/RGB_Clean/{Train,Val,Test}/{same_filename}.png
"""

import re
import shutil
from pathlib import Path

BASE       = Path("/path/to/aurora-root/PhD_Project_2025/DMD3C/BP-Net/DMD3C/datas")
FOGGY_SET  = BASE / "AURORA_Foggy_Set_Day" / "RGB_Foggy"
CLEAN_RAW  = BASE / "kitti_depth_DC" / "raw"
OUTPUT     = BASE / "AURORA_Foggy_Set_Day" / "RGB_Clean"

# Matches: {seq_id}_image_{frame_id}_{camera}
# e.g.  2011_09_26_drive_0001_sync_image_0000000005_image_02
FNAME_RE = re.compile(r'^(.+)_image_(\d+)_(image_\d+)\.png$')
DATE_RE  = re.compile(r'^(\d{4}_\d{2}_\d{2})_')


def parse_foggy_filename(fname):
    """
    '2011_09_26_drive_0001_sync_image_0000000005_image_02.png'
    -> (seq_id='2011_09_26_drive_0001_sync', frame='0000000005', camera='image_02')
    """
    m = FNAME_RE.match(fname)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def get_date(seq_id):
    m = DATE_RE.match(seq_id)
    return m.group(1) if m else None


def main():
    splits = ["Train", "Val", "Test"]

    for split in splits:
        out_dir = OUTPUT / split
        out_dir.mkdir(parents=True, exist_ok=True)

        copied   = 0
        missing  = 0
        bad_name = 0

        # Walk both Day and Night subfolders
        for tod in ("Day", "Night"):
            src_dir = FOGGY_SET / split / tod
            if not src_dir.is_dir():
                print(f"  [SKIP] {src_dir} not found")
                continue

            for foggy_file in sorted(src_dir.glob("*.png")):
                parsed = parse_foggy_filename(foggy_file.name)
                if parsed is None:
                    print(f"  [WARN] Cannot parse: {foggy_file.name}")
                    bad_name += 1
                    continue

                seq_id, frame_id, camera = parsed
                date = get_date(seq_id)
                if date is None:
                    print(f"  [WARN] Cannot extract date from: {seq_id}")
                    bad_name += 1
                    continue

                clean_src = CLEAN_RAW / date / seq_id / camera / "data" / f"{frame_id}.png"
                clean_dst = out_dir / foggy_file.name  # same filename, no Day/Night

                if clean_src.exists():
                    shutil.copy2(clean_src, clean_dst)
                    copied += 1
                else:
                    print(f"  [MISSING] {clean_src}")
                    missing += 1

        total = copied + missing + bad_name
        print(f"{split}: {total} foggy files processed")
        print(f"  Clean copied : {copied}")
        print(f"  Missing      : {missing}")
        print(f"  Bad names    : {bad_name}\n")

    print("Done.")
    print(f"Output: {OUTPUT}/{{Train,Val,Test}}/")


if __name__ == "__main__":
    main()
