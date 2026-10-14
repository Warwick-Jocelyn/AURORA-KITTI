"""
Build dense_small.csv — per-image metadata for val_selection_cropped_DENSE,
column-compatible with val_selection_cropped_TEST/test_small.csv so that
eval_per_weather.py can be reused with one parser change.

Convention (confirmed against PixelAccurate benchmark source + visual check):
  - fog{N}  : N = visibility distance in metres (small N = thicker fog).
              light  : N >= 70
              medium : 30 < N < 70
              heavy  : N <= 30           # 30 visually already very thick
  - rain{N} : N = rainfall rate in mm/h (large N = heavier rain).
              Dataset has only rain15 and rain55, so:
              rain15 -> light
              rain55 -> heavy
  - clear   : severity = None
"""
import csv
import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(ROOT, "image")
OUT = os.path.join(ROOT, "dense_small.csv")

WEATHER_PATTERN = re.compile(r"^(clear|fog|rain)(\d*)$")


def classify_fog(v: int) -> str:
    if v >= 70:
        return "light"
    if v > 30:
        return "medium"
    return "heavy"


def classify_rain(v: int) -> str:
    return "light" if v <= 15 else "heavy"  # only 15 and 55 in this dataset


def main():
    rows = []
    counters = Counter()
    for fn in sorted(os.listdir(IMG_DIR)):
        if not fn.endswith(".png"):
            continue
        stem = os.path.splitext(fn)[0]            # e.g. scene1_day_fog30_4
        parts = stem.split("_")
        if len(parts) < 4:
            print("SKIP unexpected:", fn)
            continue
        scene, tod, w_tag = parts[0], parts[1], parts[2]
        idx = "_".join(parts[3:])
        m = WEATHER_PATTERN.match(w_tag)
        if not m:
            print("SKIP bad weather tag:", fn)
            continue
        wkind, wnum = m.group(1), m.group(2)
        raw_value = int(wnum) if wnum else ""
        if wkind == "clear":
            weather, severity = "Clear", "None"
        elif wkind == "fog":
            weather, severity = "Fog", classify_fog(int(wnum))
        else:
            weather, severity = "Rain", classify_rain(int(wnum))

        sequence_id = f"{scene}_{tod}_{w_tag}"
        time_of_day = "Day" if tod == "day" else "Night"
        rows.append({
            "sequence_id": sequence_id,
            "name": stem,
            "Scenario": "DENSE",
            "weather": weather,
            "severity": severity,
            "Ground_Condition": "Unknown",
            "Sky_Condition": "Unknown",
            "Lens_Condition": "Clean",
            "Time_of_Day": time_of_day,
            "raw_value": raw_value,
            "scene": scene,
        })
        counters[(weather, severity, time_of_day)] += 1

    cols = ["sequence_id", "name", "Scenario", "weather", "severity",
            "Ground_Condition", "Sky_Condition", "Lens_Condition",
            "Time_of_Day", "raw_value", "scene"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {OUT}  ({len(rows)} rows)")
    print("\nDistribution (weather × severity × Time_of_Day):")
    for k, v in sorted(counters.items()):
        print(f"  {k[0]:<8s} / {k[1]:<8s} / {k[2]:<6s} : {v}")


if __name__ == "__main__":
    main()
