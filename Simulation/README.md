# Simulation

The physics-consistent weather generation pipeline behind **AURORA-KITTI**
(Information Fusion 139 (2027) 104830).

The guiding property is **cross-modal consistency**: RGB and LiDAR are degraded under a matched
weather type and severity for the same scene, so a "heavy fog" image and its sparse depth map
describe the same physical conditions.

## Layout

```
Simulation/
├── lidar/           LiDAR weather simulation — corruption in raw 3D, before projection
│   ├── simulate_from_csv.py        the AURORA-KITTI generator
│   ├── configs/noise_params/       physical parameters per model
│   ├── configs/severity_presets/   light / moderate / heavy, made explicit (see below)
│   ├── configs/sensor_params/      Velodyne HDL-64E model + factory calibration
│   └── pre_computed_materials/     fog integral LUTs, Mie scattering cache
├── rgb/             RGB weather rendering — fog, rain streaks, lens occlusion
│   └── validate/    FID / KID / IQA realism scoring
├── projection/      weathered 3D point cloud → image-aligned sparse depth
│   └── validate/    range-bin statistics, drop-rate measurements
├── realism/         quantitative realism vs real data (MUSES, DENSE)
│   └── lidar_degradation/   the published degradation table + snow_rescue analysis
└── annotations/     per-frame weather / severity / lens / time-of-day labels
```

## Pipeline

```
KITTI raw                                           AURORA-KITTI
  ├─ RGB   ──► rgb/      fog · rain · snow  ──┐
  │              + lens occlusion            ├──► weather RGB
  └─ .bin  ──► lidar/    attenuation,        │
                 backscatter, dropout        └──► projection/ ──► sparse depth + height + intensity
```

Degradation is injected into the **raw 3D point cloud** and only then projected, which is what
produces realistic non-uniform sparsity, range-dependent attenuation and spurious returns —
rather than painting noise onto a 2D depth map.

## Severity, and an important caveat

In the original pipeline **severity was not a code parameter**. It was set by hand-editing
`configs/noise_params/*.yaml` between runs; the `--csv_severity` flag only selected *which
frames* to process. That tacit convention is the single biggest reproducibility hazard in this
pipeline, so it is now written down explicitly in `configs/severity_presets/`:

| Weather | light | moderate | heavy | Measured point retention |
|---|---|---|---|---|
| Fog / haze (β, 1/m) | 0.01 | 0.1 | 0.2 | 97.99 % → ~91 % → ~76 % |
| Rain (mm/h) | 10 | 100 | 200 | 86.56 % → 87.16 % → — |
| Snow (mm/h) | 0.5 | 1.5 | 2.5 | not recorded |
| Wet ground (pavement/water/floor) | .00012/.00002/.10 | .00015/.00012/.10 | .00150/.00020/.20 | 94.49 % → 95.85 % → 95.65 % |

Three caveats, stated rather than hidden:

1. The **fog `alpha` ladder is not recorded anywhere.** Every logged command used the `haze`
   model (β), though `--sim_weather fog` is also a valid path.
2. **Wet-ground retention is non-monotonic** (heavy retains more than light). Either the note is
   mislabelled or the parameterisation genuinely is non-monotonic — unverified.
3. Snow simulation is **not bit-reproducible** across runs, nor is anything under `n_jobs > 1`.

## Running it

```bash
cd Simulation/lidar
python3 simulate_from_csv.py \
    --csv_path ./raw_lidar/val.csv \
    --simulated_root ./raw_lidar_simulated_Rain_heavy \
    --csv_weather Rain --csv_severity heavy \
    --sim_weather rain_wet_ground \
    --sensor_cfg ./configs/sensor_params/lidar_velodyne_64_config.yaml
```

Run from `lidar/` — the weather models resolve `./weather_models/npy`,
`./pre_computed_materials/...` and `./configs/sensor_params/...` by relative path.

**Before first use** you must fetch the five physical weather models and the snow particle
fields; they are third-party and not redistributed here. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[lidar/weather_models/PORTING.md](lidar/weather_models/PORTING.md).

## Two defects in the released annotations

Both are in `annotations/*.csv` and are documented rather than silently patched, because the
published dataset was generated with them in place:

1. `severity` for Clear rows is `" None"` — with a **leading space**. So `--csv_severity None`
   matches nothing; use `" None"` or strip before comparing.
2. In the original tree the train point clouds lived in a directory named `missing/`, which
   `--split` never resolves to. It is the train pool, not a scratch directory.

## Realism validation

`realism/` holds the quantitative evidence that the synthetic weather behaves like real weather:
FID/KID against MUSES and DENSE, nine physically-grounded descriptors (dark channel, Koschmieder
transmission, airlight, …), and Weather-Shift Consistency — the headline metric, which compares
*differences* (weather minus clear) between synthetic and real rather than raw appearance.

`realism/lidar_degradation/` produces the published degradation table by applying the paper's own
denoising masks to paired clean/weather sparse LiDAR. Its `snow_rescue/` subfolder is worth
reading: snow's severity signal is flat in globally-pooled statistics, and that analysis recovers
it in far-range dropout and spurious-return counts — running fog as a positive control and
discarding two candidate statistics that failed it.

## Status

Snow RGB rendering code was lost; see [rgb/snow/NOTICE.md](rgb/snow/NOTICE.md). The released
snow **data** is unaffected.
