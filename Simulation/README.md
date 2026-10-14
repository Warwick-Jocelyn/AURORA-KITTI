# Simulation

The physics-consistent weather generation pipeline behind **AURORA-KITTI**
(Information Fusion 139 (2027) 104830).

The guiding property is **cross-modal consistency**: RGB and LiDAR are degraded under a matched
weather type and severity for the same scene, so a heavy-fog image and its sparse depth map
describe the same physical conditions.

## Layout

```
Simulation/
├── lidar/           LiDAR weather simulation — corruption in raw 3D, before projection
│   ├── simulate_from_csv.py        the AURORA-KITTI generator
│   ├── configs/noise_params/       physical parameters per model
│   ├── configs/severity_presets/   light / moderate / heavy presets
│   ├── configs/sensor_params/      Velodyne HDL-64E model + factory calibration
│   └── pre_computed_materials/     fog integral tables, Mie scattering cache
├── rgb/             RGB weather rendering — fog, rain streaks, lens occlusion
│   └── validate/    FID / KID / IQA realism scoring
├── projection/      weathered 3D point cloud → image-aligned sparse depth
│   └── validate/    range-bin statistics, drop-rate measurements
├── realism/         quantitative realism against real data (MUSES, DENSE)
│   └── lidar_degradation/   the published degradation analysis
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

Degradation is injected into the **raw 3D point cloud** and only then projected. This is what
produces realistic non-uniform sparsity, range-dependent attenuation and spurious returns,
rather than painting noise onto a 2D depth map.

## Severity presets

`lidar/configs/severity_presets/` holds the physical parameters for each weather type and
severity level, with the measured mean point retention for each setting.

| Weather | light | moderate | heavy |
|---|---|---|---|
| Fog / haze — extinction β (1/m) | 0.01 | 0.1 | 0.2 |
| Rain — rate (mm/h) | 10 | 100 | 200 |
| Snow — rate (mm/h) | 0.5 | 1.5 | 2.5 |
| Wet ground — pavement / water / floor | .00012 / .00002 / .10 | .00015 / .00012 / .10 | .00150 / .00020 / .20 |

Measured mean point retention: fog 97.99 % → ~91 % → ~76 %; rain 86.56 % at light.

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

Before first use, fetch the five physical weather models and the snow particle fields from
their original authors; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[lidar/weather_models/PORTING.md](lidar/weather_models/PORTING.md).

## Annotation schema

`annotations/{train,val,test}.csv` carry one row per frame, joined to every modality by the
`name` column.

| Column | Values |
|---|---|
| `weather` | `Clear`, `Fog`, `Rain`, `Snow` |
| `severity` | `" None"` (clear frames), `light`, `moderate`, `heavy` |
| `Time_of_Day` | `Day`, `Night` |
| `Lens_Condition` | `Clean`, `raindrop_occluded`, `snowflake_occluded` |
| `Scenario` | `City`, `Residential`, `Road` |
| `Ground_Condition` | `Dry`, `Wet`, `Snow_Cover` |
| `Sky_Condition` | `None`, `Rainfall`, `Snowfall` |

Note that `severity` for clear frames is the string `" None"` with a leading space; strip
whitespace before comparing.

## Realism validation

`realism/` holds the quantitative evidence that the synthetic weather behaves like real weather:
FID and KID against MUSES and DENSE, nine physically-grounded image descriptors (dark channel,
Koschmieder transmission, airlight, contrast, …), and Weather-Shift Consistency, which compares
*differences* — weather minus clear — between synthetic and real data rather than raw appearance.

`realism/lidar_degradation/` reproduces the published LiDAR degradation analysis by applying the
denoising masks of the DCD formulation to paired clean and weather sparse projections, reporting
dropout, spurious, perturbed and preserved rates, effective sensing range and near-range clutter
per weather and severity.
