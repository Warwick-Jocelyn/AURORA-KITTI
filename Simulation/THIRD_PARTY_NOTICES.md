# Third-party notices — Simulation

The AURORA-KITTI weather simulation builds on published physics code from several groups. This
file records exactly what came from where, so the provenance of every physical model is explicit.

## LiDAR weather models — NOT redistributed here

The five physical weather models are third-party academic code, each under its own licence.
They are not redistributed here. Fetch each from its upstream and place it in
`lidar/weather_models/`:

| Module | Upstream | Paper | Licence |
|---|---|---|---|
| `snow.py` | [SysCV/LiDAR_snow_sim](https://github.com/SysCV/LiDAR_snow_sim) | Hahner et al., *LiDAR Snowfall Simulation for Robust 3D Object Detection*, CVPR 2022 | non-commercial research |
| `fog.py` | [MartinHahner/LiDAR_fog_sim](https://github.com/MartinHahner/LiDAR_fog_sim) | Hahner et al., *Fog Simulation on Real LiDAR Point Clouds*, ICCV 2021 | non-commercial research |
| `haze.py` | [princeton-computational-imaging/SeeingThroughFog](https://github.com/princeton-computational-imaging/SeeingThroughFog) (`tools/DatasetFoggification/beta_modification.py`) | Bijelic et al., *Seeing Through Fog Without Seeing Fog*, CVPR 2020 | non-commercial research |
| `rain.py` | [velatkilic/LISA](https://github.com/velatkilic/LISA) | Kilic et al., *LISA: A Package for Lidar Simulation in Adverse Weather* (arXiv 2107.07004) | MIT |
| `wet_ground.py` | [SysCV/LiDAR_snow_sim](https://github.com/SysCV/LiDAR_snow_sim) (wet-ground module, after Rasshofer et al. 2011) | Hahner et al., CVPR 2022 | non-commercial research |

Our local copies carry small adaptations: each was fitted to the `weather_model_base`
interface in `lidar/weather_models/base.py`; `snow.py` had its multiprocessing path replaced
with a serial channel loop; `rain.py` carries a SciPy `trapz` → `trapezoid` compatibility shim.
Those adaptations are described in `lidar/weather_models/PORTING.md`.

## Required data assets — not redistributed

| Asset | Size | Source |
|---|---|---|
| `weather_models/npy/` — 5,120 pre-sampled snow particle fields (`gunn_*`, `sekhon_*`) | 2.3 GB | Hahner `LiDAR_snow_sim` |
| KITTI raw Velodyne scans (the simulation input) | 44 GB | [cvlibs.net](https://www.cvlibs.net/datasets/kitti/) |

Included here because they are small and slow to regenerate: the nine fog integral lookup
tables (`lidar/pre_computed_materials/fog/*.pickle`, one per allowed α) and the Mie scattering
cache for rain.

## Framework provenance

The LiDAR simulation framework originates from the WMG ROADVIEW project
(`lidar_compound_noise_models`), developed under the EU Horizon ROADVIEW grant. Files
originating there — `lidar.py`, `util.py`, `internal_noise_models/` — are included with that
attribution. `compound_model.py` was substantially rewritten for AURORA-KITTI (wet-ground token
parsing, main-weather normalisation), and `simulate_from_csv.py` is new.

Please cite the ROADVIEW project alongside this work when using the LiDAR simulation
framework.

## RGB weather rendering

| Component | Origin |
|---|---|
| Fog rendering (`rgb/fog_simulation_pipeline.py`) | ours — transmission map + β attenuation |
| Adherent raindrop occlusion (`rgb/Rain_generation/Raindrops_gen.py`) | ours |
| Rain streak rendering | [astra-vision/rain-rendering](https://github.com/astra-vision/rain-rendering) @ `0a9d8ca7` — Halder et al. ICCV 2019 / Tremblay et al. IJCV 2020, **Apache-2.0**. Not vendored; clone upstream. |
| Snow rendering (`rgb/snow/`) | ours — falling snow through a sensor model, plus lens-adherent snowflakes |

## Dataset

AURORA-KITTI derives from KITTI, so the **data** inherits **CC BY-NC-SA 3.0** (non-commercial,
share-alike, attribution to Geiger et al.) regardless of this repository's code licence.
