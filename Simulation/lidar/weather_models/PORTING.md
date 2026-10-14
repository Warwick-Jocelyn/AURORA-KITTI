# Porting the weather models

`lidar/weather_models/` ships only `base.py` (our adapter) and `__init__.py`. The five physical
models are third-party; see `../../THIRD_PARTY_NOTICES.md` for sources and licences.

After fetching each upstream file into this directory, apply these adaptations:

1. **All five** — conform to the `weather_model_base` interface in `base.py`: a single
   `simulate(pcl, intensity)` entry point. `base.py` also performs the spherical-projection
   zero-depth infill in `pre_process`.
2. **`snow.py`** — the upstream multiprocessing path is commented out in favour of a serial
   per-channel loop. It reads `./weather_models/npy` and
   `./configs/sensor_params/20171102_64E_S3.yaml` by **relative** path, so the simulator must be
   run from the repository root.
3. **`rain.py`** — SciPy compatibility shim: `scipy.integrate.trapz` → `trapezoid`.
4. **`fog.py`** — `alpha` is hard-constrained to `{0.005, 0.01, 0.02, 0.03, 0.06, 0.1, 0.12,
   0.15, 0.2}` by the precomputed integral tables in `../pre_computed_materials/fog/`.

A sixth file, an alternative LISA-based snow model, existed locally as `snow(LISA).py`. It was
dead code — the filename is not a legal Python module name and its parameters are absent from
`snow_params.yaml`, so it could not run. Not carried over.
