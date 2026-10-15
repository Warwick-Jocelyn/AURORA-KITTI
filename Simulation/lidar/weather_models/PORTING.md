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

## Note on the Mie cache filename

`../pre_computed_materials/rain/mie_1.328_λ_905.npz` contains a non-ASCII character (λ) in its
filename. This matches what upstream `rain.py` expects, so do not rename it. On Windows or with
tooling that mishandles UTF-8 filenames, check out the repository with
`git config core.quotepath false` and ensure a UTF-8 locale.
