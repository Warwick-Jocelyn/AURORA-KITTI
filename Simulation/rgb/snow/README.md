# Snow rendering

Camera-side snow for AURORA-KITTI: falling snow rendered through a physical sensor model, and
lens-adherent snowflakes composited onto the image.

## Components

| File | Role |
|---|---|
| `fallingsnow_main.py` | Depth-aware falling-snow renderer. Reads RGB + 8-bit depth (`MAX_DEPTH = 80 m`), converts the colour frame to a Bayer CFA mosaic, applies the snow and wind models in sensor space, then demosaics with Malvar (2004). |
| `Snowflakes_gen.py` | Lens-adherent snowflakes. Builds an RGBA sprite layer from a grayscale flake mask (Gaussian blur, `ALPHA_MAX = 180`), punches alpha holes for spatial variation, and tints flakes per channel from a blurred random field. Structural twin of `../Rain_generation/Raindrops_gen.py`. |
| `snowmask.py` | Severity-dependent mask compositing from a directory of flake sprites. |
| `models/00_weather_class.py` | Particle physics — `cam_rain`, `cam_snow`, `Fog_Param`. Snow uses the rain framework's variables with snow-specific velocity and size distributions. |
| `models/07_wind.py` | Wind-driven streak geometry: lookup tables, FOV cuboid sampling, frame-to-pixel projection, and a numba-accelerated streak rasteriser (`@jit(nopython=True, parallel=True)`). |
| `models/00_cam_class.py` | Sensor models — `Cam_Param` (Lucid Triton TRI028S-CC, 80.8°×61.6° FoV, 1936×1464, 4.5 µm pixels, f/2.1, 6 mm), plus KITTI and ACDC variants. |
| `checking_depth_and_save_txt.py` | Prepares the depth index consumed by `main.py`. |
| `INVOCATIONS.txt` | Recorded run commands. |

## Module assembly

`fallingsnow_main.py` resolves its two physics classes dynamically:

```python
weather_class = getattr(importlib.import_module("noise_class"), 'cam_snow')
noise_class   = getattr(importlib.import_module("noise_class"), 'cam_wind')
```

`noise_class` is an aggregator module that re-exports from the `models/` package. `cam_snow` is
defined in `models/00_weather_class.py`. The wind component draws on the routines in
`models/07_wind.py`. Assemble `noise_class.py` to expose both names before running, or import
them directly and bypass `importlib`.

## Running

```bash
python3 checking_depth_and_save_txt.py    # build the depth index
python3 main.py img_bayer <image_dir> depth <depth_index.txt>
python3 fallingsnow_main.py               # falling snow  -> output_snow_<severity>/
python3 Snowflakes_gen.py                 # lens-adherent snowflakes
```

Frame selection is driven by the weather annotation CSVs, filtered on weather and severity —
for example `df[(df.weather=='Snow') & (df.severity=='light')]`.

Dependencies beyond the common stack: `colour_demosaicing`, `numba`, `Pillow`, `tqdm`.
