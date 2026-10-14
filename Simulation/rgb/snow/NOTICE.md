# Snow RGB rendering — status

The two scripts that rendered AURORA-KITTI's camera-side snow (`fallingsnow_main.py` for
falling snow, `Snowflakes_gen.py` for lens-adherent snowflakes) were lost in a disk cleanup
and are not recoverable from this machine.

What survives, and is sufficient to reconstruct them:

| Asset | What it gives you |
|---|---|
| A working ancestor of the compositing pipeline (cv2 `absdiff` + `addWeighted`) | the rendering structure |
| Vendored `imgaug` `class SnowflakesLayer` / `class Snowflakes` | the flake physics — `flake_size`, `density`, `angle`, `speed`, `_motion_blur`. The class name matches `Snowflakes_gen.py`, so this is near-certainly its upstream ancestor. |
| Snowflake sprite sets (large / medium / small) | the actual flake textures used |
| `../Rain_generation/Raindrops_gen.py` | the structural twin — the surviving lens-occlusion renderer for rain |
| `annotations/*.csv` → `Lens_Condition = snowflake_occluded` | which frames carry lens occlusion |

Two different snow generators were used for the released dataset: a procedural one (47,673
`mix_snow` frames, "falling snowflakes") and a GAN-based one (`snowgan_fixed_46968`). Which
produced the published frames is not currently recorded.

The released snow **data** is unaffected — only the generating code is missing. Reconstruction
is tracked as an open item; see the repository root README.
