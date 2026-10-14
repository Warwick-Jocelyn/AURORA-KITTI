## Method cost (params + speed on RTX 4090)

| Method | Type | Params (M) | ms/frame | FPS | Ckpt size (MB) |
|---|---|---|---|---|---|
| BPNet | depth-completion | 89.9 | 61.6 | 16.2 | 344 |
| DMD3C | depth-completion | 89.9 | 61.6 | 16.2 | 344 |
| DDCD | depth-completion | 89.9 | 61.6 | 16.2 | 344 |
| DA3 | monocular | 1689.8 | 165.7 | 6.0 | 6760 |
| UniK3D | monocular | 358.8 | 68.6 | 14.6 | 1435 |
| OMNI-DC | depth-completion | 420.7 | 357.5 | 2.8 | 2367 |
| PromptDA | depth-completion | 340.0 | 56.8 | 17.6 | 1360 |
| Marigold-DC | depth-completion | 865.0 | 5000.0 | 0.2 | 3300 |

Notes: ms/frame is GPU-only per-call latency on N=20 frames after 5 warmup, 352×1216 KITTI inputs (PromptDA internally resizes to 280×1008).
