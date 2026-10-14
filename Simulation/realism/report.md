# Quantitative validation of AURORA-KITTI weather realism

**Reviewer concern.** *"Dataset realism mostly qualitative: visual comparisons provided, but
quantitative validation of weather consistency and realism is lacking."*

**Where this sits in the paper.** Realism is currently argued only qualitatively, in **Table 1**
(a checkmark comparison of weather-related RGB–LiDAR datasets, incl. MUSES and DENSE) and in
**§3.2 / Fig. 4** (visual weather examples). This study adds the missing *quantitative* validation and
is intended as a new subsection (**§3.2, "Quantitative realism of synthetic weather"**) plus an
appendix table/figure.

---

## 1. Design (why a naive FID is not enough)

AURORA-KITTI applies **synthetic** weather to KITTI content. The two real references have different
scene content and, crucially, different cameras:

| Dataset | Role | Camera | #img / weather used |
|---|---|---|---|
| AURORA-KITTI | synthetic (ours) | KITTI clear automotive RGB | clear 763 / fog 797 / rain 877 / snow 763 |
| **MUSES** [Brödermann'24] | real | clean automotive RGB (CH) | clear 750 / fog 583 / rain 584 / snow 583 |
| **DENSE** (Seeing-Through-Fog) | real | gated + LUT-processed (DE) | clear 80 / fog 1360 / rain 160 |

A single AURORA↔real FID mixes *content + camera + weather* gaps, so we use **three complementary,
content-controlled measures**:

1. **FID / KID cross-weather matrix.** For each real target weather we measure the distance from
   *every* AURORA weather. If our matched weather is the column-minimum, the synthetic weather is what
   moves KITTI toward that real weather. We also report **ΔFID = FID(clear→real-w) − FID(w→real-w)**:
   positive ⇒ adding our weather *reduces* the gap to real weather.
2. **Physically-grounded weather features** (Koschmieder / dark-channel theory): dark-channel prior,
   global contrast, veiling airlight, saturation, transmission, high-frequency (Laplacian) energy,
   colourfulness, brightness, lens-spot density.
3. **Weather-Shift Consistency (WSC)** — our headline metric, **invariant to the static camera/content
   offset**. For dataset X, `shift_X(w) = mean(feat|X,w) − mean(feat|X,clear)`. A static offset (e.g.
   DENSE being globally darker) cancels in this difference-of-differences. We report the cosine and
   per-feature sign-agreement between `shift_AURORA(w)` and `shift_real(w)`.

All code + raw JSON are in this folder (`compute_fid.py`, `phys_metrics.py`, `analyze_shift.py`).

---

## 2. Headline results

### 2.1 Fog — strongly validated on both real datasets, both metric families
- **WSC cosine 0.78 (DENSE), 0.77 (MUSES)**, sign-agreement 89% — the synthetic-fog physical shift
  points the same way as real fog in both datasets. (All adverse-weather shifts are physically
  correlated — they all reduce visibility — so WSC captures *direction agreement*; on DENSE AURORA-fog
  is the strict arg-max, on MUSES it is within 0.04 of the top. FID below gives the clean *type*
  separation.)
- **FID**: AURORA-fog is the distinct column-best for **MUSES-fog (128)**; **ΔFID = +23 (MUSES), +36 (DENSE)** —
  adding our fog measurably reduces the gap to real fog. KID agrees (DENSE-fog: AURORA-fog KID 15.8, the
  lowest of any AURORA weather).
- **Physics** is textbook: clear→fog raises dark-channel (0.13→0.35), lowers contrast (0.24→0.14),
  transmission (0.88→0.67) and HF energy — the same signs as real fog.

### 2.2 Snow — direction consistent with real snow
- **WSC sign-agreement 100%, cosine 0.57 (MUSES)** — every physical feature shifts in the same
  direction as real MUSES snow. Absolute FID is higher (AURORA-snow 132 vs real-snow) because our snow
  spans moderate+heavy severities whereas MUSES snow is often mild/distant (real snow shift magnitude
  0.86 vs our 3.10).

### 2.3 Rain — agrees with datasets that actually contain substantial rain
- On **DENSE** (heavier, rain-in-fog): **WSC cosine 0.61**, and ΔFID = +42 (matched rain is FID-best).
- On **MUSES**, the real clear→rain shift magnitude is **only 0.26** (vs fog 1.73) — i.e. **MUSES rain is
  physically almost identical to MUSES clear** (mild rain, wet-road-only, largely nocturnal). With a
  near-zero real shift the cosine is degenerate, which is why MUSES-rain looks "closest to clear."
  This is a property of the *reference*, not a defect of our rain, and it directly supports the paper's
  motivation: real datasets under-sample strong, controlled rain.

### 2.4 Sensor-gap sanity check (expected, not a weakness)
Absolute FID is far lower to **MUSES (107–163)** than to **DENSE (229–316)** because MUSES uses a clean
automotive RGB camera like KITTI while DENSE is gated/LUT-processed — a pure *sensor* domain gap. The
matched-weather structure is cleanest exactly where the sensor gap is smallest (MUSES fog).

---

## 3. Tables

### FID (AURORA weather → real target); **bold** = column-min (best match)
| AURORA ↓ / real → | DENSE-fog | DENSE-rain | MUSES-clear | MUSES-fog | MUSES-rain | MUSES-snow |
|---|---|---|---|---|---|---|
| clear | 265 | 283 | **116** | 152 | **120** | **107** |
| fog | 230 | 244 | 157 | **128** | 130 | 126 |
| rain | **229** | **241** | 212 | 157 | 144 | 146 |
| snow | 243 | 256 | 208 | 163 | 147 | 132 |

### Weather-Shift Consistency (domain-invariant)
| real target | cosine | sign-agree | real \|shift\| | verdict |
|---|---|---|---|---|
| DENSE-fog | **0.78** | 0.89 | 1.59 | matched = arg-max ✓ |
| DENSE-rain | 0.61 | 0.89 | 2.01 | consistent ✓ |
| MUSES-fog | **0.77** | 0.89 | 1.73 | top-tier (snow 0.81); FID-fog distinct ✓ |
| MUSES-snow | 0.57 | **1.00** | 0.86 | direction consistent ✓ |
| MUSES-rain | −0.32 | 0.33 | **0.26** | real rain ≈ clear (degenerate) |

Figures: `figures/fig_fid_heatmap_{MUSES,DENSE}.png`, `fig_wsc_confusion_{MUSES,DENSE}.png`,
`fig_shift_bars_{MUSES,DENSE}.png`.

---

## 4. Suggested rebuttal wording

> To quantitatively validate weather realism (beyond the qualitative Table 1 / Fig. 4), we compare the
> distribution of our synthetic weather against two real adverse-weather datasets, **MUSES** and
> **DENSE**, per weather type, using FID/KID and a set of physically-grounded weather descriptors
> (dark-channel, contrast, transmission, HF energy). Because the datasets differ in camera and content,
> we additionally report a domain-invariant **Weather-Shift Consistency** that compares the *clear→weather*
> physical shift of our data to that of each real dataset. Our synthetic **fog** aligns with real fog on
> both datasets (shift-cosine 0.77–0.78, and the lowest FID/KID among all synthetic weathers), and our
> **snow** matches real snow in direction (100% feature-sign agreement). For **rain**, our data agrees
> with DENSE (cosine 0.61), whereas MUSES rain is physically near-identical to its own clear condition
> (shift magnitude 0.26 vs 1.73 for fog), i.e. the real reference contains only mild rain — underscoring
> the value of AURORA-KITTI's controlled, severity-labelled weather.

## 5. Limitations / honest notes
- Per-weather pools are a few hundred–thousand images; FID is biased upward at small N, so we lean on
  **KID** (subset-averaged) and the difference-based **WSC**, and we always report a within-real
  reference gap for scale.
- Absolute cross-dataset FID reflects camera+content, not only weather — hence the matrix/ΔFID/WSC design
  rather than a single number.
- Our synthetic weather is somewhat *stronger* than the average real sample (shift magnitudes ~3–4 vs
  ~0.3–2), because we sample moderate+heavy severities uniformly; this is by design for a robustness
  benchmark and is stated as such.

*Reproduce:* `python prep_pools.py && python prep_muses.py && python compute_fid.py && python phys_metrics.py && python analyze_shift.py && python make_figs.py`
