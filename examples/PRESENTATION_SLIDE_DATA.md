# Slide 8 — Runtime (measured)

Timed with `scripts/benchmark_colorize.py` against a running backend (`python app.py` in `colorization-app/backend`). Same settings as the web UI defaults for **Balanced** quality: `grayscale_only=1`, `clahe=0`, `saturation=1.08`, sharpness auto.

**Slide set (6 images for the main results slide):**

| Image ID | Seconds |
|----------|---------|
| p01 | 4.16 |
| p02 | 2.35 |
| p03 | 1.65 |
| p04 | 1.29 |
| p05 | 1.63 |
| p06 | 2.64 |

**Copy onto Slide 8:**

- **Average:** 2.29 s / image  
- **Min:** 1.29 s  
- **Max:** 4.16 s  
- **Setup note:** Local Flask + OpenCV DNN (Zhang et al. Caffe model), **Balanced** preset, CPU inference; end-to-end server time per POST (not only model forward pass).

**Grids for Slide 8:** insert `examples/grids/p01_grid.png` … `p06_grid.png` (Input | Output).

**Extra images (not in the table above):** `p07`, `p08` are available for backup slides or failure analysis; grids exist in `examples/grids/`.

---

# Slide 9 — Three failure cases (assigned + slide text)

Pick the **three grids** below (or swap if your visual check disagrees). Each explanation matches a **real limitation** of luminance→chroma models, tied to the **photo you used**.

## Failure A — Color / ambiguity  
**Photo:** **p03** (Lionel Messi)  
**Grid:** `examples/grids/p03_grid.png`  
**1–2 sentences for the slide:** Striped jerseys and busy backgrounds collapse to similar gray values, so more than one color assignment is plausible; the model can lock onto plausible but not exact team or kit hues.

## Failure B — Detail / high-frequency regions (halos, texture)  
**Photo:** **p06** (Brad Pitt — highest resolution in the set, 1683×2244)  
**Grid:** `examples/grids/p06_grid.png`  
**1–2 sentences for the slide:** Fine hair and film-grain-like detail create rapid luminance changes; chroma is predicted at a coarser scale and then blended back, which can produce mild halos or uneven saturation along hair and shadow boundaries.

## Failure C — Blur / very low resolution  
**Photo:** **p04** (Leonardo DiCaprio — smallest image in the set, 366×489 px)  
**Grid:** `examples/grids/p04_grid.png`  
**1–2 sentences for the slide:** With very few pixels, facial structure is under-specified; the network still predicts chroma, which can look slightly waxy or “invented” compared with a sharp, high-res capture.

---

## Optional checklist (viva)

- **Success:** plausible global colors, skin without extreme tint shifts, no harsh seams at edges.  
- **Failure:** wrong clothing/background hue, plastic skin on tiny inputs, or color fringing on hair/edges.
