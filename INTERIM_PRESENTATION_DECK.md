# ReAnimateAI — Interim Presentation (Engineering Track)

> How to use: paste each slide section into PowerPoint/Canva/Google Slides.  
> Main deck is **10 slides max**. Backup slides follow at the end.

---

## Slide 1 — Title
**ReAnimateAI: Intelligent Portrait Restoration Framework**  
**Track:** Engineering  
**Team:** Keyurkumar Desai (MT25071), Ishan Kumar (MT25069), Rishabh Bajpai (MT25079), Yash Kumar Saini (MT25054)  
**1-line pitch:** Upload an old/degraded portrait → restore + upscale + colorize + enhance face → optionally generate subtle facial animation.

---

## Slide 2 — Problem statement, scope, users
- **Problem**: Historical/grayscale portraits often have multiple degradations (noise, scratches, blur, low resolution, fading) and lack color/motion; manual restoration is slow and expert-driven.
- **Goal**: Build an integrated system that performs:
  - **Image restoration** (noise/scratch/artifact removal)
  - **Super-resolution**
  - **Colorization**
  - **Face enhancement**
  - **Photo animation** (subtle, realistic facial motion: blinking / slight head movement)
- **Input**: single portrait photo (RGB or grayscale; often degraded).
- **Outputs**: restored image, SR image, colorized image, face-enhanced image, short animation clip (if enabled).
- **Primary users**: families/archivists/creators needing quick restoration in a simple web interface.
- **Scope limits**: focuses on portraits/archival photos; performance may degrade on extremely damaged images or non-portrait/group scenes; animation limited to subtle motion.

---

## Slide 3 — Related work & baseline 1 (Colorization)
**DeOldify (GAN-based colorization; review/implementation e.g., arXiv:2101.04061 / IPOL review)**  
- **Key idea**: Learned generative priors to map grayscale → color with visually pleasing outputs.
- **Why it’s relevant**: strong practical baseline for “old photo” colorization.
- **Methodological shortcomings (gaps)**:
  - **Hallucination risk**: produces plausible but historically/semantically incorrect colors.
  - **Ambiguity**: multiple valid colors for the same luminance structure (clothes, background, sky).
  - **Domain shift sensitivity**: film grain/scratches/fading can mislead the generator.

---

## Slide 4 — Related work & baseline 2 (Blind face restoration)
**CodeFormer (NeurIPS) + generative facial prior family (GFP-like priors)**  
- **Key idea**: Use priors/codebook lookups to restore realistic face details from degraded inputs.
- **Why it’s relevant**: portraits are the primary target; face quality dominates perceived realism.
- **Methodological shortcomings (gaps)**:
  - **Identity–realism trade-off**: sharpening can change identity (over-regularization toward “average” faces).
  - **Over-synthesis**: adds details that were never present (problematic for archival fidelity).
  - **Fragility**: severe occlusions/scratches can cause artifacts.

---

## Slide 5 — Related work & baseline 3 (Animation)
**First Order Motion Model (FOMM; arXiv:2003.00196)** and **Bringing Old Photos Back to Life (CVPR 2020)**  
- **Key idea**: motion transfer from a driving video/latent motion representation to animate a static image.
- **Why it’s relevant**: “re-animation” is a key differentiator beyond restoration.
- **Methodological shortcomings (gaps)**:
  - **Temporal instability** (flicker/warping) when keypoints are uncertain.
  - **Identity drift** across frames (especially with large head turns).
  - **Limited control** for subtle “realistic” motion without artifacts.

---

## Slide 6 — Dataset & evaluation metrics
### Datasets (from proposal; used for different modules)
- **ImageNet (subset)**: controlled experiments; generate grayscale inputs from GT color for paired evaluation.
- **HumanFace8000 (Kaggle)**: portrait-focused evaluation for face enhancement/restoration.
- **VoxCeleb2**: motion/animation testing (driving signals; diverse identities).
- **Public-domain historical archive**: real-world qualitative validation (domain shift).

### Metrics (module-wise + system)
- **Restoration**: PSNR, SSIM, LPIPS
- **Colorization**: PSNR/SSIM (when GT exists), perceptual evaluation, small-scale human study
- **Face enhancement**: identity preservation score, landmark consistency
- **Animation**: temporal consistency metric, frame stability analysis, identity preservation across frames
- **System-level**: end-to-end inference time, GPU memory usage, FPS, module-wise latency breakdown

---

## Slide 7 — System overview (end-to-end)
### User-facing workflow (Engineering Track emphasis)
1. User uploads portrait in web UI
2. Backend runs pipeline modules (toggleable)
3. UI displays intermediate outputs + final result + download

### Architecture (block diagram)
**Web UI** → **API** → **Preprocess** → {Restoration → SR → Colorization → Face Enhancement} → **(optional) Animation** → **Results**

### Key engineering constraints
- Deterministic dependency setup (no runtime installs)
- Clear module boundaries to swap baselines
- Measurable latency per module + overall

---

## Slide 8 — Baseline results (what to present now)
> Put a 2×3 grid of images here (even if only 3–6 examples).

- **Qualitative grid** (per example):
  - Input degraded/grayscale
  - After restoration
  - After SR
  - After colorization
  - After face enhancement
  - (Optional) animation frame montage / GIF

- **Quantitative (early)**:
  - Table of PSNR/SSIM/LPIPS on a small paired subset (ImageNet subset or synthetic degradation).
- **Runtime**:
  - Per-module latency + end-to-end time on your available compute (CPU/GPU).

---

## Slide 9 — Error analysis & failure cases
### Failure case categories (show at least 3)
- **Heavy scratches/noise** → incomplete removal or unnatural textures
- **Blur/low-res** → SR hallucination; face detail mismatch
- **Color ambiguity** → wrong clothing/skin/background hues; desaturation
- **Animation** → flicker/warping around mouth/eyes; identity instability

### Likely causes (tie to observations)
- Domain shift of archival photos vs training data
- Ambiguity in mapping luminance → chroma
- Priors overpowering identity cues in face restoration
- Keypoint/motion estimation uncertainty in animation

---

## Slide 10 — Next steps + individual tasks (include a table)
### Next steps (component breakdown)
- **Pipeline integration**: unify modules behind a single API with toggles
- **Evaluation harness**: dataset splits + EDA + metrics computation + latency logging
- **UI polish**: upload, progress states, intermediate results, download; optional model selector
- **Robustness**: input validation, file size limits, graceful failures
- **Animation**: stabilize and constrain motion for subtle realism; add temporal smoothing

### Individual ownership (edit as needed)
| Member | Owned component | Done so far | Next tasks |
|---|---|---|---|
| Keyurkumar | Restoration + evaluation harness | Defined restoration goals \& degradation types; set up metric plan (PSNR/SSIM/LPIPS) and dataset shortlist (ImageNet subset + historical portraits). | Implement restoration baseline (noise/scratch removal); run paired evaluation on synthetic degradations; compile failure-case taxonomy + visuals for Slide 9. |
| Ishan | Super-resolution + performance profiling | Shortlisted SR baseline(s) and integration plan; set up measurement plan for latency/VRAM and output quality comparisons. | Integrate SR module into pipeline; benchmark latency + memory; add module-wise latency breakdown and 3–5 qualitative SR examples for slides. |
| Rishabh | Colorization + web UI integration | Integrated working colorization web demo (upload → backend → output image); ensured deterministic setup and clean frontend build/audit. | Add UI to display intermediate pipeline outputs + downloads; add model/module toggles; collect 6 demo examples + 3 failure cases with short explanations. |
| Yash | Animation (FOMM / old-photo animation) | Reviewed animation baselines (FOMM, CVPR’20 old-photo animation) and defined “subtle motion” target (blink/slight head movement) + evaluation plan. | Integrate animation baseline; add temporal smoothing/stability constraints; report FPS + temporal stability metrics; prepare 1–2 short demo clips + failure analysis. |

---

# Backup slides (Q&A)

## Backup 1 — Module interfaces (inputs/outputs)
- Restoration: image → image
- SR: image → higher-res image
- Colorization: grayscale/RGB → color image
- Face enhancement: image → face-enhanced image
- Animation: identity image + driving signal → video/GIF

## Backup 2 — Compute requirements & deployment
- **Dev env**: Python 3.10+, PyTorch, OpenCV; notebook + VSCode
- **Hardware**: 16GB RAM min; NVIDIA GPU 8GB+ recommended; 20–30GB storage
- **Compute options**: Colab Pro / Kaggle GPUs / institute servers / local GPU
- **Deployment**: Flask/FastAPI backend; web frontend; optional Streamlit demo

## Backup 3 — Human study plan (small-scale)
- Pairwise preference: baseline vs improved module output
- Criteria: realism, identity fidelity, temporal stability (for animation)
*** End Patch"}}
