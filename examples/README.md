# Examples for slides (Input vs Output + timing)

## 1) Save six portraits

Use the same **base name** for each pair, e.g. `p01` … `p06`.

| Step | Where to save | Example |
|------|----------------|---------|
| Input (what you fed the app) | `examples/input/` | `p01.jpg` |
| Colorized (Download result) | `examples/output/` | `p01.jpg` or `p01_colorized.jpg` |

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`.

## 2) Build slide-ready grids (PowerPoint-ready PNGs)

From **project root** (`Re-Animate AI`):

```powershell
cd "C:\Users\DELL\OneDrive\Desktop\Re-Animate AI"
python scripts\make_grids.py
```

Grids appear in `examples/grids/` as `p01_grid.png`, etc.

(Optional) Taller panels for 4K slides:

```powershell
python scripts\make_grids.py --height 900
```

Requires **Python** with **OpenCV** (same as colorization backend). If needed:

```powershell
pip install opencv-python numpy
```

## 3) Runtime table (Slide 8)

`examples/PRESENTATION_SLIDE_DATA.md` is **pre-filled** with measured times. To re-run on your PC:

1. Terminal A: `cd colorization-app\backend` → `python app.py` (wait until the model is ready).  
2. Terminal B:  
   `python scripts\benchmark_colorize.py --ids p01,p02,p03,p04,p05,p06`  

Requires `pip install requests` if needed.

## 4) Failure cases (Slide 9)

Pick **3** bad outputs, paste thumbnails into Slide 9, and copy the short explanations from `PRESENTATION_SLIDE_DATA.md` (edit to match what you actually see).
