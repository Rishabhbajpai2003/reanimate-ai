# Interim report → PDF (CVPR-style)

**Main file:** `INTERIM_REPORT_CVPR24.tex`  
It includes rubric sections (i)–(viii), measured latency table, failure analysis, and **supplementary figures** (grids from `examples/grids/`).

## Option A — Overleaf (easiest if you have no LaTeX on Windows)

1. Zip this folder **or** upload `INTERIM_REPORT_CVPR24.tex` plus the folder `examples/grids/` (keep the same relative paths).
2. Set **Main document** to `INTERIM_REPORT_CVPR24.tex`.
3. **Menu → Compiler → pdfLaTeX.**
4. Click **Recompile** twice (references).
5. **Download PDF.**

## Option B — Local (MiKTeX or TeX Live on Windows)

1. Install [MiKTeX](https://miktex.org/) or [TeX Live](https://tug.org/texlive/).
2. Open **PowerShell**:

```powershell
cd "C:\Users\DELL\OneDrive\Desktop\Re-Animate AI"
pdflatex -interaction=nonstopmode INTERIM_REPORT_CVPR24.tex
pdflatex -interaction=nonstopmode INTERIM_REPORT_CVPR24.tex
```

3. Output: `INTERIM_REPORT_CVPR24.pdf`

**If `\includegraphics` fails:** compile from the repo root (so `examples/grids/...` exists), or move/copy the `examples` folder next to the `.tex` file.

## Option C — Official CVPR 2024 template (submission polish)

1. Download the **official CVPR 2024 LaTeX template**.
2. Copy the **section text** from `INTERIM_REPORT_CVPR24.tex` into the template’s `main.tex` (replace `\documentclass{article}` block with the template’s preamble + `\begin{document}` structure).
3. Keep **main narrative ≤ 3 pages** if your course requires it; leave figures in the **supplementary** part as the rubric allows.

## Submit

Upload **PDF only** (no `.docx`).
