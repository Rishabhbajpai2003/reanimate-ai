#!/usr/bin/env python3
"""
Build slide-ready side-by-side PNGs: Input | Output.

Put matching files in:
  examples/input/p01.jpg … p06.jpg   (your saved inputs from the app / originals)
  examples/output/p01.jpg …          (downloaded colorized results, same base name)

Run from project root:
  python scripts/make_grids.py

Outputs:
  examples/grids/p01_grid.png …
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np


def find_output_file(output_dir: str, stem: str) -> str | None:
    exts = (".jpg", ".jpeg", ".png", ".webp")
    candidates = [f"{stem}{e}" for e in exts]
    candidates += [f"{stem}_colorized{e}" for e in exts]
    candidates += [f"colorized_{stem}{e}" for e in exts]
    candidates += [f"colorized-{stem}{e}" for e in exts]
    for name in candidates:
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            return path
    return None


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    if h == target_h:
        return img
    scale = target_h / h
    nw = max(1, int(round(w * scale)))
    nh = target_h
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    """Dark bar at bottom with white text."""
    h, w = img.shape[:2]
    bar = 36
    out = np.zeros((h + bar, w, 3), dtype=np.uint8)
    out[:h, :w] = img
    out[h:, :] = (32, 32, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = h + (bar + th) // 2
    cv2.putText(out, text, (x, y), font, scale, (240, 240, 245), thickness, cv2.LINE_AA)
    return out


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_in = os.path.join(root, "examples", "input")
    default_out = os.path.join(root, "examples", "output")
    default_grids = os.path.join(root, "examples", "grids")

    ap = argparse.ArgumentParser(description="Make Input|Output slide grids.")
    ap.add_argument("--input-dir", default=default_in)
    ap.add_argument("--output-dir", default=default_out)
    ap.add_argument("--grids-dir", default=default_grids)
    ap.add_argument(
        "--height",
        type=int,
        default=720,
        help="Target height per panel (before label bar).",
    )
    ap.add_argument(
        "--gap",
        type=int,
        default=16,
        help="White pixels between Input and Output.",
    )
    args = ap.parse_args()

    os.makedirs(args.grids_dir, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    inputs = []
    for name in sorted(os.listdir(args.input_dir)):
        stem, ext = os.path.splitext(name.lower())
        if ext not in exts:
            continue
        inputs.append((stem, os.path.join(args.input_dir, name)))

    if not inputs:
        print(f"No images found in {args.input_dir}", file=sys.stderr)
        print("Add files like p01.jpg, p02.jpg, …", file=sys.stderr)
        return 1

    made = 0
    for stem, in_path in inputs:
        out_path = find_output_file(args.output_dir, stem)
        if not out_path:
            print(f"Skip {stem}: no matching output in {args.output_dir}")
            continue

        a = cv2.imread(in_path)
        b = cv2.imread(out_path)
        if a is None or b is None:
            print(f"Skip {stem}: could not read image")
            continue

        a = resize_to_height(a, args.height)
        b = resize_to_height(b, args.height)
        # Same height; widths may differ — pad narrower side to max width for alignment
        max_w = max(a.shape[1], b.shape[1])
        def pad_w(img):
            h, w = img.shape[:2]
            if w == max_w:
                return img
            pad = np.zeros((h, max_w, 3), dtype=np.uint8)
            x0 = (max_w - w) // 2
            pad[:, x0 : x0 + w] = img
            return pad

        a = pad_w(a)
        b = pad_w(b)

        a = add_label(a, "Input")
        b = add_label(b, "Output")

        gap = np.ones((a.shape[0], args.gap, 3), dtype=np.uint8) * 255
        grid = np.hstack([a, gap, b])

        dest = os.path.join(args.grids_dir, f"{stem}_grid.png")
        cv2.imwrite(dest, grid)
        print(f"Wrote {dest}")
        made += 1

    if made == 0:
        return 1
    print(f"Done. {made} grid(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
