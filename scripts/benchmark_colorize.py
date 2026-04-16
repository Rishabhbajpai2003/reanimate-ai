#!/usr/bin/env python3
"""
Time POST /colorize for each image in examples/input (same form as the web UI).

1) Start backend:  cd colorization-app\backend && python app.py
2) Run:             python scripts/benchmark_colorize.py

Prints a markdown table + avg/min/max.
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5000/colorize")
    ap.add_argument(
        "--input-dir",
        default=os.path.join(ROOT, "examples", "input"),
    )
    ap.add_argument(
        "--ids",
        default="p01,p02,p03,p04,p05,p06",
        help="Comma-separated stems (no extension), e.g. p01,p02,...",
    )
    ap.add_argument("--quality", default="balanced")
    args = ap.parse_args()

    stems = [s.strip() for s in args.ids.split(",") if s.strip()]
    times: list[tuple[str, float]] = []

    for stem in stems:
        path = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            cand = os.path.join(args.input_dir, stem + ext)
            if os.path.isfile(cand):
                path = cand
                break
        if not path:
            print(f"Missing {stem} in {args.input_dir}", file=sys.stderr)
            continue

        with open(path, "rb") as f:
            files = {"image": (os.path.basename(path), f, "image/jpeg")}
            data = {
                "grayscale_only": "1",
                "quality": args.quality,
                "clahe": "0",
                "saturation": "1.08",
            }
            t0 = time.perf_counter()
            r = requests.post(args.url, files=files, data=data, timeout=300)
            elapsed = time.perf_counter() - t0

        if r.status_code != 200:
            print(f"{stem}: HTTP {r.status_code} {r.text[:200]}", file=sys.stderr)
            continue

        times.append((stem, elapsed))
        print(f"{stem}: {elapsed:.2f}s")

    if not times:
        return 1

    vals = [t for _, t in times]
    print()
    print("| Image ID | Seconds |")
    print("|----------|---------|")
    for stem, sec in times:
        print(f"| {stem} | {sec:.2f} |")
    print()
    print(
        f"**Average:** {statistics.mean(vals):.2f} s/image  \n"
        f"**Min:** {min(vals):.2f} s  \n"
        f"**Max:** {max(vals):.2f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
