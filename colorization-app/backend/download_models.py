#!/usr/bin/env python3
"""
download_models.py — downloads all pretrained model weights.

Usage:
    python download_models.py [--all] [--colorize] [--gfpgan] [--realesrgan] [--codeformer]
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ─── Model Registry ───────────────────────────────────────────────────────────
MODELS = {
    "colorize_proto": {
        "url":  "https://github.com/richzhang/colorization/raw/caffe/colorization/models/colorization_deploy_v2.prototxt",
        "dest": MODELS_DIR / "colorization_deploy_v2.prototxt",
        "desc": "Colorization prototxt (Zhang et al.)",
    },
    "colorize_model": {
        "url":  "https://github.com/spmallick/learnopencv/releases/download/Colorization/colorization_release_v2.caffemodel",
        "dest": MODELS_DIR / "colorization_release_v2.caffemodel",
        "desc": "Colorization caffemodel (Zhang et al.)",
    },
    "colorize_pts": {
        "url":  "https://github.com/richzhang/colorization/raw/caffe/colorization/resources/pts_in_hull.npy",
        "dest": MODELS_DIR / "pts_in_hull.npy",
        "desc": "Colorization cluster centres",
    },
    "gfpgan": {
        "url":  "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        "dest": MODELS_DIR / "GFPGANv1.4.pth",
        "desc": "GFPGAN v1.4 restoration model",
    },
    "realesrgan_x2": {
        "url":  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "dest": MODELS_DIR / "RealESRGAN_x2plus.pth",
        "desc": "Real-ESRGAN x2 super resolution",
    },
    "realesrgan_x4": {
        "url":  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "dest": MODELS_DIR / "RealESRGAN_x4plus.pth",
        "desc": "Real-ESRGAN x4 super resolution",
    },
    "codeformer": {
        "url":  "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "dest": MODELS_DIR / "codeformer.pth",
        "desc": "CodeFormer face enhancement",
    },
}

# ─── Download Helpers ─────────────────────────────────────────────────────────

def _progress(blocks_done, block_size, total_size):
    if total_size <= 0:
        print(f"\r  Downloaded {blocks_done * block_size // 1024} KB …", end="", flush=True)
        return
    pct  = min(100, blocks_done * block_size * 100 // total_size)
    done = min(blocks_done * block_size, total_size)
    mb   = done / (1024 * 1024)
    tot  = total_size / (1024 * 1024)
    bar  = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{bar}] {pct:3d}%  {mb:.1f}/{tot:.1f} MB", end="", flush=True)


def download(key: str):
    entry = MODELS[key]
    dest: Path = entry["dest"]
    url:  str  = entry["url"]
    desc: str  = entry["desc"]

    if dest.exists():
        print(f"  ✓ {desc} already exists – skipping.")
        return

    print(f"\n⬇  Downloading {desc}")
    print(f"   Source  : {url}")
    print(f"   Dest    : {dest}")
    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
        print(f"\n  ✓ Saved → {dest}")
    except Exception as exc:
        print(f"\n  ✗ FAILED: {exc}")
        if dest.exists():
            dest.unlink()

# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download ReAnimateAI model weights")
    parser.add_argument("--all",         action="store_true", help="Download everything")
    parser.add_argument("--colorize",    action="store_true", help="Colorization DNN")
    parser.add_argument("--gfpgan",      action="store_true", help="GFPGAN restoration")
    parser.add_argument("--realesrgan",  action="store_true", help="Real-ESRGAN x2 + x4")
    parser.add_argument("--codeformer",  action="store_true", help="CodeFormer enhancement")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        print("\n  Tip: run with --all to download everything.\n")
        sys.exit(0)

    print("\n══════════════════════════════════════════")
    print("  ReAnimateAI — Model Downloader")
    print("══════════════════════════════════════════")

    if args.all or args.colorize:
        download("colorize_proto")
        download("colorize_model")
        download("colorize_pts")

    if args.all or args.gfpgan:
        download("gfpgan")

    if args.all or args.realesrgan:
        download("realesrgan_x2")
        download("realesrgan_x4")

    if args.all or args.codeformer:
        download("codeformer")

    print("\n══════════════════════════════════════════")
    print("  Done! Start the server: python app.py")
    print("══════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
