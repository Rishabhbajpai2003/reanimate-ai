"""
Animation Module — bring a still portrait to life.

Creates a looping GIF with combined motion effects:
  1. Slow Ken-Burns zoom + pan
  2. Subtle breathing / sway (sinusoidal warp)
  3. Simulated eye blink via brief darkening
  4. Gentle head-tilt oscillation

No external models required — pure OpenCV + NumPy.
If SadTalker is installed, it will be preferred.
"""

import logging
import math
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("pipeline.animate")

SADTALKER_DIR = Path(__file__).parent.parent.parent / "SadTalker"

# ─── Configuration ────────────────────────────────────────────────────────────
FPS            = 20
DURATION_SEC   = 4.0
TOTAL_FRAMES   = int(FPS * DURATION_SEC)
OUTPUT_SIZE    = (512, 512)  # GIF frame size


class AnimateModule:
    def __init__(self):
        self._sadtalker_available = (
            SADTALKER_DIR.exists() and (SADTALKER_DIR / "inference.py").exists()
        )
        if self._sadtalker_available:
            logger.info("SadTalker found at %s ✓", SADTALKER_DIR)
        else:
            logger.info("SadTalker not found — will generate motion-effect GIF")

    # ── Public API ────────────────────────────────────────────────────────────
    def process(
        self,
        input_path: str,
        output_path: str,
        audio_path: Optional[str] = None,
    ) -> str:
        """
        Animate a portrait.  Returns the path to the output GIF/MP4.

        * If SadTalker + audio → talking head MP4
        * Otherwise → motion-effect GIF (Ken Burns + breathing + blink)
        """
        if self._sadtalker_available and audio_path:
            return self._run_sadtalker(input_path, output_path, audio_path)

        # Always produce a .gif regardless of whatever extension was passed
        gif_path = str(Path(output_path).with_suffix(".gif"))
        return self._generate_motion_gif(input_path, gif_path)

    # ── Motion-effect GIF (the main feature) ─────────────────────────────────
    @staticmethod
    def _generate_motion_gif(img_path: str, output_path: str) -> str:
        """
        Generates a lively looping GIF from a still portrait.

        Effects applied per frame:
          • Ken Burns   – slow zoom-in with gentle horizontal drift
          • Breathing   – vertical sinusoidal stretch near the lower half
          • Eye blink   – brief darkening of the upper-third every ~3 s
          • Head tilt   – small affine rotation oscillation
        """
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        # Pre-process: resize to working resolution (keep aspect, pad)
        h, w = img.shape[:2]
        scale = max(OUTPUT_SIZE[0] / w, OUTPUT_SIZE[1] / h) * 1.15  # 15% margin for zoom
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LANCZOS4,
        )
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2

        frames = []

        for i in range(TOTAL_FRAMES):
            t = i / TOTAL_FRAMES          # 0 → 1 over loop
            angle = 2 * math.pi * t       # full cycle

            # ── 1. Ken Burns: gentle zoom 1.00→1.06→1.00 + horizontal drift ──
            zoom = 1.0 + 0.06 * (0.5 - 0.5 * math.cos(angle))
            dx   = 8 * math.sin(angle)       # ±8 px horizontal
            dy   = 4 * math.sin(angle * 0.5)  # ±4 px vertical

            crop_w = int(OUTPUT_SIZE[0] / zoom)
            crop_h = int(OUTPUT_SIZE[1] / zoom)
            x1 = int(cx - crop_w // 2 + dx)
            y1 = int(cy - crop_h // 2 + dy)
            x1 = max(0, min(x1, w - crop_w))
            y1 = max(0, min(y1, h - crop_h))
            crop = img[y1:y1 + crop_h, x1:x1 + crop_w]
            frame = cv2.resize(crop, OUTPUT_SIZE, interpolation=cv2.INTER_LANCZOS4)

            # ── 2. Subtle head-tilt rotation: ±1.5 degrees ──
            tilt = 1.5 * math.sin(angle)
            M_rot = cv2.getRotationMatrix2D(
                (OUTPUT_SIZE[0] // 2, OUTPUT_SIZE[1] // 2), tilt, 1.0
            )
            frame = cv2.warpAffine(
                frame, M_rot, OUTPUT_SIZE,
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            # ── 3. Breathing / sway: sinusoidal vertical stretch in lower half ──
            breath = 0.008 * math.sin(angle * 2)
            cols = np.arange(OUTPUT_SIZE[0], dtype=np.float32)
            rows = np.arange(OUTPUT_SIZE[1], dtype=np.float32)
            map_x, map_y = np.meshgrid(cols, rows)
            frac = map_y / OUTPUT_SIZE[1]
            map_y = map_y + breath * frac * OUTPUT_SIZE[1]
            frame = cv2.remap(
                frame, map_x, map_y.astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            # ── 4. Eye-blink: brief dimming of upper-third every ~loop ──
            blink_cycle = (t * 2) % 1.0      # blink twice per loop
            blink_strength = max(0, 1.0 - abs(blink_cycle - 0.05) / 0.03)
            if blink_strength > 0:
                upper = frame[:OUTPUT_SIZE[1] // 3].copy()
                dark  = (upper.astype(np.float32) * (1 - 0.35 * blink_strength))
                frame[:OUTPUT_SIZE[1] // 3] = np.clip(dark, 0, 255).astype(np.uint8)

            # ── 5. Slight colour-temperature oscillation for "life" ──
            warmth = 3 * math.sin(angle)
            frame_float = frame.astype(np.float32)
            frame_float[:, :, 2] = np.clip(frame_float[:, :, 2] + warmth, 0, 255)  # R
            frame_float[:, :, 0] = np.clip(frame_float[:, :, 0] - warmth, 0, 255)  # B
            frame = frame_float.astype(np.uint8)

            # BGR → RGB for GIF
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ── Write GIF with Pillow ─────────────────────────────────────────────
        try:
            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / FPS),
                loop=0,
                optimize=True,
            )
            logger.info("Motion GIF (%d frames) → %s", len(frames), output_path)
            return output_path
        except ImportError:
            # Fallback: write as MP4 via imageio/ffmpeg
            logger.warning("Pillow not found — trying ffmpeg for MP4")
            return AnimateModule._write_mp4_ffmpeg(frames, output_path)

    # ── ffmpeg MP4 fallback ───────────────────────────────────────────────────
    @staticmethod
    def _write_mp4_ffmpeg(frames, output_path):
        """Write frames list to MP4 via piping raw RGB to ffmpeg."""
        mp4_path = str(Path(output_path).with_suffix(".mp4"))
        h, w = frames[0].shape[:2]
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgb24",
            "-r", str(FPS),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            mp4_path,
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for f in frames:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        proc.wait(timeout=60)
        logger.info("MP4 video (%d frames) → %s", len(frames), mp4_path)
        return mp4_path

    # ── SadTalker path ────────────────────────────────────────────────────────
    def _run_sadtalker(self, img_path, output_path, audio_path):
        try:
            result_dir = str(Path(output_path).parent / "sadtalker_tmp")
            cmd = [
                "python",
                str(SADTALKER_DIR / "inference.py"),
                "--driven_audio", audio_path,
                "--source_image", img_path,
                "--result_dir",   result_dir,
                "--still",
                "--preprocess",   "full",
                "--enhancer",     "gfpgan",
            ]
            subprocess.run(cmd, check=True, timeout=300,
                           capture_output=True, text=True)
            mp4_files = list(Path(result_dir).rglob("*.mp4"))
            if mp4_files:
                shutil.move(str(mp4_files[0]), output_path)
                logger.info("SadTalker video → %s", output_path)
                return output_path
            raise RuntimeError("No MP4 output from SadTalker")
        except Exception as exc:
            logger.warning("SadTalker failed (%s) — falling back to motion GIF", exc)
            gif_path = str(Path(output_path).with_suffix(".gif"))
            return self._generate_motion_gif(img_path, gif_path)
