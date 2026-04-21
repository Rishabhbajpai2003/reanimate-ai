"""
Super Resolution Module — Real-ESRGAN x2 / x4 upscaling.

If the Real-ESRGAN model is unavailable, falls back to OpenCV
Lanczos4 / EDSR-style bicubic upsampling.
"""

import logging
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger("pipeline.super_res")

# Target upscale factor
SCALE = 2


class SuperResModule:
    def __init__(self):
        self._upsampler = None
        self._try_load_realesrgan()

    # ── Real-ESRGAN (optional) ────────────────────────────────────────────────
    def _try_load_realesrgan(self):
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model_path = (
                Path(__file__).parent.parent
                / "models"
                / f"RealESRGAN_x{SCALE}plus.pth"
            )
            if not model_path.exists():
                logger.info("Real-ESRGAN model not found – using OpenCV fallback")
                return

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=SCALE,
            )
            self._upsampler = RealESRGANer(
                scale=SCALE,
                model_path=str(model_path),
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=(device == "cuda"),
                device=device,
            )
            logger.info("Real-ESRGAN x%d loaded on %s ✓", SCALE, device)
        except ImportError:
            logger.info("realesrgan/basicsr not installed – using OpenCV fallback")
        except Exception as exc:
            logger.warning("Real-ESRGAN load error: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────
    def process(self, input_path: str, output_path: str) -> str:
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {input_path}")

        if self._upsampler is not None:
            result = self._run_realesrgan(img)
        else:
            result = self._run_opencv(img)

        cv2.imwrite(output_path, result)
        logger.info("Super-res saved → %s", output_path)
        return output_path

    # ── Real-ESRGAN path ──────────────────────────────────────────────────────
    def _run_realesrgan(self, img: np.ndarray) -> np.ndarray:
        try:
            output, _ = self._upsampler.enhance(img, outscale=SCALE)
            return output
        except Exception as exc:
            logger.warning("Real-ESRGAN inference error (%s) – fallback", exc)
            return self._run_opencv(img)

    # ── OpenCV fallback ───────────────────────────────────────────────────────
    @staticmethod
    def _run_opencv(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        upscaled = cv2.resize(
            img,
            (w * SCALE, h * SCALE),
            interpolation=cv2.INTER_LANCZOS4,
        )
        # Unsharp mask for crisp edges
        blurred  = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=3)
        return cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
