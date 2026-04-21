"""
Restoration Module — denoising + artifact removal.

Primary  : OpenCV Non-Local Means Denoising
Advanced : GFPGAN (if installed and model available)
"""

import logging
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger("pipeline.restore")


class RestoreModule:
    def __init__(self):
        self._gfpgan = None
        self._try_load_gfpgan()

    # ── GFPGAN (optional) ─────────────────────────────────────────────────────
    def _try_load_gfpgan(self):
        try:
            from gfpgan import GFPGANer
            model_path = Path(__file__).parent.parent / "models" / "gfpgan" / "GFPGANv1.4.pth"
            if not model_path.exists():
                model_path = Path(__file__).parent.parent / "models" / "GFPGANv1.4.pth"
            if model_path.exists():
                self._gfpgan = GFPGANer(
                    model_path=str(model_path),
                    upscale=1,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None,
                )
                logger.info("GFPGAN loaded ✓")
            else:
                logger.info("GFPGAN model not found – using OpenCV fallback")
        except ImportError as err:
            logger.info("realesrgan/basicsr not installed or import error: %s – using OpenCV fallback", err)
        except Exception as exc:
            logger.warning("Real-ESRGAN load error: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────
    def process(self, input_path: str, output_path: str) -> str:
        """Denoise and restore the portrait. Returns output_path."""
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {input_path}")

        img = self._resize_if_large(img)

        if self._gfpgan is not None:
            result = self._run_gfpgan(img, input_path)
        else:
            result = self._run_opencv(img)

        cv2.imwrite(output_path, result)
        logger.info("Restoration saved → %s", output_path)
        return output_path

    # ── GFPGAN path ───────────────────────────────────────────────────────────
    def _run_gfpgan(self, img: np.ndarray, input_path: str) -> np.ndarray:
        try:
            logger.info("Starting GFPGAN inference...")
            cropped_faces, restored_faces, restored_img = self._gfpgan.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
            logger.info("GFPGAN inference complete. Faces detected: %d", len(cropped_faces) if cropped_faces else 0)
            if restored_img is not None:
                return restored_img
            return img
        except Exception as exc:
            logger.warning("GFPGAN inference error (%s) – falling back to OpenCV", exc)
            return self._run_opencv(img)

    # ── OpenCV path ───────────────────────────────────────────────────────────
    @staticmethod
    def _run_opencv(img: np.ndarray) -> np.ndarray:
        """Fast Non-Local Means denoising."""
        denoised = cv2.fastNlMeansDenoisingColored(
            img,
            None,
            h=4,
            hColor=4,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        # Light sharpening
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        sharp = cv2.filter2D(denoised, -1, kernel)
        return sharp

    # ── Utilities ─────────────────────────────────────────────────────────────
    @staticmethod
    def _resize_if_large(img: np.ndarray, max_dim: int = 1024) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
            logger.debug("Resized to %dx%d", img.shape[1], img.shape[0])
        return img
