"""
Colorization Module — Zhang et al. ECCV 2016 (OpenCV DNN).

Model files (downloaded via download_models.py):
    models/colorization_deploy_v2.prototxt
    models/colorization_release_v2.caffemodel
    models/pts_in_hull.npy

Falls back to a histogram-stretch colour boost when the DNN model
is unavailable (keeps the image in whatever colour space it was in).
"""

import logging
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger("pipeline.colorize")

_MODEL_CANDIDATE_DIRS = [
    Path(__file__).parent.parent / "models",      # backend/models
    Path(__file__).parent.parent.parent / "models",  # repo/models
]


def _resolve_model_paths():
    for model_dir in _MODEL_CANDIDATE_DIRS:
        prototxt = model_dir / "colorization_deploy_v2.prototxt"
        caffemodel = model_dir / "colorization_release_v2.caffemodel"
        pts_hull = model_dir / "pts_in_hull.npy"
        if prototxt.exists() and caffemodel.exists() and pts_hull.exists():
            return prototxt, caffemodel, pts_hull
    # Default to backend/models for log clarity if nothing is found.
    model_dir = _MODEL_CANDIDATE_DIRS[0]
    return (
        model_dir / "colorization_deploy_v2.prototxt",
        model_dir / "colorization_release_v2.caffemodel",
        model_dir / "pts_in_hull.npy",
    )


PROTOTXT, CAFFEMODEL, PTS_HULL = _resolve_model_paths()


class ColorizeModule:
    def __init__(self):
        self._net   = None
        self._pts   = None
        self._try_load_model()

    # ── Model loading ─────────────────────────────────────────────────────────
    def _try_load_model(self):
        if PROTOTXT.exists() and CAFFEMODEL.exists() and PTS_HULL.exists():
            try:
                net = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(CAFFEMODEL))
                pts = np.load(str(PTS_HULL))
                # Inject cluster centres
                class8  = net.getLayerId("class8_ab")
                conv8   = net.getLayerId("conv8_313_rh")
                pts_in  = pts.transpose().reshape(2, 313, 1, 1).astype(np.float32)
                net.getLayer(class8).blobs = [pts_in]
                net.getLayer(conv8).blobs  = [np.full([1, 313], 2.606, dtype=np.float32)]
                self._net = net
                self._pts = pts
                logger.info("Colorization DNN loaded ✓")
            except Exception as exc:
                logger.warning("Colorization DNN load error: %s", exc)
        else:
            logger.info(
                "Colorization model files not found at %s, %s, %s – will use fallback",
                PROTOTXT,
                CAFFEMODEL,
                PTS_HULL,
            )

    # ── Public API ────────────────────────────────────────────────────────────
    def process(self, input_path: str, output_path: str) -> str:
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {input_path}")

        is_greyscale = self._is_greyscale(img)
        if self._net is not None and is_greyscale:
            result = self._run_dnn(img)
        elif is_greyscale:
            # If model is missing, keep grayscale image natural (no yellow tint).
            logger.warning("Colorization model unavailable for grayscale input; passing image through unchanged")
            result = img
        else:
            # Image already coloured or model missing — apply vibrance boost
            result = self._run_vibrance(img)

        cv2.imwrite(output_path, result)
        logger.info("Colorization saved → %s", output_path)
        return output_path

    # ── DNN colorization ──────────────────────────────────────────────────────
    def _run_dnn(self, img: np.ndarray) -> np.ndarray:
        try:
            # Convert to Lab
            img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_lab  = cv2.cvtColor(img_rgb.astype(np.float32) / 255.0,
                                    cv2.COLOR_RGB2Lab)
            L        = img_lab[:, :, 0]
            L_resized = cv2.resize(L, (224, 224))
            blob     = cv2.dnn.blobFromImage(L_resized - 50)
            self._net.setInput(blob)
            ab_dec   = self._net.forward()[0, :, :, :].transpose(1, 2, 0)
            # Resize ab back
            ab_us    = cv2.resize(ab_dec, (img.shape[1], img.shape[0]))
            result_lab = np.concatenate([L[:, :, np.newaxis], ab_us], axis=2)
            result_rgb = np.clip(
                cv2.cvtColor(result_lab.astype(np.float32), cv2.COLOR_Lab2RGB) * 255,
                0, 255
            ).astype(np.uint8)
            return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            logger.warning("DNN colorize failed (%s) – fallback", exc)
            return self._run_vibrance(img)

    # ── Vibrance fallback ─────────────────────────────────────────────────────
    @staticmethod
    def _run_vibrance(img: np.ndarray) -> np.ndarray:
        """Boost colour saturation for already-coloured images."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _is_greyscale(img: np.ndarray, thresh: float = 5.0) -> bool:
        b, g, r = cv2.split(img)
        channel_delta = (
            float(np.mean(np.abs(b.astype(int) - g.astype(int)))) < thresh
            and float(np.mean(np.abs(b.astype(int) - r.astype(int)))) < thresh
        )
        # Robustness for JPEG artifacts / scanned photos that are nearly grayscale.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat_mean = float(np.mean(hsv[:, :, 1]))
        low_saturation = sat_mean < 12.0
        return channel_delta or low_saturation
