import logging
import os
import random
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger("pipeline.animate")

SADTALKER_DIR = Path(__file__).parent.parent.parent / "SadTalker"

def _resolve_sadtalker_python() -> Path:
    # Windows venv: sadtalker_env/Scripts/python.exe
    # Unix venv   : sadtalker_env/bin/python
    win_py = SADTALKER_DIR / "sadtalker_env" / "Scripts" / "python.exe"
    if win_py.exists():
        return win_py
    return SADTALKER_DIR / "sadtalker_env" / "bin" / "python"


SADTALKER_PYTHON = _resolve_sadtalker_python()
MODEL_PATH = Path(__file__).parent.parent / "models" / "face_landmarker.task"

FPS = 10
DURATION_SEC = 2.0
TOTAL_FRAMES = int(FPS * DURATION_SEC)


class AnimateModule:
    def __init__(self):
        exists = SADTALKER_DIR.exists()
        inf_exists = (SADTALKER_DIR / "inference.py").exists()
        self._sadtalker_available = exists and inf_exists

        if self._sadtalker_available:
            logger.info("SadTalker integration ACTIVE (found at %s)", SADTALKER_DIR.absolute())
        else:
            logger.warning("SadTalker NOT FOUND (checked %s). Using static blink fallback.", SADTALKER_DIR.absolute())

        self._model_path = str(MODEL_PATH) if MODEL_PATH.exists() else None
        if not self._model_path:
            logger.warning("Face landmarker model not found at %s – eye detection will use fallback", MODEL_PATH)

    def process(self, input_path: str, output_path: str, audio_path: Optional[str] = None) -> str:
        if self._sadtalker_available:
            return self._run_sadtalker(input_path, output_path, audio_path)

        gif_path = str(Path(output_path).with_suffix(".gif"))
        return self._generate_static_blink_gif(input_path, gif_path)

    def _detect_eye_boxes_once(self, frame):
        if not self._model_path:
            return self._fallback_eye_boxes(frame.shape[1], frame.shape[0])

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        with FaceLandmarker.create_from_options(options) as landmarker:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if not result.face_landmarks:
                return self._fallback_eye_boxes(frame.shape[1], frame.shape[0])

            landmarks = result.face_landmarks[0]
            h, w = frame.shape[:2]

            def get_box(indices):
                xs = [landmarks[i].x * w for i in indices]
                ys = [landmarks[i].y * h for i in indices]

                x1, x2 = int(min(xs)), int(max(xs))
                y1, y2 = int(min(ys)), int(max(ys))

                pad_x = int((x2 - x1) * 0.3)
                pad_y = int((y2 - y1) * 0.6)

                return (
                    max(0, x1 - pad_x),
                    max(0, y1 - pad_y),
                    min(w, x2 + pad_x),
                    min(h, y2 + pad_y),
                )

            left_eye = [33, 160, 158, 133, 153, 144]
            right_eye = [362, 385, 387, 263, 373, 380]
            mouth = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375]

            return [get_box(left_eye), get_box(right_eye)], get_box(mouth)

    @staticmethod
    def _fallback_eye_boxes(w, h):
        eyes = [
            (int(w * 0.22), int(h * 0.26), int(w * 0.46), int(h * 0.40)),
            (int(w * 0.54), int(h * 0.26), int(w * 0.78), int(h * 0.40)),
        ]
        mouth = (int(w * 0.35), int(h * 0.65), int(w * 0.65), int(h * 0.85))
        return eyes, mouth

    @staticmethod
    def _apply_blink(frame, boxes, strength):
        if strength <= 0:
            return frame

        out = frame.copy()

        for (x1, y1, x2, y2) in boxes:
            roi = out[y1:y2, x1:x2].copy()
            h_r, w_r = roi.shape[:2]
            if h_r <= 0 or w_r <= 0:
                continue

            closed_h = max(1, int(h_r * (1 - 0.9 * strength)))
            squashed = cv2.resize(roi, (w_r, closed_h))

            padded = np.zeros_like(roi)
            top = (h_r - closed_h) // 2
            padded[top:top + closed_h] = squashed

            mask = np.zeros((h_r, w_r), dtype=np.uint8)
            cv2.ellipse(mask, (w_r // 2, h_r // 2), (int(w_r * 0.42), int(h_r * 0.34)), 0, 0, 360, 255, -1)

            sigma_mask = max(1.0, h_r * 0.05)
            mask = cv2.GaussianBlur(mask, (0, 0), sigma_mask)

            alpha = (mask / 255.0)[:, :, None] * strength
            blended = roi * (1 - alpha) + padded * alpha

            shadow = np.zeros((h_r, w_r), dtype=np.uint8)
            cv2.ellipse(shadow, (w_r // 2, int(h_r * 0.42)), (int(w_r * 0.4), int(h_r * 0.12)), 0, 0, 360, 255, -1)

            sigma_shadow = max(1.0, h_r * 0.07)
            shadow = cv2.GaussianBlur(shadow, (0, 0), sigma_shadow)

            blended *= (1 - (shadow / 255.0)[:, :, None] * 0.2 * strength)
            out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        return out

    @staticmethod
    def _apply_lips(frame, mouth_box, strength):
        if strength <= 0:
            return frame

        x1, y1, x2, y2 = mouth_box
        out = frame.copy()
        roi = out[y1:y2, x1:x2].copy()
        h_r, w_r = roi.shape[:2]
        if h_r <= 0 or w_r <= 0:
            return frame

        new_h = int(h_r * (1 + 0.15 * strength))
        stretched = cv2.resize(roi, (w_r, new_h))

        start = (new_h - h_r) // 2
        cropped = stretched[start:start + h_r, :]

        mask = np.zeros((h_r, w_r), dtype=np.uint8)
        cv2.ellipse(mask, (w_r // 2, h_r // 2), (int(w_r * 0.45), int(h_r * 0.35)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), h_r * 0.1)

        alpha = (mask / 255.0)[:, :, None] * strength
        blended = roi * (1 - alpha) + cropped * alpha

        out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return out

    def _generate_static_blink_gif(self, img_path, output_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not load image at {img_path}")

            eye_boxes, mouth_box = self._detect_eye_boxes_once(img)

            blink_centers = sorted(random.uniform(0.2, 0.8) for _ in range(random.randint(3, 4)))
            lip_centers = sorted(random.uniform(0.1, 0.9) for _ in range(random.randint(8, 12)))

            frames = []
            for i in range(TOTAL_FRAMES):
                t = i / TOTAL_FRAMES

                blink_strength = 0
                for c in blink_centers:
                    blink_strength = max(blink_strength, max(0, 1 - abs(t - c) / 0.04))

                lip_strength = 0
                for c in lip_centers:
                    lip_strength = max(lip_strength, max(0, 1 - abs(t - c) / 0.03))

                frame = self._apply_blink(img.copy(), eye_boxes, blink_strength)
                frame = self._apply_lips(frame, mouth_box, lip_strength)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            from PIL import Image

            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / FPS),
                loop=0,
            )

            return output_path
        except Exception:
            logger.error("Static blink GIF generation failed", exc_info=True)
            raise
    def _run_sadtalker(self, img_path, output_path, audio_path):
        try:
            if not audio_path:
                audio_path = str(SADTALKER_DIR / "examples" / "driven_audio" / "yash.wav")

            result_dir = Path(output_path).parent / "tmp"
            result_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                str(SADTALKER_PYTHON),
                "inference.py",
                "--driven_audio",
                str(Path(audio_path).absolute()),
                "--source_image",
                str(Path(img_path).absolute()),
                "--result_dir",
                str(result_dir.absolute()),
                "--preprocess",
                "crop",
                "--still",
                "--size",
                "256",
                "--expression_scale",
                "0.6",
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = str(SADTALKER_DIR)
            env.pop("PYTHONHOME", None)

            logger.info("Running SadTalker in isolated env...")
            subprocess.run(
                cmd,
                check=True,
                cwd=str(SADTALKER_DIR),
                env=env,
            )

            mp4_files = sorted(result_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            if mp4_files:
                final_output = str(Path(output_path).with_suffix(".mp4"))
                shutil.move(str(mp4_files[0]), final_output)
                return final_output

            raise RuntimeError("SadTalker did not generate output")

        except Exception as exc:
            logger.warning("SadTalker failed (%s) -> fallback", exc)
            logger.debug("SadTalker traceback: %s", traceback.format_exc())
            return self._generate_static_blink_gif(img_path, str(Path(output_path).with_suffix(".gif")))
