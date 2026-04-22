import logging
import random
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import mediapipe as mp
import traceback
import os

SADTALKER_DIR = Path(__file__).parent.parent.parent / "SadTalker"
SADTALKER_PYTHON = SADTALKER_DIR / "sadtalker_env" / "bin" / "python"

logger = logging.getLogger("pipeline.animate")

SADTALKER_DIR = Path(__file__).parent.parent.parent / "SadTalker"
MODEL_PATH = Path(__file__).parent.parent / "models" / "face_landmarker.task"

FPS = 10
DURATION_SEC = 2.0
TOTAL_FRAMES = int(FPS * DURATION_SEC)


class AnimateModule:
    def __init__(self):
        # Improved detection with absolute path resolution for logging
        exists = SADTALKER_DIR.exists()
        inf_exists = (SADTALKER_DIR / "inference.py").exists()
        self._sadtalker_available = exists and inf_exists
        
        if self._sadtalker_available:
            logger.info("SadTalker integration ACTIVE (found at %s)", SADTALKER_DIR.absolute())
        else:
            logger.warning("SadTalker NOT FOUND (checked %s). Using static blink fallback.", SADTALKER_DIR.absolute())

        # New MediaPipe Tasks API (0.10.30+)
        self._model_path = str(MODEL_PATH) if MODEL_PATH.exists() else None
        if not self._model_path:
            logger.warning("Face landmarker model not found at %s – eye detection will use fallback", MODEL_PATH)

    def process(self, input_path: str, output_path: str, audio_path: Optional[str] = None) -> str:
        if self._sadtalker_available:
            return self._run_sadtalker(input_path, output_path, audio_path)

        gif_path = str(Path(output_path).with_suffix(".gif"))
        return self._generate_static_blink_gif(input_path, gif_path)

    # ─────────────────────────────────────────────
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

            LEFT_EYE  = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]
            MOUTH     = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375]

            return [get_box(LEFT_EYE), get_box(RIGHT_EYE)], get_box(MOUTH)

    # ─────────────────────────────────────────────
    def _fallback_eye_boxes(self, w, h):
        eyes = [
            (int(w*0.22), int(h*0.26), int(w*0.46), int(h*0.40)),
            (int(w*0.54), int(h*0.26), int(w*0.78), int(h*0.40)),
        ]
        mouth = (int(w*0.35), int(h*0.65), int(w*0.65), int(h*0.85))
        return eyes, mouth

    # ─────────────────────────────────────────────
    def _apply_blink(self, frame, boxes, strength):
        if strength <= 0:
            return frame

        out = frame.copy()

        for (x1, y1, x2, y2) in boxes:
            roi = out[y1:y2, x1:x2].copy()
            h_r, w_r = roi.shape[:2]
            if h_r <= 0 or w_r <= 0: continue

            closed_h = max(1, int(h_r * (1 - 0.9 * strength)))
            squashed = cv2.resize(roi, (w_r, closed_h))

            padded = np.zeros_like(roi)
            top = (h_r - closed_h) // 2
            padded[top:top+closed_h] = squashed

            mask = np.zeros((h_r, w_r), dtype=np.uint8)
            cv2.ellipse(mask, (w_r//2, h_r//2),
                        (int(w_r*0.42), int(h_r*0.34)),
                        0, 0, 360, 255, -1)
            
            # Scale blur by ROI height (approx 3.0 for 512x512 eyes)
            sigma_mask = max(1.0, h_r * 0.05)
            mask = cv2.GaussianBlur(mask, (0,0), sigma_mask)

            alpha = (mask/255.0)[:, :, None] * strength
            blended = roi*(1-alpha) + padded*alpha

            # eyelid shadow
            shadow = np.zeros((h_r, w_r), dtype=np.uint8)
            cv2.ellipse(shadow, (w_r//2, int(h_r*0.42)),
                        (int(w_r*0.4), int(h_r*0.12)),
                        0, 0, 360, 255, -1)
            
            sigma_shadow = max(1.0, h_r * 0.07)
            shadow = cv2.GaussianBlur(shadow, (0,0), sigma_shadow)

            blended *= (1 - (shadow/255.0)[:, :, None]*0.2*strength)

            out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        return out

    # ─────────────────────────────────────────────
    def _apply_lips(self, frame, mouth_box, strength):
        if strength <= 0:
            return frame
            
        x1, y1, x2, y2 = mouth_box
        out = frame.copy()
        roi = out[y1:y2, x1:x2].copy()
        h_r, w_r = roi.shape[:2]
        if h_r <= 0 or w_r <= 0: return frame

        # Subtle vertical stretch for "talking"
        new_h = int(h_r * (1 + 0.15 * strength))
        stretched = cv2.resize(roi, (w_r, new_h))
        
        # Center crop the stretched lip area back to original ROI size
        start = (new_h - h_r) // 2
        cropped = stretched[start:start+h_r, :]
        
        # Soft mask for blending
        mask = np.zeros((h_r, w_r), dtype=np.uint8)
        cv2.ellipse(mask, (w_r//2, h_r//2), (int(w_r*0.45), int(h_r*0.35)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (0,0), h_r*0.1)
        
        alpha = (mask/255.0)[:, :, None] * strength
        blended = roi*(1-alpha) + cropped*alpha
        
        out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return out

    # ─────────────────────────────────────────────
    def _generate_static_blink_gif(self, img_path, output_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not load image at {img_path}")

            res = self._detect_eye_boxes_once(img)
            logger.info("Detect result type: %s, length: %d", type(res), len(res))
            eye_boxes, mouth_box = res

            # random blink timings
            blink_centers = sorted(random.uniform(0.2, 0.8) for _ in range(random.randint(3,4)))
            # random lip timings (more frequent for talking)
            lip_centers = sorted(random.uniform(0.1, 0.9) for _ in range(random.randint(8,12)))

            frames = []
            for i in range(TOTAL_FRAMES):
                t = i / TOTAL_FRAMES

                # Blink strength
                b_strength = 0
                for c in blink_centers:
                    b_strength = max(b_strength, max(0, 1 - abs(t-c)/0.04))

                # Lip strength
                l_strength = 0
                for c in lip_centers:
                    l_strength = max(l_strength, max(0, 1 - abs(t-c)/0.03))

                frame = self._apply_blink(img.copy(), eye_boxes, b_strength)
                frame = self._apply_lips(frame, mouth_box, l_strength)
                
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000/FPS),
                loop=0
            )

            return output_path
        except Exception as exc:
            logger.error("Static blink GIF generation failed", exc_info=True)
            raise
    # SADTALKER_DIR = Path(__file__).parent.parent.parent / "SadTalker"
    # SADTALKER_PYTHON = SADTALKER_DIR / "sadtalker_env" / "bin" / "python"
    print("Python path:", SADTALKER_PYTHON)
    print("Exists:", SADTALKER_PYTHON.exists())
    print("\n🚀 Running SadTalker...")
    print("Python path:", SADTALKER_PYTHON)
    print("Exists:", SADTALKER_PYTHON.exists())

    


    def _run_sadtalker(self, img_path, output_path, audio_path):
        try:
            # fallback audio if none
            if not audio_path:
                audio_path = str(SADTALKER_DIR / "examples" / "driven_audio" / "yash.wav")

            result_dir = Path(output_path).parent / "tmp"
            result_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                str(SADTALKER_PYTHON),
                "inference.py",
                "--driven_audio", str(Path(audio_path).absolute()),
                "--source_image", str(Path(img_path).absolute()),
                "--result_dir", str(result_dir.absolute()),

                # 🔥 FAST SETTINGS
                "--preprocess", "crop",          # already good
                "--still",                       # minimal head motion
                "--size", "256",                 # ⬅️ BIGGEST SPEED BOOST
                "--expression_scale", "0.6",     # less computation
                # "--enhancer", "none",            # disable GFPGAN (very important)
            ]

            # 🔥 isolate environment
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SADTALKER_DIR)
            env.pop("PYTHONHOME", None)

            print("🚀 Running SadTalker in isolated env...")
            subprocess.run(
                cmd,
                check=True,
                cwd=str(SADTALKER_DIR),
                env=env
            )

            # find output
            mp4_files = list(result_dir.rglob("*.mp4"))
            if mp4_files:
                final_output = str(Path(output_path).with_suffix(".mp4"))
                shutil.move(str(mp4_files[0]), final_output)
                return final_output

            raise RuntimeError("SadTalker did not generate output")

        except Exception as e:
            print("\n❌ SadTalker failed → fallback")
            traceback.print_exc()
            print("Error message:", e)
            return self._generate_static_blink_gif(img_path, output_path)
            
    # ─────────────────────────────────────────────
    # def _run_sadtalker(self, img_path, output_path, audio_path):
    #     try:
    #         # Use a default audio file if none provided to trigger AI animation
    #         if not audio_path:
    #             audio_path = str(SADTALKER_DIR / "examples" / "driven_audio" / "imagine.wav")
    #             if not Path(audio_path).exists():
    #                 logger.warning("Default audio not found, falling back to static blink")
    #                 return self._generate_static_blink_gif(img_path, output_path)

    #         result_dir = str(Path(output_path).parent / "tmp")
            
    #         # Use absolute paths because we change CWD
    #         abs_img = str(Path(img_path).absolute())
    #         abs_audio = str(Path(audio_path).absolute())
    #         abs_result = str(Path(result_dir).absolute())

    #         cmd = [
    #             str(Path(__file__).parent.parent / ".venv" / "bin" / "python"),
    #             "inference.py",
    #             "--driven_audio", abs_audio,
    #             "--source_image", abs_img,
    #             "--result_dir", abs_result,
    #             "--preprocess", "full",
    #             "--enhancer", "gfpgan",
    #             "--background_enhancer", "realesrgan", # Better background
    #             "--expression_scale", "1.1",           # More natural expression
    #         ]
    #         # Note: Removed --still to allow natural head/neck motion
    #         logger.info("Running SadTalker (Best Practices) in %s: %s", SADTALKER_DIR, " ".join(cmd))
    #         subprocess.run(cmd, check=True, cwd=str(SADTALKER_DIR))

    #         mp4_files = list(Path(result_dir).rglob("*.mp4"))
    #         if mp4_files:
    #             # Ensure output_path has .mp4 extension for clarity if it's a video
    #             final_output = str(Path(output_path).with_suffix(".mp4"))
    #             shutil.move(str(mp4_files[0]), final_output)
    #             logger.info("SadTalker SUCCESS: Result saved to %s", final_output)
                
    #             # Cleanup temp alignment frames to save space
    #             try:
    #                 shutil.rmtree(result_dir)
    #                 logger.info("Temporary frames cleaned up.")
    #             except Exception:
    #                 pass

    #             return final_output
            
    #         logger.warning("SadTalker finished but no MP4 was found in %s – falling back to static blink", result_dir)
    #         return self._generate_static_blink_gif(img_path, output_path)

    #     except Exception as exc:
    #         logger.error("SadTalker CRASHED: %s", exc, exc_info=True)
    #         return self._generate_static_blink_gif(img_path, output_path)