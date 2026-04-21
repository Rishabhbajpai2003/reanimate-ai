"""
Super Resolution Module.

Supports model selection and SR ablation-friendly comparison runs across:
    - Real-ESRGAN
    - SwinIR
    - HAT

If a model is unavailable, the module falls back to an OpenCV profile and
returns metadata so clients can surface this to users.
"""

import logging
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("pipeline.super_res")

# Target upscale factor
SCALE = 2
SUPPORTED_MODELS = ("realesrgan", "swinir", "hat")


def normalize_model_name(name: str) -> str:
    """Map user-provided model aliases to canonical names."""
    key = (name or "").strip().lower().replace("-", "_").replace(" ", "")
    aliases = {
        "realesrgan": "realesrgan",
        "real_esrgan": "realesrgan",
        "swinir": "swinir",
        "hat": "hat",
    }
    return aliases.get(key, "")


class SuperResModule:
    def __init__(self):
        self._models_dir = Path(__file__).parent.parent / "models"

        self._torch = None
        self._device = "cpu"

        self._upsampler = None
        self._spandrel_loader = None
        self._native_models: Dict[str, object] = {}

        self._init_torch()
        self._try_load_realesrgan()
        self._try_init_spandrel_loader()
        self._try_load_spandrel_model("swinir")
        self._try_load_spandrel_model("hat")

    def _init_torch(self):
        try:
            import torch

            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("SR runtime device: %s", self._device)
        except Exception as exc:
            logger.warning("Torch unavailable for native SR: %s", exc)

    def _model_path(self, model_name: str) -> Path:
        if model_name == "swinir":
            return self._models_dir / "001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
        if model_name == "hat":
            return self._models_dir / "HAT_SRx2_ImageNet-pretrain.pth"
        return self._models_dir / f"RealESRGAN_x{SCALE}plus.pth"

    # ── Real-ESRGAN (optional) ────────────────────────────────────────────────
    def _try_load_realesrgan(self):
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model_path = self._model_path("realesrgan")
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

    def _try_init_spandrel_loader(self):
        try:
            from spandrel import ModelLoader

            # Registers extra architectures, including HAT/SwinIR variants.
            import spandrel_extra_arches  # noqa: F401

            self._spandrel_loader = ModelLoader()
            logger.info("Spandrel model loader initialised ✓")
        except Exception as exc:
            logger.info("Spandrel not available for SwinIR/HAT (%s)", exc)

    def _try_load_spandrel_model(self, model_name: str):
        if self._spandrel_loader is None:
            return

        model_path = self._model_path(model_name)
        if not model_path.exists():
            logger.info("%s checkpoint missing at %s", model_name, model_path)
            return

        try:
            descriptor = self._spandrel_loader.load_from_file(model_path)
            # Keep descriptors on CPU by default to avoid filling small GPUs
            # when multiple SR models are loaded at startup.
            if hasattr(descriptor, "to"):
                descriptor.to("cpu")
            if hasattr(descriptor, "eval"):
                descriptor.eval()
            self._native_models[model_name] = descriptor
            arch_name = getattr(getattr(descriptor, "architecture", None), "name", "unknown")
            logger.info("%s native checkpoint loaded (%s) ✓", model_name, arch_name)
        except Exception as exc:
            logger.warning("Failed to load %s checkpoint (%s): %s", model_name, model_path.name, exc)

    # ── Public API ────────────────────────────────────────────────────────────
    def process(self, input_path: str, output_path: str, model_name: str = "realesrgan") -> Dict[str, str]:
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {input_path}")

        chosen = normalize_model_name(model_name) or "realesrgan"
        result, backend = self._run_model(img, chosen)

        cv2.imwrite(output_path, result)
        logger.info("Super-res saved (%s/%s) → %s", chosen, backend, output_path)
        return {
            "output_path": output_path,
            "model": chosen,
            "backend": backend,
        }

    def process_compare(self, input_path: str, output_dir: str, job_id: str, model_names: List[str]) -> Dict[str, Dict[str, str]]:
        """Run SR for all requested models and save one output per model."""
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {input_path}")

        canonical: List[str] = []
        for model in model_names:
            m = normalize_model_name(model)
            if m and m not in canonical:
                canonical.append(m)

        if not canonical:
            canonical = ["realesrgan"]

        outputs: Dict[str, Dict[str, str]] = {}
        base = Path(output_dir)
        for model in canonical:
            t0 = time.perf_counter()
            error_msg = None
            try:
                result, backend = self._run_model(img, model)
            except Exception as exc:
                # Do not fail the entire ablation run for one model.
                error_msg = str(exc)
                logger.warning("%s SR failed (%s) — using OpenCV fallback", model, exc)
                result = self._run_opencv(img, profile=model)
                backend = "fallback-opencv"

            out_filename = f"{job_id}_super_res_{model}.png"
            out_path = str(base / out_filename)
            cv2.imwrite(out_path, result)
            outputs[model] = {
                "filename": out_filename,
                "backend": backend,
                "latency_s": round(time.perf_counter() - t0, 3),
            }
            if error_msg:
                outputs[model]["error"] = error_msg
            logger.info("Super-res compare saved (%s/%s) → %s", model, backend, out_path)

        return outputs

    def _run_model(self, img: np.ndarray, model_name: str):
        if model_name == "realesrgan":
            if self._upsampler is not None:
                return self._run_realesrgan(img), "native"
            return self._run_opencv(img, profile="realesrgan"), "fallback-opencv"

        if model_name == "swinir":
            if "swinir" in self._native_models:
                return self._run_spandrel(img, "swinir")
            return self._run_opencv(img, profile="swinir"), "fallback-opencv"

        if model_name == "hat":
            if "hat" in self._native_models:
                return self._run_spandrel(img, "hat")
            return self._run_opencv(img, profile="hat"), "fallback-opencv"

        logger.warning("Unknown SR model '%s'; using Real-ESRGAN fallback profile", model_name)
        return self._run_opencv(img, profile="realesrgan"), "fallback-opencv"

    # ── Real-ESRGAN path ──────────────────────────────────────────────────────
    def _run_realesrgan(self, img: np.ndarray) -> np.ndarray:
        try:
            output, _ = self._upsampler.enhance(img, outscale=SCALE)
            return output
        except Exception as exc:
            logger.warning("Real-ESRGAN inference error (%s) – fallback", exc)
            return self._run_opencv(img, profile="realesrgan")

    def _run_spandrel(self, img: np.ndarray, model_name: str):
        if self._torch is None:
            raise RuntimeError("Torch is not available for native model inference")

        descriptor = self._native_models.get(model_name)
        if descriptor is None:
            raise RuntimeError(f"Model '{model_name}' is not loaded")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        backend = "native"

        def _infer_on(device_name: str):
            if hasattr(descriptor, "to"):
                descriptor.to(device_name)
            tensor = self._torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device_name)
            with self._torch.inference_mode():
                out = descriptor(tensor)
            return out

        try:
            output = _infer_on(self._device)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if self._device == "cuda" and "out of memory" in msg:
                logger.warning("%s CUDA OOM — retrying on CPU", model_name)
                self._torch.cuda.empty_cache()
                output = _infer_on("cpu")
                backend = "native-cpu-fallback"
            else:
                raise
        finally:
            # Keep descriptor on CPU between calls to minimize persistent VRAM usage.
            if hasattr(descriptor, "to"):
                descriptor.to("cpu")
            if self._device == "cuda":
                self._torch.cuda.empty_cache()

        output = output.squeeze(0).detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        output_u8 = (output * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(output_u8, cv2.COLOR_RGB2BGR), backend

    # ── OpenCV fallback ───────────────────────────────────────────────────────
    @staticmethod
    def _run_opencv(img: np.ndarray, profile: str = "realesrgan") -> np.ndarray:
        h, w = img.shape[:2]
        upscaled = cv2.resize(
            img,
            (w * SCALE, h * SCALE),
            interpolation=cv2.INTER_LANCZOS4,
        )

        # Model-specific fallback profiles so ablation UI still shows distinct outputs.
        if profile == "swinir":
            den = cv2.fastNlMeansDenoisingColored(upscaled, None, 3, 3, 7, 21)
            blur = cv2.GaussianBlur(den, (0, 0), sigmaX=1.2)
            return cv2.addWeighted(den, 1.18, blur, -0.18, 0)

        if profile == "hat":
            den = cv2.bilateralFilter(upscaled, d=7, sigmaColor=45, sigmaSpace=45)
            blur = cv2.GaussianBlur(den, (0, 0), sigmaX=2.0)
            return cv2.addWeighted(den, 1.28, blur, -0.28, 0)

        # Real-ESRGAN-like sharper fallback.
        blur = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=3)
        return cv2.addWeighted(upscaled, 1.5, blur, -0.5, 0)
