"""
Face Enhancement Module — CodeFormer.

Primary  : CodeFormer (if codeformer-pytorch is installed and model exists)
Fallback : OpenCV CLAHE + bilateral filter face sharpening
"""

import logging
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger("pipeline.enhance")

MODELS_DIR  = Path(__file__).parent.parent / "models"
CF_WEIGHTS  = MODELS_DIR / "codeformer.pth"


class EnhanceModule:
    def __init__(self):
        self._cf = None
        self._try_load_codeformer()

    # ── CodeFormer (optional) ─────────────────────────────────────────────────
    def _try_load_codeformer(self):
        if not CF_WEIGHTS.exists():
            logger.info("CodeFormer weights not found – using OpenCV fallback")
            return
        try:
            import torch
            from basicsr.utils.download_util import load_file_from_url  # noqa
            # CodeFormer net
            from basicsr.archs.codeformer_arch import CodeFormer
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net = CodeFormer(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            ).to(device)
            ckpt = torch.load(str(CF_WEIGHTS), map_location=device)
            net.load_state_dict(ckpt["params_ema"])
            net.eval()
            self._cf     = net
            self._device = device
            logger.info("CodeFormer loaded on %s ✓", device)
        except ImportError:
            logger.info("codeformer/basicsr not installed – using OpenCV fallback")
        except Exception as exc:
            logger.warning("CodeFormer load error: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────
    def process(self, input_path: str, output_path: str) -> str:
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {input_path}")

        if self._cf is not None:
            result = self._run_codeformer(img)
        else:
            result = self._run_opencv(img)

        cv2.imwrite(output_path, result)
        logger.info("Enhancement saved → %s", output_path)
        return output_path

    # ── CodeFormer path ───────────────────────────────────────────────────────
    def _run_codeformer(self, img: np.ndarray) -> np.ndarray:
        try:
            import torch
            from torchvision.transforms.functional import normalize
            from basicsr.utils import img2tensor, tensor2img
            from basicsr.utils.registry import ARCH_REGISTRY  # noqa

            face_input = cv2.resize(img, (512, 512))
            face_t     = img2tensor(face_input / 255.0, bgr2rgb=True, float32=True)
            normalize(face_t, [0.5] * 3, [0.5] * 3, inplace=True)
            face_t = face_t.unsqueeze(0).to(self._device)
            with torch.no_grad():
                output = self._cf(face_t, w=0.7, adain=True)[0]
            restored = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            return cv2.resize(restored, (img.shape[1], img.shape[0]))
        except Exception as exc:
            logger.warning("CodeFormer inference error (%s) – fallback", exc)
            return self._run_opencv(img)

    # ── OpenCV CLAHE + bilateral ───────────────────────────────────────────────
    @staticmethod
    def _run_opencv(img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))  # Reduced from 2.0
        l_eq  = clahe.apply(l)
        merged = cv2.merge([l_eq, a, b])
        # Very light sharpening instead of aggressive kernel
        kernel = np.array([[0, -0.2, 0], [-0.2, 1.8, -0.2], [0, -0.2, 0]])
        sharpened = cv2.filter2D(merged, -1, kernel)
        result = cv2.cvtColor(sharpened, cv2.COLOR_Lab2BGR)
        return cv2.bilateralFilter(result, d=5, sigmaColor=50, sigmaSpace=50) # Reduced d and sigmas
