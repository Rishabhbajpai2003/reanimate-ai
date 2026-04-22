"""
Image quality metrics helpers.

Metrics are computed relative to a reference image (typically the uploaded input).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover - optional dependency guard
    structural_similarity = None


def _read_rgb(path: str) -> np.ndarray:
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image at path: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _align_candidate(reference: np.ndarray, candidate: np.ndarray) -> Tuple[np.ndarray, bool]:
    if reference.shape[:2] == candidate.shape[:2]:
        return candidate, False

    ref_h, ref_w = reference.shape[:2]
    cand_h, cand_w = candidate.shape[:2]
    upscaling = ref_h > cand_h or ref_w > cand_w
    interpolation = cv2.INTER_CUBIC if upscaling else cv2.INTER_AREA
    resized = cv2.resize(candidate, (ref_w, ref_h), interpolation=interpolation)
    return resized, True


@lru_cache(maxsize=1)
def _load_lpips_model():
    import lpips
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    return model, device, torch


def _to_lpips_tensor(image_rgb: np.ndarray, torch_module):
    tensor = torch_module.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor * 2.0 - 1.0


def compute_image_quality_metrics(reference_path: str, candidate_path: str) -> Dict[str, Optional[float]]:
    """
    Compute PSNR, SSIM and LPIPS between two images.

    Returns a dict with numeric values when available and optional error fields
    when a metric cannot be computed.
    """
    out: Dict[str, Optional[float]] = {
        "psnr": None,
        "ssim": None,
        "lpips": None,
        "resized_for_eval": False,
    }

    try:
        reference = _read_rgb(reference_path)
        candidate = _read_rgb(candidate_path)
    except Exception as exc:
        out["error"] = str(exc)
        return out

    candidate, resized = _align_candidate(reference, candidate)
    out["resized_for_eval"] = resized

    # PSNR: higher is better.
    out["psnr"] = round(float(cv2.PSNR(reference, candidate)), 4)

    # SSIM: higher is better, bounded in [-1, 1].
    if structural_similarity is None:
        out["ssim_error"] = "scikit-image not installed"
    else:
        ssim_value = structural_similarity(reference, candidate, channel_axis=2, data_range=255)
        out["ssim"] = round(float(ssim_value), 4)

    # LPIPS: lower is better.
    try:
        model, device, torch_module = _load_lpips_model()
        ref_t = _to_lpips_tensor(reference, torch_module).to(device)
        cand_t = _to_lpips_tensor(candidate, torch_module).to(device)
        with torch_module.no_grad():
            score = model(ref_t, cand_t).item()
        out["lpips"] = round(float(score), 4)
    except Exception as exc:
        out["lpips_error"] = str(exc)

    return out