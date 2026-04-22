import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .colorize import ColorizeModule

logger = logging.getLogger("pipeline.color_compare")

SUPPORTED_COLOR_MODELS = (
    "eccv16",
    "deoldify_artistic",
    "deoldify_stable",
    "ddcolor",
)


def normalize_color_model_name(name: str) -> str:
    key = (name or "").strip().lower().replace("-", "_").replace(" ", "")
    aliases = {
        "eccv16": "eccv16",
        "zhang": "eccv16",
        "zhang2016": "eccv16",
        "deoldify": "deoldify_artistic",
        "deoldify_artistic": "deoldify_artistic",
        "deoldify_stable": "deoldify_stable",
        "ddcolor": "ddcolor",
    }
    return aliases.get(key, "")


class ColorCompareModule:
    """
    Colorization comparison runner.

    - eccv16: Zhang et al. 2016 (OpenCV DNN) via existing ColorizeModule
    - deoldify: artistic + stable variants (local weights downloaded on first run)
    - ddcolor: ModelScope pipeline (weights downloaded on first run)
    """

    def __init__(self):
        self._eccv16 = ColorizeModule()
        self._deoldify_artistic = None
        self._deoldify_stable = None
        self._ddcolor_pipe = None

    def process_compare(
        self,
        input_path: str,
        output_dir: str,
        job_id: str,
        model_names: List[str],
    ) -> Dict[str, Dict[str, object]]:
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {input_path}")

        canonical = []
        for m in (model_names or []):
            key = normalize_color_model_name(str(m))
            if key and key not in canonical:
                canonical.append(key)
        if not canonical:
            canonical = ["eccv16"]

        out_base = Path(output_dir)
        outputs: Dict[str, Dict[str, object]] = {}

        for model in canonical:
            t0 = time.perf_counter()
            out_filename = f"{job_id}_colorize_{model}.png"
            out_path = str(out_base / out_filename)
            error_msg = None

            try:
                if model == "eccv16":
                    self._eccv16.process(input_path, out_path)
                    backend = "opencv-dnn"
                elif model == "deoldify_artistic":
                    bgr = self._run_deoldify(input_path, artistic=True)
                    cv2.imwrite(out_path, bgr)
                    backend = "deoldify"
                elif model == "deoldify_stable":
                    bgr = self._run_deoldify(input_path, artistic=False)
                    cv2.imwrite(out_path, bgr)
                    backend = "deoldify"
                elif model == "ddcolor":
                    bgr = self._run_ddcolor(input_path)
                    cv2.imwrite(out_path, bgr)
                    backend = "modelscope-ddcolor"
                else:
                    raise ValueError(f"Unknown colorization model: {model}")
            except Exception as exc:
                error_msg = str(exc)
                logger.warning("Color compare failed for %s (%s) — falling back to eccv16", model, exc)
                self._eccv16.process(input_path, out_path)
                backend = "opencv-dnn-fallback"

            outputs[model] = {
                "filename": out_filename,
                "backend": backend,
                "latency_s": round(time.perf_counter() - t0, 3),
            }
            if error_msg:
                outputs[model]["error"] = error_msg

        return outputs

    def _run_deoldify(self, input_path: str, artistic: bool) -> np.ndarray:
        """
        Returns BGR uint8 image.
        """
        try:
            from deoldify.visualize import get_image_colorizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("DeOldify is not installed. Run: pip install deoldify") from exc

        if artistic:
            if self._deoldify_artistic is None:
                self._deoldify_artistic = get_image_colorizer(artistic=True)
            colorizer = self._deoldify_artistic
            render_factor = 35
        else:
            if self._deoldify_stable is None:
                self._deoldify_stable = get_image_colorizer(artistic=False)
            colorizer = self._deoldify_stable
            render_factor = 25

        pil_img = colorizer.get_transformed_image(
            input_path,
            render_factor=render_factor,
            watermarked=False,
        )
        rgb = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _run_ddcolor(self, input_path: str) -> np.ndarray:
        """
        Returns BGR uint8 image.
        """
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("ModelScope is not installed. Run: pip install modelscope") from exc

        if self._ddcolor_pipe is None:
            self._ddcolor_pipe = pipeline(Tasks.image_colorization, model="damo/cv_ddcolor_image-colorization")

        result = self._ddcolor_pipe(input_path)
        out = result.get("output") if isinstance(result, dict) else result
        if out is None:
            raise RuntimeError("DDColor pipeline returned no output")

        # ModelScope may return PIL or ndarray depending on version.
        if hasattr(out, "convert"):
            rgb = np.array(out.convert("RGB"))
        else:
            arr = np.asarray(out)
            if arr.ndim != 3:
                raise RuntimeError("Unexpected DDColor output shape")
            # Heuristic: assume RGB if last dim is 3.
            rgb = arr[..., :3].astype(np.uint8)

        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

