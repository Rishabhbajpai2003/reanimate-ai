import sys
import time
import logging
from pathlib import Path
from typing import Any, Optional, Dict

# ─── Monkeypatching & Compatibility Fixes ─────────────────────────────────────
# 1. Fix torchvision compatibility for basicsr (0.15+)
import torchvision.transforms.functional as tf_f
sys.modules['torchvision.transforms.functional_tensor'] = tf_f

# 2. Fix torch.load for newer versions (2.0+)
import torch
_original_load = torch.load
def _safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _safe_load

# ─── Pipeline Imports ─────────────────────────────────────────────────────────
from .restore   import RestoreModule
from .super_res import SuperResModule
from .colorize  import ColorizeModule
from .enhance   import EnhanceModule
from .animate   import AnimateModule

logger = logging.getLogger("pipeline.main")


class PipelineController:
    """Singleton that loads models once and runs the full pipeline per job."""

    def __init__(self):
        logger.info("Initialising pipeline modules …")
        t0 = time.perf_counter()
        self.restore_mod   = RestoreModule()
        self.super_res_mod = SuperResModule()
        self.colorize_mod  = ColorizeModule()
        self.enhance_mod   = EnhanceModule()
        self.animate_mod   = AnimateModule()
        logger.info("All modules ready in %.2fs", time.perf_counter() - t0)

    # ──────────────────────────────────────────────────────────────────────────
    def run(
        self,
        input_path: str,
        output_dir: str,
        job_id: str,
        options: Dict[str, bool],
        audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run enabled pipeline stages in order.

        Returns a dict with:
            final_filename   – basename of the final output image
            steps            – ordered list of {"step", "latency_s", "skipped"} dicts
            intermediates    – {step_name: filename} for intermediate outputs
        """
        current_path = input_path
        steps: list[dict] = []
        intermediates: dict[str, str] = {}

        stages = [
            ("restore",   options.get("restore",   True),  self.restore_mod),
            ("colorize",  options.get("colorize",  True),  self.colorize_mod),
            ("super_res", options.get("super_res", True),  self.super_res_mod),
            ("enhance",   options.get("enhance",   True),  self.enhance_mod),
        ]

        for step_name, enabled, module in stages:
            if not enabled:
                steps.append({"step": step_name, "latency_s": 0, "skipped": True})
                logger.info("Skipping %s", step_name)
                continue

            logger.info("Running %s …", step_name)
            t0 = time.perf_counter()
            try:
                out_filename = f"{job_id}_{step_name}.png"
                out_path     = str(Path(output_dir) / out_filename)
                current_path = module.process(current_path, out_path)
                latency      = round(time.perf_counter() - t0, 3)
                steps.append({"step": step_name, "latency_s": latency, "skipped": False})
                intermediates[step_name] = out_filename
                logger.info("  ✓ %s done in %.3fs → %s", step_name, latency, out_path)
            except Exception as exc:
                latency = round(time.perf_counter() - t0, 3)
                logger.warning("  ✗ %s FAILED (%.3fs): %s — using previous output", step_name, latency, exc)
                steps.append({"step": step_name, "latency_s": latency, "skipped": False, "error": str(exc)})

        # ── Animation (optional, produces GIF/MP4) ────────────────────────────
        animation_filename = None
        if options.get("animate", False):
            t0 = time.perf_counter()
            try:
                anim_base = f"{job_id}_animate.gif"
                anim_path = str(Path(output_dir) / anim_base)
                result_path    = self.animate_mod.process(current_path, anim_path, audio_path=audio_path)
                # The module may change the extension (.gif / .mp4)
                animation_filename = Path(result_path).name
                latency = round(time.perf_counter() - t0, 3)
                steps.append({"step": "animate", "latency_s": latency, "skipped": False})
                intermediates["animate"] = animation_filename
                logger.info("  ✓ animate done in %.3fs → %s", latency, result_path)
            except Exception as exc:
                latency = round(time.perf_counter() - t0, 3)
                logger.warning("  ✗ animate FAILED: %s", exc)
                steps.append({"step": "animate", "latency_s": latency, "skipped": False, "error": str(exc)})
        else:
            steps.append({"step": "animate", "latency_s": 0, "skipped": True})

        # ── Save final image ──────────────────────────────────────────────────
        import shutil
        final_filename = f"{job_id}_final.png"
        final_path     = str(Path(output_dir) / final_filename)
        if current_path != final_path:
            shutil.copy2(current_path, final_path)

        total = sum(s["latency_s"] for s in steps)
        logger.info("Pipeline complete – total %.3fs", total)

        return {
            "final_filename":     final_filename,
            "animation_filename": animation_filename,
            "steps":              steps,
            "intermediates":      intermediates,
        }
