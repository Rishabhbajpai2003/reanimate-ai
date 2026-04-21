"""
Pipeline Controller — orchestrates all restoration modules sequentially.
"""

import time
import logging
from pathlib import Path
from typing import Any, Optional, Dict, List

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
        options: Dict[str, Any],
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
        sr_compare_outputs: dict[str, dict] = {}

        sr_compare = bool(options.get("sr_compare", False) and options.get("super_res", True))
        sr_models: List[str] = options.get("sr_models") or ["realesrgan"]

        stages = [
            ("restore",   options.get("restore",   True),  self.restore_mod),
            ("colorize",  options.get("colorize",  True),  self.colorize_mod),
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

        # ── Super resolution (single or compare mode) ───────────────────────
        if not options.get("super_res", True):
            steps.append({"step": "super_res", "latency_s": 0, "skipped": True})
            logger.info("Skipping super_res")
        elif sr_compare:
            logger.info("Running super_res compare for models: %s", ", ".join(sr_models))
            t0 = time.perf_counter()
            try:
                sr_outputs = self.super_res_mod.process_compare(
                    input_path=current_path,
                    output_dir=output_dir,
                    job_id=job_id,
                    model_names=sr_models,
                )
                compare_latency = round(time.perf_counter() - t0, 3)

                default_model = "realesrgan" if "realesrgan" in sr_outputs else next(iter(sr_outputs))
                default_filename = sr_outputs[default_model]["filename"]
                current_path = str(Path(output_dir) / default_filename)

                for model_name, info in sr_outputs.items():
                    step_name = f"super_res:{model_name}"
                    steps.append({
                        "step": step_name,
                        "latency_s": info.get("latency_s", compare_latency),
                        "skipped": False,
                        "backend": info.get("backend", "unknown"),
                    })
                    inter_key = f"super_res_{model_name}"
                    intermediates[inter_key] = info["filename"]
                    sr_compare_outputs[model_name] = {
                        "filename": info["filename"],
                        "backend": info.get("backend", "unknown"),
                        "latency_s": info.get("latency_s", 0),
                    }
                    if "error" in info:
                        sr_compare_outputs[model_name]["error"] = info["error"]

                logger.info("  ✓ super_res compare done in %.3fs", compare_latency)
            except Exception as exc:
                compare_latency = round(time.perf_counter() - t0, 3)
                logger.warning("  ✗ super_res compare FAILED (%.3fs): %s", compare_latency, exc)
                steps.append({"step": "super_res", "latency_s": compare_latency, "skipped": False, "error": str(exc)})
        else:
            logger.info("Running super_res …")
            t0 = time.perf_counter()
            try:
                out_filename = f"{job_id}_super_res.png"
                out_path = str(Path(output_dir) / out_filename)
                sr_model = (sr_models[0] if sr_models else "realesrgan")
                sr_result = self.super_res_mod.process(current_path, out_path, model_name=sr_model)
                current_path = sr_result["output_path"]
                latency = round(time.perf_counter() - t0, 3)
                steps.append({
                    "step": "super_res",
                    "latency_s": latency,
                    "skipped": False,
                    "model": sr_result.get("model", sr_model),
                    "backend": sr_result.get("backend", "unknown"),
                })
                intermediates["super_res"] = out_filename
                logger.info("  ✓ super_res done in %.3fs → %s", latency, out_path)
            except Exception as exc:
                latency = round(time.perf_counter() - t0, 3)
                logger.warning("  ✗ super_res FAILED (%.3fs): %s — using previous output", latency, exc)
                steps.append({"step": "super_res", "latency_s": latency, "skipped": False, "error": str(exc)})

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
            "sr_compare_outputs": sr_compare_outputs,
        }
