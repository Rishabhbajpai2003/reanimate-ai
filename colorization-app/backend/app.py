"""
ReAnimateAI - Flask Backend
Production-ready portrait restoration API
"""

import os
import time
import uuid
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from pipeline.main import PipelineController

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("reanimateai")

# ─── App Config ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULTS_DIR = BASE_DIR / "static" / "results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}
MAX_CONTENT_MB = 20

for d in [UPLOAD_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(BASE_DIR / "static"))
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ─── Singleton Pipeline ────────────────────────────────────────────────────────
pipeline = PipelineController()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def build_result_url(filename: str) -> str:
    return url_for("serve_static", subfolder="results", filename=filename, _external=True)


def build_upload_url(filename: str) -> str:
    return url_for("serve_static", subfolder="uploads", filename=filename, _external=True)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Service health check."""
    return jsonify({"status": "ok", "service": "ReAnimateAI", "version": "1.0.0"})


@app.route("/api/upload", methods=["POST"])
def upload():
    """
    Upload image and run the restoration pipeline.

    Form fields:
        image (file)  – portrait image
        restore   (bool-string) – enable restoration
        super_res (bool-string) – enable super resolution
        colorize  (bool-string) – enable colorization
        enhance   (bool-string) – enable face enhancement
        animate   (bool-string) – enable animation (requires audio)
        audio     (file, opt)  – audio for animation
    """
    start = time.perf_counter()

    # ── Validate image ─────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not allowed_file(file.filename):
        return jsonify(
            {"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}
        ), 415

    # ── Parse toggle flags ─────────────────────────────────────────────────
    def flag(name: str, default: bool = True) -> bool:
        val = request.form.get(name, str(default)).lower()
        return val in ("true", "1", "yes", "on")

    options = {
        "restore":   flag("restore",   True),
        "super_res": flag("super_res", True),
        "colorize":  flag("colorize",  True),
        "enhance":   flag("enhance",   True),
        "animate":   flag("animate",   False),
    }

    # ── Save uploaded image ────────────────────────────────────────────────
    job_id = uuid.uuid4().hex
    ext = secure_filename(file.filename).rsplit(".", 1)[-1].lower()
    input_filename = f"{job_id}_input.{ext}"
    input_path = UPLOAD_DIR / input_filename
    file.save(str(input_path))
    logger.info("Saved upload → %s", input_path)

    # ── Optional audio for animation ───────────────────────────────────────
    audio_path = None
    if options["animate"] and "audio" in request.files:
        audio_file = request.files["audio"]
        if audio_file and audio_file.filename:
            audio_filename = f"{job_id}_audio.{audio_file.filename.rsplit('.',1)[-1]}"
            audio_path = str(UPLOAD_DIR / audio_filename)
            audio_file.save(audio_path)

    # ── Run pipeline ───────────────────────────────────────────────────────
    try:
        result = pipeline.run(
            input_path=str(input_path),
            output_dir=str(RESULTS_DIR),
            job_id=job_id,
            options=options,
            audio_path=audio_path,
        )
    except Exception as exc:
        logger.exception("Pipeline failed for job %s", job_id)
        return jsonify({"error": f"Pipeline error: {str(exc)}"}), 500

    elapsed = round(time.perf_counter() - start, 3)
    logger.info("Job %s completed in %.3fs", job_id, elapsed)

    # ── Build response ─────────────────────────────────────────────────────
    response = {
        "job_id": job_id,
        "elapsed_seconds": elapsed,
        "original_url": build_upload_url(input_filename),
        "result_url": build_result_url(result["final_filename"]),
        "animation_url": (
            build_result_url(result["animation_filename"])
            if result.get("animation_filename")
            else None
        ),
        "steps": result["steps"],
        "intermediates": {
            step: build_result_url(fname)
            for step, fname in result.get("intermediates", {}).items()
        },
    }
    return jsonify(response), 200


@app.route("/api/result/<job_id>", methods=["GET"])
def get_result(job_id: str):
    """Retrieve metadata for a previously processed job."""
    matches = list(RESULTS_DIR.glob(f"{job_id}_final.*"))
    if not matches:
        return jsonify({"error": "Result not found."}), 404
    filename = matches[0].name
    return jsonify({"job_id": job_id, "result_url": build_result_url(filename)}), 200


@app.route("/static/<subfolder>/<filename>")
def serve_static(subfolder: str, filename: str):
    """Serve uploaded / result images."""
    directory = str(BASE_DIR / "static" / subfolder)
    return send_from_directory(directory, filename)


# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify({"error": f"File too large. Max size is {MAX_CONTENT_MB} MB."}), 413


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(500)
def server_error(exc):
    logger.exception("Unhandled server error")
    return jsonify({"error": "Internal server error."}), 500


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    # 🔥 FORCE production mode (no reload, no double imports)
    debug = False

    logger.info("Starting ReAnimateAI on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
