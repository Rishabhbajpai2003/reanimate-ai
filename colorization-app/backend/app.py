"""
Backend server for the image colorization demo.

Dependencies are managed via requirements.txt. Avoid installing packages at runtime.
"""

# -------- Imports -------- #
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import time
import io
import traceback

# Optional AI stylization (local ONNX model)
try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None

app = Flask(__name__)
CORS(app)

print("[backend] Loading model...")

# -------- Paths -------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
proto = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
model = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
points = os.path.join(MODEL_DIR, "pts_in_hull.npy")

# AI stylization model (optional)
AI_MODEL_PATH = os.path.join(MODEL_DIR, "animeganv2.onnx")
AI_READY = False
AI_ERROR = None
AI_SESSION = None

def _try_load_ai_model():
    global AI_READY, AI_ERROR, AI_SESSION
    if ort is None:
        AI_READY = False
        AI_ERROR = "onnxruntime not installed"
        return
    if not os.path.exists(AI_MODEL_PATH):
        AI_READY = False
        AI_ERROR = f"Missing AI model at {AI_MODEL_PATH}"
        return
    try:
        # CPU execution provider for broad compatibility
        AI_SESSION = ort.InferenceSession(AI_MODEL_PATH, providers=["CPUExecutionProvider"])
        AI_READY = True
        AI_ERROR = None
    except Exception as e:
        AI_READY = False
        AI_ERROR = str(e)

_try_load_ai_model()

MODEL_READY = False
MODEL_ERROR = None
net = None

try:
    # -------- Check files -------- #
    if not os.path.exists(proto) or not os.path.exists(model) or not os.path.exists(points):
        raise FileNotFoundError(
            "Model files missing. Place these in `colorization-app/backend/model/`: "
            "`colorization_deploy_v2.prototxt`, `colorization_release_v2.caffemodel`, `pts_in_hull.npy`."
        )

    # -------- Load model -------- #
    net = cv2.dnn.readNetFromCaffe(proto, model)
    pts = np.load(points)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")

    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    print("[backend] Warming up model...")
    dummy = np.zeros((224, 224), dtype="float32")
    net.setInput(cv2.dnn.blobFromImage(dummy))
    net.forward()

    MODEL_READY = True
    print("[backend] Model loaded & ready.")
except Exception as e:
    MODEL_ERROR = str(e)
    print("[backend] ERROR:", MODEL_ERROR)

NET_IN = 224  # matches colorization_deploy_v2.prototxt input shape


def _clamp_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    s = max_side / m
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _parse_float(val, default: float, lo: float, hi: float) -> float:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


def _apply_clahe_lab_bgr(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def _boost_saturation_bgr(img_bgr: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1.0:
        return img_bgr
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _unsharp_bgr(img_bgr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img_bgr
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=1.0)
    out = cv2.addWeighted(img_bgr, 1.0 + amount, blur, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _parse_int(val, default: int, lo: int, hi: int) -> int:
    try:
        x = int(val)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


def _kmeans_quantize_bgr(img_bgr: np.ndarray, k: int) -> np.ndarray:
    """Color quantization to flatten tones (cartoon/anime look)."""
    h, w = img_bgr.shape[:2]
    data = img_bgr.reshape((-1, 3)).astype(np.float32)
    # kmeans can be heavy; keep iterations small.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 18, 1.0)
    _ret, labels, centers = cv2.kmeans(
        data, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS
    )
    centers = centers.astype(np.uint8)
    out = centers[labels.flatten()].reshape((h, w, 3))
    return out


def _xdog_edges(gray: np.ndarray, sigma: float, k: float, gamma: float, eps: float, phi: float) -> np.ndarray:
    """
    XDoG edge map (gives cleaner ink-like lines than Canny on many portraits).
    Returns uint8 edges where 0=black line, 255=background.
    """
    g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma * k)
    dog = g1 - gamma * g2
    xdog = 1.0 + np.tanh(phi * (dog.astype(np.float32) - eps))
    # xdog in (0..2); threshold to ink
    ink = (xdog < 1.0).astype(np.uint8) * 255  # 255 where line
    ink = cv2.GaussianBlur(ink, (0, 0), sigmaX=0.6)
    # Convert to mask where 255 = keep color, 0 = draw black line
    return cv2.bitwise_not(ink)


def _cartoonize_bgr(
    img_bgr: np.ndarray,
    mode: str = "anime",
    strength: float = 0.75,
    preset: str | None = None,
) -> np.ndarray:
    """
    Improved OpenCV stylization (no extra models).
    Presets (recommended):
      - preset=clean_anime
      - preset=soft_cartoon
      - preset=ink
    You can also use mode=anime|cartoon|sketch for backwards compatibility.
    """
    if preset:
        preset = preset.lower().strip()
    mode = (mode or "anime").lower()
    s = float(max(0.0, min(1.0, strength)))

    if preset in ("clean_anime", "anime_clean", "anime"):
        mode = "anime"
    elif preset in ("soft_cartoon", "cartoon_soft", "cartoon"):
        mode = "cartoon"
    elif preset in ("ink", "manga", "lineart"):
        mode = "ink"
    elif preset in ("sketch",):
        mode = "sketch"

    img = img_bgr
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if mode == "sketch":
        # Pencil sketch (more realistic than plain edges):
        # gray -> invert -> blur -> dodge blend (divide)
        g = gray
        inv = 255 - g
        sigma = 10 + 18 * s
        blur = cv2.GaussianBlur(inv, (0, 0), sigmaX=sigma)
        sketch = cv2.divide(g, 255 - blur, scale=256)

        # Add subtle paper-like contrast control
        sketch = cv2.GaussianBlur(sketch, (0, 0), sigmaX=0.6)
        sketch = cv2.normalize(sketch, None, 0, 255, cv2.NORM_MINMAX)

        # Optional light ink outline to keep facial features readable
        outline = _xdog_edges(g, sigma=0.70, k=1.6, gamma=0.98, eps=0.0, phi=12.0)
        out = cv2.min(sketch, outline)
        return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # 1) Smooth while preserving edges
    if mode == "cartoon":
        d = int(7 + 10 * s)
        sigma_c = 60 + 110 * s
        sigma_s = 60 + 110 * s
        smooth = cv2.bilateralFilter(img, d, sigma_c, sigma_s)
        smooth = cv2.bilateralFilter(smooth, d, sigma_c, sigma_s)
    else:
        # anime/ink: slightly sharper smoothing
        sigma_s = int(45 + 70 * s)
        sigma_r = float(0.20 + 0.28 * s)
        smooth = cv2.edgePreservingFilter(img, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)

    # 2) Flatten colors (quantize)
    if mode == "cartoon":
        k = int(12 + 10 * (1.0 - s))  # 12..22
    else:
        k = int(10 + 8 * (1.0 - s))   # 10..18
    k = int(max(6, min(28, k)))
    flat = _kmeans_quantize_bgr(smooth, k=k)

    # 3) Better edges (XDoG)
    if mode == "cartoon":
        edge_mask = _xdog_edges(gray, sigma=0.85, k=1.6, gamma=0.98, eps=0.0, phi=10.0)
        thick = max(1, int(1 + 2 * s))
        if thick > 1:
            edge_mask = cv2.erode(edge_mask, np.ones((thick, thick), np.uint8), iterations=1)
    else:
        edge_mask = _xdog_edges(gray, sigma=0.65, k=1.6, gamma=0.98, eps=0.0, phi=12.0)

    # 4) Composite: draw black lines over flat colors
    out = cv2.bitwise_and(flat, cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR))

    if mode == "ink":
        # Stronger ink look: reduce shading
        out = _kmeans_quantize_bgr(out, k=6)
        out = cv2.addWeighted(out, 0.92, cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR), 0.08, 0)

    # Small color pop + micro-contrast helps the "anime" look on many photos.
    if mode in ("anime", "cartoon"):
        # Scale with strength but keep safe.
        sat = 1.05 + 0.18 * s
        out = _boost_saturation_bgr(out, sat)
        out = _unsharp_bgr(out, amount=0.06 + 0.10 * s)

    return out


# -------- API -------- #
@app.route("/colorize", methods=["POST"])
def colorize():
    try:
        if not MODEL_READY or net is None:
            return jsonify({"error": MODEL_ERROR or "Model is not ready."}), 500

        print("\n[backend] /colorize request received")

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        input_path = os.path.join(BASE_DIR, "input.jpg")
        output_path = os.path.join(BASE_DIR, "output.jpg")

        file.save(input_path)
        print("[backend] Image saved")

        img = cv2.imread(input_path)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        quality = (request.form.get("quality") or "balanced").lower()
        if quality not in ("fast", "balanced", "best"):
            quality = "balanced"

        grayscale_only = request.form.get("grayscale_only", "0") == "1"
        clahe_on = request.form.get("clahe", "0") == "1"
        sat = _parse_float(request.form.get("saturation"), 1.08, 0.8, 1.45)
        if quality == "fast":
            max_side, square, sharp_default = 896, 256, 0.0
        elif quality == "best":
            max_side, square, sharp_default = 1600, 512, 0.25
        else:
            max_side, square, sharp_default = 1200, 384, 0.12
        raw_sharp = request.form.get("sharpness")
        if raw_sharp is None or raw_sharp == "":
            sharp = sharp_default
        else:
            sharp = _parse_float(raw_sharp, sharp_default, 0.0, 0.6)

        img = _clamp_side(img, max_side)

        if grayscale_only:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if clahe_on:
            img = _apply_clahe_lab_bgr(img)

        orig_h, orig_w = img.shape[:2]

        img_sq = cv2.resize(img, (square, square), interpolation=cv2.INTER_AREA)

        scaled = img_sq.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        lab_net = cv2.resize(lab, (NET_IN, NET_IN), interpolation=cv2.INTER_AREA)
        L = cv2.split(lab_net)[0].astype("float32")
        L -= 50

        print("[backend] Running model...")

        start = time.time()

        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0].transpose((1, 2, 0))

        end = time.time()
        print(f"[backend] Inference time: {end - start:.2f}s")

        print("[backend] Model inference done")

        ab = cv2.resize(ab, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        lab_original = cv2.cvtColor(img.astype("float32") / 255.0, cv2.COLOR_BGR2LAB)
        L_original = cv2.split(lab_original)[0]

        colorized = np.concatenate((L_original[:, :, np.newaxis], ab), axis=2)

        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")

        colorized = _boost_saturation_bgr(colorized, sat)
        colorized = _unsharp_bgr(colorized, sharp)

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        cv2.imwrite(output_path, colorized, encode_params)

        print("[backend] Output ready")

        return send_file(output_path, mimetype="image/jpeg")

    except Exception as e:
        print("[backend] ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/cartoonize", methods=["POST"])
def cartoonize():
    """
    Stylize a photo into cartoon/anime/sketch look.

    Multipart:
      - image: input photo (required)
    Form fields:
      - mode: anime|cartoon|sketch (default: anime)
      - strength: 0..1 (default: 0.75)
      - max_side: 384..1600 (default: 1200)
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        preset = (request.form.get("preset") or "").strip().lower() or None
        mode = (request.form.get("mode") or "anime").lower()
        if mode not in ("anime", "cartoon", "sketch", "ink"):
            mode = "anime"
        strength = _parse_float(request.form.get("strength"), 0.75, 0.0, 1.0)
        max_side = _parse_int(request.form.get("max_side"), 1200, 384, 1600)

        file = request.files["image"]
        input_path = os.path.join(BASE_DIR, "cartoon_input.jpg")
        file.save(input_path)
        img = cv2.imread(input_path)
        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        img = _clamp_side(img, max_side)
        out = _cartoonize_bgr(img, mode=mode, strength=strength, preset=preset)

        ok, buf = cv2.imencode(".png", out)
        if not ok:
            return jsonify({"error": "Failed to encode output"}), 500
        name = (preset or mode) if (preset or mode) else "stylized"
        return send_file(io.BytesIO(buf.tobytes()), mimetype="image/png", download_name=f"{name}.png")

    except Exception as e:
        tb = traceback.format_exc()
        print("[backend] CARTOONIZE ERROR:", str(e))
        print(tb)
        return jsonify({"error": str(e), "traceback": tb}), 500


def _ai_preprocess_bgr(img_bgr: np.ndarray, size: int, layout: str):
    """
    AnimeGAN-style preprocessing:
      - resize to size x size
      - BGR -> RGB
      - normalize to [-1, 1]
      - NCHW float32
    Returns (input_tensor, original_hw, resized_rgb).
    """
    h, w = img_bgr.shape[:2]
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = (rgb / 127.5) - 1.0
    if layout == "NHWC":
        x = x[None, :, :, :]  # 1xHxWx3
    else:
        x = np.transpose(x, (2, 0, 1))[None, :, :, :]  # 1x3xHxW
    return x.astype(np.float32), (h, w)


def _ai_postprocess_rgb(out: np.ndarray, out_hw) -> np.ndarray:
    """
    Postprocess from model output to uint8 BGR, resized to original hw.
    Expected model output: 1x3xHxW or 1xHxWx3 in [-1,1] or [0,1].
    """
    y = out
    y = np.squeeze(y)
    if y.ndim == 3 and y.shape[0] == 3:
        y = np.transpose(y, (1, 2, 0))
    y = y.astype(np.float32)
    # map to 0..255
    if y.min() < 0:
        y = (y + 1.0) * 127.5
    else:
        y = y * 255.0
    y = np.clip(y, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
    h, w = out_hw
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_CUBIC)


@app.route("/stylize_ai", methods=["POST"])
def stylize_ai():
    """
    AI cartoon/anime stylization using a local ONNX model.

    Multipart:
      - image: input photo (required)
    Form fields:
      - max_side: 384..1600 (default: 1200)
    """
    try:
        if not AI_READY or AI_SESSION is None:
            return jsonify(
                {
                    "error": "AI stylizer not ready.",
                    "details": AI_ERROR,
                    "hint": "Run the model downloader script to place `animeganv2.onnx` into `colorization-app/backend/model/` and restart the backend.",
                }
            ), 500

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        max_side = _parse_int(request.form.get("max_side"), 1200, 384, 1600)
        file = request.files["image"]
        input_path = os.path.join(BASE_DIR, "ai_input.jpg")
        file.save(input_path)
        img = cv2.imread(input_path)
        if img is None:
            return jsonify({"error": "Invalid image"}), 400
        img = _clamp_side(img, max_side)

        inp = AI_SESSION.get_inputs()[0]
        inp_name = inp.name
        shp = inp.shape or []
        # Infer expected layout. Many AnimeGAN exports use NHWC (1,512,512,3).
        layout = "NCHW"
        if len(shp) == 4 and shp[-1] == 3:
            layout = "NHWC"
        size = 512
        # Try to infer size if the model declares it
        if len(shp) == 4:
            if layout == "NHWC" and isinstance(shp[1], int):
                size = int(shp[1])
            elif layout == "NCHW" and isinstance(shp[2], int):
                size = int(shp[2])

        x, hw = _ai_preprocess_bgr(img, size=size, layout=layout)
        outs = AI_SESSION.run(None, {inp_name: x})
        y = outs[0]
        out_bgr = _ai_postprocess_rgb(y, hw)

        ok, buf = cv2.imencode(".png", out_bgr)
        if not ok:
            return jsonify({"error": "Failed to encode output"}), 500
        return send_file(io.BytesIO(buf.tobytes()), mimetype="image/png", download_name="ai-cartoon.png")

    except Exception as e:
        tb = traceback.format_exc()
        print("[backend] STYLIZE_AI ERROR:", str(e))
        print(tb)
        return jsonify({"error": str(e), "traceback": tb}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "ok": True,
            "backend_version": "core-v1",
            "colorization_model_ready": bool(MODEL_READY),
            "colorization_model_error": MODEL_ERROR,
            "ai_stylizer_ready": bool(AI_READY),
            "ai_stylizer_error": AI_ERROR,
        }
    )


@app.route("/routes", methods=["GET"])
def routes():
    rules = sorted({r.rule for r in app.url_map.iter_rules()})
    return jsonify({"routes": rules})


# -------- Run -------- #
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
