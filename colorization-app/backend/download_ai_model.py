"""
Download an anime/cartoon ONNX model for local AI stylization.

This keeps the main app lightweight: you only download the model once, then you can run offline.
"""

import os
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# One known source of AnimeGANv2 ONNX models:
# Repo: TachibanaYoshino/AnimeGANv2 (pb_and_onnx_model)
# If GitHub blocks direct downloads, use the HuggingFace mirror in the list.
URLS = [
    # HuggingFace mirror (usually most reliable)
    "https://huggingface.co/akhaliq/AnimeGANv2-ONNX/resolve/main/Shinkai_53.onnx",
    # GitHub raw (may rate-limit)
    "https://raw.githubusercontent.com/TachibanaYoshino/AnimeGANv2/master/pb_and_onnx_model/Shinkai_53.onnx",
]

OUT_PATH = os.path.join(MODEL_DIR, "animeganv2.onnx")


def main():
    if os.path.exists(OUT_PATH) and os.path.getsize(OUT_PATH) > 5 * 1024 * 1024:
        print("Model already exists:", OUT_PATH)
        return

    last_err = None
    for url in URLS:
        try:
            print("Downloading:", url)
            urllib.request.urlretrieve(url, OUT_PATH)
            if os.path.exists(OUT_PATH) and os.path.getsize(OUT_PATH) > 5 * 1024 * 1024:
                print("Saved:", OUT_PATH)
                return
            else:
                last_err = RuntimeError("Downloaded file is too small; retrying.")
        except Exception as e:
            last_err = e
            print("Failed:", str(e))

    raise SystemExit(f"Could not download model. Last error: {last_err}")


if __name__ == "__main__":
    main()

