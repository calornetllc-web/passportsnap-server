"""
PassportSnap AI — Cloud Background Removal Server (Google Cloud Run)
======================================================================
Runs on Cloud Run's free tier. We give the container 2GiB RAM (well
within the free monthly quota for low/medium traffic), so we can use
the full-quality 'isnet-general-use' model without memory tricks.
"""

import io
import os
import onnxruntime as ort
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image, ImageFilter
from rembg import remove, new_session

HOST = "0.0.0.0"
# Cloud Run injects the PORT env var automatically (usually 8080)
PORT = int(os.environ.get("PORT", 8080))

app = Flask(__name__)
CORS(app)

print("Loading ISNet model (isnet-general-use)...")

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1

SESSION = new_session("isnet-general-use", session_options=opts)
print("ISNet model loaded successfully.")


def _clean_edge_spill(png_bytes: bytes) -> bytes:
    """
    Alpha matting leaves a thin ring of partially-transparent pixels at
    hair edges. Their stored color is a blend of hair + the ORIGINAL
    photo's background, so when composited onto a NEW background color
    later, that old-background color bleeds through as a light halo.

    We fix this by slightly shrinking (eroding) the alpha mask so those
    worst-contaminated edge pixels get dropped instead of kept semi-visible.
    This trades a hair or two of fine detail for a clean, halo-free edge.
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    r, g, b, a = img.split()

    # Shrink the opaque region by ~1-2px. MinFilter takes the darkest
    # (=most transparent) value in each neighborhood, eating into the
    # semi-transparent spill ring from the outside in.
    a = a.filter(ImageFilter.MinFilter(3))
    a = a.filter(ImageFilter.MinFilter(3))

    # Slight blur + re-sharpen keeps the remaining edge smooth rather than jagged
    a = a.filter(ImageFilter.GaussianBlur(radius=0.6))

    cleaned = Image.merge("RGBA", (r, g, b, a))
    out = io.BytesIO()
    cleaned.save(out, format="PNG")
    return out.getvalue()


@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        input_bytes = request.files["image"].read()

        output_bytes = remove(
            input_bytes,
            session=SESSION,
            alpha_matting=True,
            alpha_matting_foreground_threshold=250,
            alpha_matting_background_threshold=20,
            alpha_matting_erode_size=12,
        )

        output_bytes = _clean_edge_spill(output_bytes)

        return send_file(io.BytesIO(output_bytes), mimetype="image/png")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine": "rembg (isnet-general-use)"})


@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "message": "PassportSnap AI background removal server"})


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
