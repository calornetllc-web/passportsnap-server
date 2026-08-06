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
from PIL import Image, ImageOps
from rembg import remove, new_session

HOST = "0.0.0.0"
# Cloud Run injects the PORT env var automatically (usually 8080)
PORT = int(os.environ.get("PORT", 8080))

# A modern phone photo can be 4000x3000px+. Once PIL decodes that and
# onnxruntime runs it through the model, the in-memory footprint is far
# bigger than the original file size — this is what was pushing the
# container past its 2GiB limit and getting OOM-killed (Cloud Run's
# Errors tab: "Memory limit of 2048 MiB exceeded with 2209 MiB used").
# Passport photos don't need more than this to produce a sharp 2x2in /
# 35x45mm output, so we downscale before the image ever reaches the model.
MAX_DIMENSION = 1600

app = Flask(__name__)
CORS(app, resources={r"/remove-bg": {"origins": [
    "https://passportsnapai.com",
    "https://www.passportsnapai.com",
]}})

# Reject absurdly large uploads before they hit the model (protects RAM on
# a 2GiB Cloud Run instance; a 25MB cap is generous for a phone photo).
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

print("Loading ISNet model (isnet-general-use)...")

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1

SESSION = new_session("isnet-general-use", session_options=opts)
print("ISNet model loaded successfully.")


def downscale_if_needed(input_bytes: bytes) -> bytes:
    """If the uploaded photo's longest side is over MAX_DIMENSION, shrink
    it proportionally before it reaches the model. Keeps the original
    bytes untouched if it's already small enough (most webcam/laptop
    photos won't need this — it's mainly phone camera uploads)."""
    with Image.open(io.BytesIO(input_bytes)) as img:
        # Many phones store portrait photos as landscape pixels + an EXIF
        # orientation tag. Since we re-encode below (dropping EXIF), we
        # need to bake the correct rotation into the actual pixels now —
        # otherwise the output would come out sideways.
        img = ImageOps.exif_transpose(img)
        width, height = img.size
        longest_side = max(width, height)

        if longest_side <= MAX_DIMENSION:
            return input_bytes

        scale = MAX_DIMENSION / longest_side
        new_size = (round(width * scale), round(height * scale))

        img = img.convert("RGB")
        resized = img.resize(new_size, Image.LANCZOS)

        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=92)
        return buffer.getvalue()


@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        input_bytes = request.files["image"].read()
        if not input_bytes:
            return jsonify({"error": "Empty image upload"}), 400

        input_bytes = downscale_if_needed(input_bytes)

        output_bytes = remove(
            input_bytes,
            session=SESSION,
            alpha_matting=False,
        )

        return send_file(io.BytesIO(output_bytes), mimetype="image/png")
    except Exception as exc:
        # Log server-side so Cloud Run logs show what actually failed
        # (OOM, decode error, etc) instead of a bare 500 with no detail.
        print(f"remove-bg failed: {exc!r}")
        return jsonify({"error": "Background removal failed, please try again."}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine": "rembg (isnet-general-use)"})


@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "message": "PassportSnap AI background removal server"})


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
