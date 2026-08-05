"""
PassportSnap AI — Cloud Background Removal Server
=================================================
Uses 'u2netp' (a ~4.7MB, low-RAM model) instead of 'isnet-general-use'
(~176MB) so the app fits comfortably inside Render's free 512 MB
memory limit.
"""

import io
import os
import onnxruntime as ort
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
from rembg import remove, new_session

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5001))

# Cap the largest side of an uploaded image before running inference.
# This bounds peak memory use regardless of what the user uploads.
MAX_DIMENSION = 1500

app = Flask(__name__)
CORS(app)

print("Loading u2netp model with ONNX memory optimization...")

# Keep ONNX Runtime's memory footprint as small as possible
opts = ort.SessionOptions()
opts.enable_cpu_mem_arena = False
opts.enable_mem_pattern = False
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

# u2netp is the lightweight variant of u2net (~4.7 MB vs ~176 MB for
# isnet-general-use). Quality is slightly lower on complex scenes,
# but for passport-style headshots against a plain backdrop it works well.
SESSION = new_session("u2netp", session_options=opts)
print("u2netp model loaded successfully within free RAM limits!")


def _resize_if_needed(image_bytes: bytes) -> bytes:
    """Downscale very large uploads so inference memory stays bounded."""
    img = Image.open(io.BytesIO(image_bytes))
    img.load()

    if max(img.size) > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    return image_bytes


@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        input_bytes = request.files["image"].read()
        input_bytes = _resize_if_needed(input_bytes)

        output_bytes = remove(
            input_bytes,
            session=SESSION,
            alpha_matting=False,
        )

        return send_file(io.BytesIO(output_bytes), mimetype="image/png")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine": "rembg (u2netp) - optimized"})


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
