"""
PassportSnap AI — Cloud Background Removal Server
Uses 'isnet-general-use' with ONNX RAM optimization.
"""

import io
import os
import onnxruntime as ort
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from rembg import remove, new_session

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5001))

app = Flask(__name__)
CORS(app)

print("Loading ISNet model (isnet-general-use)...")

# Keep RAM footprint under 230 MB
opts = ort.SessionOptions()
opts.enable_cpu_mem_arena = False
opts.intra_op_num_threads = 1

SESSION = new_session("isnet-general-use", session_options=opts)
print("ISNet Model loaded successfully!")


@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    if "image" not in request.files:
        return jsonify({"error": 'No image file provided'}), 400

    try:
        input_bytes = request.files["image"].read()

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
    return jsonify({"status": "ok", "engine": "rembg (isnet-general-use)"})


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
