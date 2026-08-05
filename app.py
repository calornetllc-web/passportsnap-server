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
from rembg import remove, new_session

HOST = "0.0.0.0"
# Cloud Run injects the PORT env var automatically (usually 8080)
PORT = int(os.environ.get("PORT", 8080))

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


@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        input_bytes = request.files["image"].read()
        if not input_bytes:
            return jsonify({"error": "Empty image upload"}), 400

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
