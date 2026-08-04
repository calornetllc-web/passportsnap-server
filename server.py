"""
PassportSnap AI — Cloud Background Removal Server
=================================================
Uses 'isnet-general-use' (ISNet) for high-accuracy edge matting.
"""

import io
import os
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from rembg import remove, new_session

# Listen on 0.0.0.0 and read Render's dynamic port variable
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5001))

app = Flask(__name__)
CORS(app)

print("Loading ISNet matting model (isnet-general-use)...")
SESSION = new_session("isnet-general-use")
print("Model loaded successfully!")


@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    if "image" not in request.files:
        return jsonify({"error": 'No image file provided'}), 400

    try:
        input_bytes = request.files["image"].read()

        # ISNet natively outputs clean soft edges
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
    print("=" * 64)
    print(f"PassportSnap AI — Running on {HOST}:{PORT}")
    print("=" * 64)
    app.run(host=HOST, port=PORT, debug=False)