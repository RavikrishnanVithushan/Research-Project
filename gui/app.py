"""
ASL Gesture Recognition — Flask Backend
Uses the trained ResNet-style model (100% accuracy) for prediction.

Usage:
    python app.py

Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import io

app = Flask(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "resnet_asl_model.h5")
model = None

def load_model():
    global model
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[OK] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
        print("[INFO] Running in DEMO mode — predictions are simulated")

# ASL letters (J and Z excluded)
ASL_LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(image_bytes):
    """Convert uploaded image bytes to 28x28 grayscale tensor."""
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("L")           # grayscale
    img = img.resize((28, 28))       # resize to 28x28
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 28, 28, 1)  # (1, 28, 28, 1)
    return arr

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    try:
        img_array = preprocess_image(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 400

    # ── Predict ──────────────────────────────────────────────────────────────
    if model is not None:
        predictions = model.predict(img_array, verbose=0)[0]  # shape (24,)
    else:
        # Demo mode — simulate a prediction when no model is loaded
        np.random.seed(int.from_bytes(image_bytes[:4], "little") % 2**31)
        raw = np.random.dirichlet(np.ones(24) * 0.5)
        # Amplify one class to simulate a confident prediction
        top_idx = np.random.randint(24)
        raw[top_idx] += 2.0
        predictions = raw / raw.sum()

    top3_idx  = np.argsort(predictions)[::-1][:3]
    top_class = int(top3_idx[0])
    confidence = float(predictions[top_class])
    letter = ASL_LETTERS[top_class]

    top3 = [
        {"letter": ASL_LETTERS[int(i)], "confidence": float(predictions[i])}
        for i in top3_idx
    ]

    return jsonify({
        "letter":     letter,
        "confidence": confidence,
        "class_idx":  top_class,
        "top3":       top3,
        "demo_mode":  model is None,
    })

@app.route("/health")
def health():
    return jsonify({
        "status":     "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    })

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)