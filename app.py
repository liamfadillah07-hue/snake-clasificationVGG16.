"""
================================================================================
TUGAS 11 - Aplikasi Web Flask
Klasifikasi Ular Berbisa vs Tidak Berbisa (Transfer Learning VGG16)
================================================================================
Cara menjalankan secara lokal:
    pip install -r requirements.txt
    python app.py
Lalu buka http://127.0.0.1:5000 di browser.

CATATAN: Sebelum menjalankan app.py, pastikan model sudah dilatih dengan
train_model.py sehingga file model/snake_vgg16_final.h5 tersedia.
================================================================================
"""

import os
import uuid

import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# ------------------------------------------------------------------
# Konfigurasi Aplikasi
# ------------------------------------------------------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "model", "snake_vgg16_final.h5")
UPLOAD_FOLDER = os.path.join(APP_ROOT, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Tidak Berbisa", "Berbisa"]  # index 0 -> non_venomous, index 1 -> venomous

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # maksimal 8 MB
app.secret_key = "ganti-dengan-secret-key-anda"

# ------------------------------------------------------------------
# Load Model (sekali saat aplikasi start)
# ------------------------------------------------------------------
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Model berhasil dimuat dari:", MODEL_PATH)
else:
    print("PERINGATAN: model belum ditemukan di", MODEL_PATH)
    print("Jalankan train_model.py terlebih dahulu untuk menghasilkan model.")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(filepath: str):
    """Preprocessing gambar lalu prediksi menggunakan model VGG16 yang telah di-fine-tune."""
    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0][0]  # nilai sigmoid 0..1
    label_index = 1 if prob >= 0.5 else 0
    confidence = prob if label_index == 1 else 1 - prob

    return {
        "label": CLASS_NAMES[label_index],
        "confidence": round(float(confidence) * 100, 2),
        "raw_score": float(prob),
    }


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        flash("Model belum tersedia. Silakan latih model terlebih dahulu.")
        return redirect(url_for("index"))

    if "file" not in request.files:
        flash("Tidak ada file yang diunggah.")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("Silakan pilih gambar terlebih dahulu.")
        return redirect(url_for("index"))

    if file and allowed_file(file.filename):
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        filename = secure_filename(unique_name)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        result = predict_image(filepath)

        return render_template(
            "result.html",
            image_path=url_for("static", filename=f"uploads/{filename}"),
            label=result["label"],
            confidence=result["confidence"],
        )

    flash("Format file tidak didukung. Gunakan PNG, JPG, atau JPEG.")
    return redirect(url_for("index"))


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    # debug=True hanya untuk pengembangan lokal, matikan saat production
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
