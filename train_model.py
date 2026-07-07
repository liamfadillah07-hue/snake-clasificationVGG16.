"""
================================================================================
TUGAS 11 - Transfer Learning VGG16
Studi Kasus: Klasifikasi Ular Berbisa vs Tidak Berbisa
================================================================================
Script ini melakukan:
1. Load model VGG16 pretrained (ImageNet) tanpa Fully Connected layer.
2. Membekukan layer convolutional (feature extractor).
3. Menambahkan layer klasifikasi baru (Dense) untuk 2 kelas:
      - venomous      (ular berbisa)
      - non_venomous  (ular tidak berbisa)
4. Fine-tuning beberapa layer terakhir VGG16 (opsional, tahap 2).
5. Menyimpan model akhir ke folder model/ dalam format .h5

Struktur dataset yang diharapkan:
data/
  train/
    venomous/       -> gambar ular berbisa
    non_venomous/   -> gambar ular tidak berbisa
  val/
    venomous/
    non_venomous/
  test/
    venomous/
    non_venomous/

Dataset dapat diunduh dari Kaggle, misalnya:
  - "Snake Species Dataset"
  - "Venomous and Non-Venomous Snakes"
Silakan cari dataset yang sesuai lalu susun ulang ke struktur folder di atas.
================================================================================
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ------------------------------------------------------------------
# 0. Konfigurasi
# ------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FEATURE_EXTRACTION = 15
EPOCHS_FINE_TUNE = 10
CLASS_NAMES = ["non_venomous", "venomous"]   # urutan alfabetis dari flow_from_directory

DATA_DIR = "data"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

print("TensorFlow version:", tf.__version__)

# ------------------------------------------------------------------
# 1. Persiapan Dataset (augmentasi untuk data training)
# ------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_data = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

val_data = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

test_data = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

print("Mapping kelas:", train_data.class_indices)

# ------------------------------------------------------------------
# 2. Load Model VGG16 Pretrained (tanpa Fully Connected Layer)
# ------------------------------------------------------------------
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Tahap 1: bekukan seluruh layer convolutional (feature extraction)
for layer in base_model.layers:
    layer.trainable = False

# ------------------------------------------------------------------
# 3. Tambahkan Layer Klasifikasi Baru
# ------------------------------------------------------------------
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid"),  # klasifikasi biner: berbisa (1) / tidak berbisa (0)
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "snake_vgg16_best.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# ------------------------------------------------------------------
# 4. Tahap 1: Latih hanya classifier baru (feature extraction)
# ------------------------------------------------------------------
history_stage1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FEATURE_EXTRACTION,
    callbacks=[checkpoint, early_stop],
)

# ------------------------------------------------------------------
# 5. Tahap 2: Fine-tuning — buka beberapa blok terakhir VGG16
# ------------------------------------------------------------------
# Buka blok convolutional terakhir (block5) agar dapat menyesuaikan
# fitur dengan pola visual ular berbisa vs tidak berbisa.
for layer in base_model.layers:
    if layer.name.startswith("block5"):
        layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # learning rate lebih kecil saat fine-tuning
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history_stage2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=[checkpoint, early_stop],
)

# ------------------------------------------------------------------
# 6. Evaluasi Model pada Data Uji
# ------------------------------------------------------------------
test_loss, test_acc = model.evaluate(test_data)
print(f"Akurasi Model pada data uji: {test_acc:.2%}")
print(f"Loss Model pada data uji  : {test_loss:.4f}")

# Simpan model final (dipakai oleh app.py)
model.save(os.path.join(MODEL_DIR, "snake_vgg16_final.h5"))
print("Model tersimpan di:", os.path.join(MODEL_DIR, "snake_vgg16_final.h5"))

# ------------------------------------------------------------------
# 7. Visualisasi Hasil Pelatihan (opsional, untuk laporan)
# ------------------------------------------------------------------
def plot_history(h1, h2, out_path):
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Akurasi Training vs Validasi")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Loss Training vs Validasi")

    plt.savefig(out_path)
    print("Grafik pelatihan disimpan di:", out_path)


plot_history(history_stage1, history_stage2, os.path.join(MODEL_DIR, "training_history.png"))
