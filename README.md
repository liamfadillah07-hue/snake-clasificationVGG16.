# SnakeSense AI — Klasifikasi Ular Berbisa vs Tidak Berbisa
### Tugas 11 — Transfer Learning VGG16 (Flask Web App)

Aplikasi web berbasis Flask yang menggunakan **transfer learning VGG16** untuk
mengklasifikasikan gambar ular menjadi **Berbisa (Venomous)** atau
**Tidak Berbisa (Non-Venomous)**.

---

## 1. Struktur Proyek

```
snake-classifier/
├── app.py                  # Aplikasi Flask (web server + prediksi)
├── train_model.py          # Script training model VGG16
├── requirements.txt
├── Procfile                 # untuk deployment (gunicorn)
├── data/                    # dataset (train/val/test) — TIDAK di-commit ke git
│   ├── train/venomous/
│   ├── train/non_venomous/
│   ├── val/venomous/
│   ├── val/non_venomous/
│   ├── test/venomous/
│   └── test/non_venomous/
├── model/                   # model .h5 hasil training (TIDAK di-commit ke git)
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   └── about.html
└── static/
    ├── css/style.css
    └── uploads/             # gambar yang diunggah pengguna
```

## 2. Menyiapkan Dataset

Dataset **tidak disertakan** dalam proyek ini karena ukurannya besar dan
berkaitan dengan hak cipta gambar. Silakan unduh dataset ular dari sumber
terbuka, misalnya di Kaggle:

- Cari kata kunci seperti **"snake species classification"**,
  **"venomous non venomous snake"**, atau **"snake identification dataset"**.
- Contoh dataset yang bisa dijadikan acuan (cek lisensi masing-masing sebelum
  dipakai): *Snake Species Dataset*, *Indian Snakes Dataset*, dsb.

Setelah diunduh, kelompokkan gambar ke folder `data/train`, `data/val`, dan
`data/test`, masing-masing berisi dua subfolder:

```
venomous/       -> foto-foto ular berbisa (kobra, viper, mamba, dll.)
non_venomous/   -> foto-foto ular tidak berbisa (sanca, ular tikus, dll.)
```

Disarankan proporsi **70% train / 15% validasi / 15% test**, dengan jumlah
gambar seimbang antara dua kelas agar model tidak bias.

## 3. Instalasi

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Melatih Model

```bash
python train_model.py
```

Script ini akan:
1. Melakukan **feature extraction** (membekukan seluruh layer VGG16, hanya
   melatih classifier baru).
2. Melakukan **fine-tuning** pada blok konvolusi terakhir (`block5`) dengan
   learning rate yang lebih kecil.
3. Menyimpan model terbaik ke `model/snake_vgg16_best.h5` dan model final ke
   `model/snake_vgg16_final.h5`.
4. Menyimpan grafik akurasi/loss ke `model/training_history.png` (untuk
   dilampirkan di laporan).

> Training sebaiknya dijalankan di lingkungan dengan GPU (Google Colab,
> Kaggle Notebook, atau komputer lokal ber-GPU) karena VGG16 cukup berat.

## 5. Menjalankan Aplikasi Secara Lokal

Pastikan `model/snake_vgg16_final.h5` sudah ada (hasil langkah 4), lalu:

```bash
python app.py
```

Buka browser ke `http://127.0.0.1:5000`.

## 6. Deployment (Hosting Gratis)

Heroku free tier sudah tidak tersedia lagi. Beberapa alternatif gratis yang
masih aktif untuk aplikasi Flask + model deep learning:

### Opsi A — Render.com (disarankan)
1. Push proyek ini ke GitHub (tanpa folder `data/`, model bisa disertakan
   jika ukurannya masih di bawah limit, atau host modelnya terpisah, misalnya
   di Hugging Face Hub / Google Drive, lalu unduh saat startup).
2. Buat akun di https://render.com, pilih **New Web Service**, hubungkan
   repo GitHub.
3. Build command: `pip install -r requirements.txt`
   Start command: `gunicorn app:app`
4. Deploy — Render akan memberikan URL publik seperti
   `https://nama-app.onrender.com`.

### Opsi B — Hugging Face Spaces (Gradio/Flask + Docker)
1. Buat Space baru dengan SDK **Docker**.
2. Tambahkan `Dockerfile` sederhana yang menjalankan `gunicorn app:app`.
3. Push kode via git ke repo Space tersebut.

### Opsi C — Railway.app
Serupa dengan Render: hubungkan repo GitHub, atur start command
`gunicorn app:app`, deploy otomatis setiap push.

> Catatan: karena file model `.h5` VGG16 relatif besar (bisa >80MB
> tergantung arsitektur classifier tambahan), pertimbangkan menyimpan model
> di Git LFS, Hugging Face Hub, atau Google Drive, lalu diunduh otomatis
> saat aplikasi start jika platform hosting membatasi ukuran repo.

## 7. Alur Penggunaan Aplikasi

1. Buka halaman utama.
2. Unggah foto ular (JPG/PNG).
3. Klik **Analisis Gambar**.
4. Aplikasi menampilkan label **Berbisa** / **Tidak Berbisa** beserta
   tingkat keyakinan model.

## 8. Disclaimer

Aplikasi ini dibuat untuk keperluan tugas akademik. Prediksi model **tidak
boleh** dijadikan satu-satunya dasar keputusan keselamatan saat berinteraksi
dengan ular di alam liar — selalu rujuk pada ahli herpetologi atau otoritas
konservasi setempat.
