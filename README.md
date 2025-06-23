# 🍌 Deteksi Kematangan Buah Pisang Menggunakan KNN dan Ekstraksi Fitur Citra

Proyek ini bertujuan untuk mendeteksi tingkat kematangan buah pisang (mentah/matang) menggunakan citra digital. Sistem ini mengaplikasikan metode ekstraksi fitur tekstur (Histogram Orde Pertama & GLCM) dan algoritma klasifikasi K-Nearest Neighbors (KNN).

---

## 🔧 Fitur Utama

- 📷 Input citra buah pisang (mentah atau matang)
- 🧠 Ekstraksi fitur (8 fitur: warna & tekstur)
- 📊 Klasifikasi menggunakan KNN dengan optimasi nilai **k**
- 🔁 Augmentasi data (flip, rotasi)
- 📈 Cross Validation (5-Fold) untuk validasi akurasi model
- 💾 Menyimpan model dan scaler menggunakan `joblib`
- 🖼️ Antarmuka pengguna berbasis **GUI (Tkinter)**
- ✅ CLI testing seluruh folder uji

---

## 📁 Struktur Folder
project_pcd/
├── dataset/
│ ├── train/
│ │ ├── mentah/
│ │ └── matang/
│ └── test/
│ ├── mentah/
│ └── matang/
├── train_model.py
├── predict.py
├── gui_app.py
├── knn_pisang_model.pkl
└── README.md


## 🧪 Ekstraksi Fitur

Setiap gambar diubah menjadi grayscale dan diekstrak 8 fitur berikut:

1. **Mean** – Rata-rata piksel grayscale
2. **Skewness** – Kemiringan distribusi piksel
3. **Energy (histogram)** – Keseragaman distribusi
4. **Smoothness**
5. **Contrast (GLCM)**
6. **Homogeneity (GLCM)**
7. **ASM (Angular Second Moment)**
8. **Energy (GLCM)**

---

## 🚀 Cara Menjalankan

### 1. Latih Model
dengan cara jalankan "python train_model.py"
dan data akan di simpan di knn_pisang_model.pkl

### 2. Jalankan GUI
dengan cara jalankan "python gui_app.py"
Pilih gambar untuk melihat hasil klasifikasi secara visual.

### 3. Prediksi Batch via CLI
dengan cara jalankan "predict.py"
Secara otomatis akan membaca semua gambar di dataset/test/**

## ✅ Hasil & Akurasi
Akurasi test set: ~88–95%
Rata-rata Cross Validation (5-fold): ~90–93%
Optimasi nilai k: menggunakan cross_val_score pada training + augmentasi

## 🛠️ Teknologi
Python 3.11+
OpenCV
NumPy
Scikit-learn
Scikit-image
Tkinter

## 📚 Referensi
KNN (K-Nearest Neighbors) - scikit-learn
GLCM (Gray Level Co-Occurrence Matrix)
Pengolahan citra digital tekstur
Proyek ini terinspirasi dari penelitian klasifikasi kematangan buah berbasis citra.

## 🙋‍♂️ Kontributor
Angga Putra Pratama – Developer, Dataset, GUI, Dokumentasi


## 📌 Lisensi
Proyek ini dibuat untuk tujuan edukasi dan bebas digunakan untuk pengembangan lebih lanjut.









