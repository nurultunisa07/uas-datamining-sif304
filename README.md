# UAS Data Mining (SIF304) — Genap 2025/2026

**Nama :** [ISI NAMA ANDA]
**NIM  :** [ISI NIM ANDA]
**Dosen Pengampu :** Teuku Rizky Noviandy, S.Kom., M.Kom.

## 📌 Deskripsi Proyek

Aplikasi web berbasis **Streamlit** yang mendemonstrasikan dua pendekatan *data mining*:

1. **Klasifikasi (Supervised Learning)** — memprediksi status diabetes pasien
   menggunakan tiga algoritma: **K-Nearest Neighbors (KNN)**, **Naive Bayes**,
   dan **Decision Tree**, dievaluasi dengan akurasi, precision, recall, F1-score,
   dan confusion matrix.
2. **Clustering (Unsupervised Learning)** — mengelompokkan lokasi gerai kopi
   menggunakan **K-Means**, memvisualisasikan sebaran klaster pada peta, serta
   mengidentifikasi **zona sepi** (klaster dengan kepadatan gerai/pelanggan
   paling rendah).

## 🗂️ Struktur Proyek

```
uas_datamining/
├── app.py                  # Aplikasi utama Streamlit
├── train_diabetes.py       # Script training model klasifikasi diabetes
├── train_coffee.py         # Script training model clustering gerai kopi
├── requirements.txt        # Daftar dependensi
├── data/
│   ├── diabetes.csv        # Dataset Pima Indians Diabetes (unduh dari Kaggle)
│   └── coffee_shops.csv    # Dataset lokasi gerai kopi (unduh dari Google Drive)
├── models/                 # Model & metrik hasil training (dibuat otomatis)
└── README.md
```

## ⚙️ Instruksi Menjalankan Aplikasi

### 1. Clone repository
```bash
git clone https://github.com/USERNAME/NAMA-REPO.git
cd NAMA-REPO
```

### 2. Buat virtual environment (opsional tapi disarankan)
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependensi
```bash
pip install -r requirements.txt
```

### 4. Siapkan dataset
- Download dataset diabetes dari:
  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
  → simpan sebagai `data/diabetes.csv`
- Download dataset gerai kopi dari link Google Drive yang diberikan dosen
  → simpan sebagai `data/coffee_shops.csv`

### 5. Latih model
```bash
python train_diabetes.py
python train_coffee.py
```

### 6. Jalankan aplikasi
```bash
streamlit run app.py
```

## 🌐 Link Aplikasi Streamlit (Live Demo)

[ISI LINK STREAMLIT CLOUD ANDA DI SINI]
Contoh: https://nama-proyek-anda.streamlit.app

## 🚀 Deployment ke Streamlit Cloud

1. Push seluruh folder proyek (termasuk folder `models/` hasil training) ke GitHub.
2. Buka https://share.streamlit.io/ lalu login dengan akun GitHub.
3. Klik **New app**, pilih repository, branch, dan file utama `app.py`.
4. Klik **Deploy**. Tunggu proses build selesai.
5. Salin link aplikasi yang aktif, lalu tempelkan di bagian atas README ini.

## 📊 Ringkasan Hasil

| Model          | Akurasi | Precision | Recall | F1-Score |
|----------------|---------|-----------|--------|----------|
| KNN            | -       | -         | -      | -        |
| Naive Bayes    | -       | -         | -      | -        |
| Decision Tree  | -       | -         | -      | -        |

*(Isi tabel di atas dengan hasil dari `models/diabetes_metrics.json` setelah training)*
