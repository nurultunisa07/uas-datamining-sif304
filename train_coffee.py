"""
train_coffee.py
----------------
Melatih model K-Means untuk mengelompokkan lokasi gerai kopi dan menandai
zona "sepi" (klaster dengan rata-rata jumlah pelanggan/kepadatan terendah).

Cara pakai:
1. Download dataset gerai kopi dari link Google Drive yang diberikan dosen,
   lalu simpan sebagai "data/coffee_shops.csv"
2. Sesuaikan nama kolom di bagian FEATURE_COLS jika berbeda dengan dataset asli
   (contoh umum: latitude, longitude, jumlah_pelanggan, kepadatan_penduduk, jarak_kompetitor)
3. Jalankan: python train_coffee.py
4. Model tersimpan otomatis ke folder models/
"""

import pandas as pd
import numpy as np
import pickle
import json
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA_PATH = "data/coffee_shops.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- SESUAIKAN dengan nama kolom pada dataset asli Anda ---
LAT_COL = "y"
LON_COL = "x"
# fitur tambahan untuk clustering, selain koordinat
EXTRA_FEATURE_COLS = ["population_density", "traffic_flow", "competitor_count", "is_commercial"]

N_CLUSTERS = 4


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def find_best_k(X_scaled, k_min=2, k_max=8):
    """Opsional: cari k terbaik pakai elbow/silhouette sederhana"""
    from sklearn.metrics import silhouette_score
    scores = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
    best_k = max(scores, key=scores.get)
    return best_k, scores


def train_and_save():
    df = load_data()
    feature_cols = [LAT_COL, LON_COL] + EXTRA_FEATURE_COLS
    X = df[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_counts = df["cluster"].value_counts().to_dict()

    # Tentukan "zona sepi": klaster dengan rata-rata population_density + traffic_flow
    # PALING RENDAH (proxy area yang kurang potensial untuk gerai kopi)
    if "population_density" in df.columns and "traffic_flow" in df.columns:
        # normalisasi sederhana sebelum dijumlah supaya skala kedua fitur sebanding
        pd_norm = (df["population_density"] - df["population_density"].min()) / \
                  (df["population_density"].max() - df["population_density"].min())
        tf_norm = (df["traffic_flow"] - df["traffic_flow"].min()) / \
                  (df["traffic_flow"].max() - df["traffic_flow"].min())
        df["potential_score"] = pd_norm + tf_norm
        avg_per_cluster = df.groupby("cluster")["potential_score"].mean()
        sepi_cluster = avg_per_cluster.idxmin()
    else:
        sepi_cluster = min(cluster_counts, key=cluster_counts.get)

    df["zona"] = df["cluster"].apply(lambda c: "Sepi" if c == sepi_cluster else "Ramai")

    # simpan model & scaler
    with open(f"{MODEL_DIR}/kmeans_coffee.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    with open(f"{MODEL_DIR}/coffee_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # simpan data hasil clustering untuk divisualisasikan di Streamlit
    df.to_csv(f"{MODEL_DIR}/coffee_clustered.csv", index=False)

    meta = {
        "feature_cols": feature_cols,
        "n_clusters": N_CLUSTERS,
        "sepi_cluster": int(sepi_cluster),
        "cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()},
    }
    with open(f"{MODEL_DIR}/coffee_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Selesai. Hasil clustering tersimpan di 'models/coffee_clustered.csv'")
    print(f"Klaster yang ditandai sebagai ZONA SEPI: cluster {sepi_cluster}")
    print("Jumlah anggota tiap klaster:", cluster_counts)


if __name__ == "__main__":
    train_and_save()