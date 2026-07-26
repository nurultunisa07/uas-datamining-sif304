"""
app.py
------
Aplikasi Streamlit UAS Data Mining (SIF304)
Berisi 2 halaman:
1. Prediksi Risiko Diabetes (Klasifikasi: KNN, Naive Bayes, Decision Tree)
2. Analisis Klaster Lokasi Gerai Kopi & Deteksi Zona Sepi (K-Means)

Jalankan lokal dengan:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(
    page_title="UAS Data Mining - SIF304",
    page_icon="📊",
    layout="wide",
)

MODEL_DIR = "models"

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGASI
# ----------------------------------------------------------------------------
st.sidebar.title("📊 UAS Data Mining")
st.sidebar.caption("SIF304 - Genap 2025/2026")
page = st.sidebar.radio(
    "Pilih Halaman",
    ["🏠 Beranda", "🩺 Klasifikasi Diabetes", "☕ Clustering Gerai Kopi"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dosen Pengampu:**\nTeuku Rizky Noviandy, S.Kom., M.Kom.")


# ----------------------------------------------------------------------------
# HALAMAN BERANDA
# ----------------------------------------------------------------------------
if page == "🏠 Beranda":
    st.title("Ujian Akhir Semester - Data Mining (SIF304)")
    st.markdown("""
    Aplikasi ini mendemonstrasikan dua pendekatan utama dalam *data mining*:

    - **Supervised Learning (Klasifikasi):** memprediksi status diabetes pasien
      menggunakan tiga algoritma — KNN, Naive Bayes, dan Decision Tree.
    - **Unsupervised Learning (Clustering):** mengelompokkan lokasi gerai kopi
      dengan K-Means dan mengidentifikasi zona dengan potensi pelanggan rendah
      ("zona sepi").

    Gunakan menu di sebelah kiri untuk berpindah halaman.
    """)
    st.info("Isi Nama dan NIM Anda di file README.md sebelum deploy.")


# ----------------------------------------------------------------------------
# HALAMAN KLASIFIKASI DIABETES
# ----------------------------------------------------------------------------
elif page == "🩺 Klasifikasi Diabetes":
    st.title("🩺 Prediksi Risiko Diabetes Berdasarkan Data Pasien")

    st.markdown("""
    **Deskripsi Proyek:**
    Model ini memprediksi apakah seorang pasien berisiko mengidap diabetes
    berdasarkan delapan fitur kesehatan (jumlah kehamilan, kadar glukosa,
    tekanan darah, ketebalan kulit, kadar insulin, BMI, riwayat keturunan
    diabetes, dan usia). Tiga algoritma klasifikasi dibandingkan performanya:
    **K-Nearest Neighbors (KNN)**, **Naive Bayes**, dan **Decision Tree**.
    """)

    metrics_path = f"{MODEL_DIR}/diabetes_metrics.json"

    if not os.path.exists(metrics_path):
        st.error(
            "Model belum dilatih. Jalankan `python train_diabetes.py` terlebih "
            "dahulu setelah menaruh dataset di data/diabetes.csv"
        )
    else:
        with open(metrics_path) as f:
            metrics = json.load(f)

        st.subheader("📈 Perbandingan Metrik Evaluasi Model")
        metric_df = pd.DataFrame({
            model_name: {
                "Akurasi": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1-Score": m["f1_score"],
            }
            for model_name, m in metrics.items()
        }).T
        st.dataframe(metric_df.style.format("{:.2%}"), use_container_width=True)

        st.subheader("🔲 Confusion Matrix")
        cols = st.columns(len(metrics))
        for col, (model_name, m) in zip(cols, metrics.items()):
            with col:
                st.markdown(f"**{model_name}**")
                cm = np.array(m["confusion_matrix"])
                fig, ax = plt.subplots(figsize=(3, 3))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Tidak Diabetes", "Diabetes"],
                    yticklabels=["Tidak Diabetes", "Diabetes"], ax=ax
                )
                ax.set_xlabel("Prediksi")
                ax.set_ylabel("Aktual")
                st.pyplot(fig)

        st.markdown("---")
        st.subheader("🔮 Coba Prediksi Pasien Baru")

        model_choice = st.selectbox("Pilih Model", list(metrics.keys()))

        c1, c2 = st.columns(2)
        with c1:
            pregnancies = st.number_input("Jumlah Kehamilan (Pregnancies)", 0, 20, 1)
            glucose = st.number_input("Kadar Glukosa (Glucose)", 0, 300, 120)
            blood_pressure = st.number_input("Tekanan Darah (BloodPressure)", 0, 200, 70)
            skin_thickness = st.number_input("Ketebalan Kulit (SkinThickness)", 0, 100, 20)
        with c2:
            insulin = st.number_input("Kadar Insulin (Insulin)", 0, 900, 80)
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Usia (Age)", 1, 120, 30)

        if st.button("Prediksi", type="primary"):
            safe_name = model_choice.lower().replace(" ", "_")
            with open(f"{MODEL_DIR}/{safe_name}.pkl", "rb") as f:
                model = pickle.load(f)
            with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
                scaler = pickle.load(f)

            input_df = pd.DataFrame([{
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age,
            }])
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]

            if pred == 1:
                st.error("⚠️ Pasien **diprediksi MENGIDAP diabetes**.")
            else:
                st.success("✅ Pasien **diprediksi TIDAK mengidap diabetes**.")


# ----------------------------------------------------------------------------
# HALAMAN CLUSTERING GERAI KOPI
# ----------------------------------------------------------------------------
elif page == "☕ Clustering Gerai Kopi":
    st.title("☕ Analisis Klaster Lokasi Gerai Kopi dan Deteksi Zona Sepi")

    st.markdown("""
    **Deskripsi Proyek:**
    Clustering (K-Means) digunakan untuk mengelompokkan gerai kopi berdasarkan
    lokasi geografis (dan variabel lain seperti kepadatan pelanggan/lingkungan
    sekitar, jika tersedia). Klaster dengan jumlah gerai/kepadatan terendah
    ditandai sebagai **zona sepi** — area yang berpotensi kurang menguntungkan
    untuk pembukaan gerai baru.
    """)

    clustered_path = f"{MODEL_DIR}/coffee_clustered.csv"
    meta_path = f"{MODEL_DIR}/coffee_meta.json"

    if not (os.path.exists(clustered_path) and os.path.exists(meta_path)):
        st.error(
            "Model belum dilatih. Jalankan `python train_coffee.py` terlebih "
            "dahulu setelah menaruh dataset di data/coffee_shops.csv"
        )
    else:
        df = pd.read_csv(clustered_path)
        with open(meta_path) as f:
            meta = json.load(f)

        lat_col, lon_col = meta["feature_cols"][0], meta["feature_cols"][1]
        extra_cols = meta["feature_cols"][2:]

        st.subheader("🗺️ Sebaran Klaster Gerai Kopi (Koordinat X, Y)")
        fig = px.scatter(
            df, x=lon_col, y=lat_col,
            color="zona",
            hover_data=["cluster"] + extra_cols,
            height=550,
            color_discrete_map={"Ramai": "#2E7D32", "Sepi": "#C62828"},
            labels={lon_col: "Koordinat X", lat_col: "Koordinat Y"},
        )
        fig.update_traces(marker=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Jumlah Gerai per Klaster")
        st.bar_chart(df["cluster"].value_counts().sort_index())

        st.info(f"Klaster **{meta['sepi_cluster']}** ditandai sebagai **zona sepi** "
                f"karena memiliki jumlah gerai paling sedikit dibanding klaster lain.")

        st.markdown("---")
        st.subheader("📍 Cek Lokasi Baru")
        c1, c2, c3 = st.columns(3)
        with c1:
            new_lon = st.number_input("Koordinat X", value=float(df[lon_col].mean()), format="%.4f")
            new_lat = st.number_input("Koordinat Y", value=float(df[lat_col].mean()), format="%.4f")
        with c2:
            new_pop = st.number_input("Population Density", value=float(df["population_density"].mean()), format="%.2f")
            new_traffic = st.number_input("Traffic Flow", value=float(df["traffic_flow"].mean()), format="%.2f")
        with c3:
            new_comp = st.number_input("Competitor Count", min_value=0, value=int(df["competitor_count"].mean()))
            new_commercial = st.selectbox("Area Komersial? (is_commercial)", [0, 1], index=1)

        if st.button("Cek Klaster Lokasi Ini", type="primary"):
            with open(f"{MODEL_DIR}/kmeans_coffee.pkl", "rb") as f:
                kmeans = pickle.load(f)
            with open(f"{MODEL_DIR}/coffee_scaler.pkl", "rb") as f:
                scaler = pickle.load(f)

            new_point = pd.DataFrame([{
                lon_col: new_lon,
                lat_col: new_lat,
                "population_density": new_pop,
                "traffic_flow": new_traffic,
                "competitor_count": new_comp,
                "is_commercial": new_commercial,
            }])[meta["feature_cols"]]  # urutkan kolom sesuai saat training

            new_scaled = scaler.transform(new_point)
            cluster_pred = kmeans.predict(new_scaled)[0]
            zona = "Sepi" if cluster_pred == meta["sepi_cluster"] else "Ramai"

            if zona == "Sepi":
                st.error(f"📍 Lokasi ini masuk **Klaster {cluster_pred}** → Zona **SEPI** ⚠️")
            else:
                st.success(f"📍 Lokasi ini masuk **Klaster {cluster_pred}** → Zona **RAMAI** ✅")