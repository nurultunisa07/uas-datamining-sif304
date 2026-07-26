"""
train_diabetes.py
------------------
Melatih 3 model klasifikasi (KNN, Naive Bayes, Decision Tree) untuk memprediksi
status diabetes berdasarkan dataset Pima Indians Diabetes.

Cara pakai:
1. Download dataset dari Kaggle:
   https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Simpan file "diabetes.csv" ke dalam folder data/
3. Jalankan: python train_diabetes.py
4. Model & metrik akan tersimpan otomatis di folder models/
"""

import pandas as pd
import numpy as np
import pickle
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

DATA_PATH = "data/diabetes.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET_COL = "Outcome"


def load_data():
    df = pd.read_csv(DATA_PATH)
    # Beberapa kolom pada dataset Pima memakai 0 sebagai nilai hilang (missing),
    # kecuali Pregnancies dan Outcome. Kita ganti 0 -> median kolom tsb.
    cols_with_invalid_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_with_invalid_zero:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    return df


def train_and_evaluate():
    df = load_data()
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    }

    results = {}

    for name, model in models.items():
        # KNN & Naive Bayes lebih baik pakai data yang sudah di-scale,
        # Decision Tree tidak sensitif terhadap scaling tapi kita samakan saja
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
        }

        # simpan model
        safe_name = name.lower().replace(" ", "_")
        with open(f"{MODEL_DIR}/{safe_name}.pkl", "wb") as f:
            pickle.dump(model, f)

        print(f"[{name}] acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

    # simpan scaler (dipakai lagi saat prediksi data baru di app.py)
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # simpan metrik ke json supaya bisa ditampilkan di Streamlit tanpa re-train
    with open(f"{MODEL_DIR}/diabetes_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSelesai. Model & metrik tersimpan di folder 'models/'.")


if __name__ == "__main__":
    train_and_evaluate()
