import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

MODEL_PATH = "data/model/model_kata_rf.pkl"
VECTORIZER_PATH = "data/model/vectorizer_kata.pkl"
UPLOAD_FOLDER = os.path.join("data", "file")


def simple_preprocess(text):
    # Lowercase, hapus tanda baca, dan spasi ekstra
    text = str(text).lower()
    text = re.sub(r"\d+", " ", text)  # Hapus angka dulu
    text = re.sub(
        r"[^a-zA-Z\s]", " ", text
    )  # Hapus tanda baca, sisakan huruf dan spasi
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_kata_model():
    # Gabungkan semua data review dari aplikasi yang sudah classified
    from ..app import load_state  # pastikan import relative sesuai struktur project

    state = load_state()
    apps = [item for item in state if item.get("status") == "classified"]
    texts = []
    labels = []
    for app in apps:
        review_path = os.path.join(UPLOAD_FOLDER, f"{app['nama_aplikasi']}-review.csv")
        if os.path.exists(review_path):
            df = pd.read_csv(review_path)
            if "komentar" in df.columns and "label" in df.columns:
                # Preprocessing komentar
                texts.extend(
                    [simple_preprocess(k) for k in df["komentar"].astype(str).tolist()]
                )
                labels.extend(df["label"].astype(str).tolist())
    if not texts or not labels:
        return None, None
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
    X = vectorizer.fit_transform(texts)
    model = RandomForestClassifier(random_state=42, class_weight="balanced")
    model.fit(X, labels)
    # Simpan model dan vectorizer
    os.makedirs("data", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    return vectorizer, model


def load_kata_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return vectorizer, model
    return None, None


def predict_kata(kalimat):
    vectorizer, model = load_kata_model()
    if vectorizer is None or model is None:
        return "Model belum tersedia. Silakan lakukan klasifikasi data terlebih dahulu."
    # Preprocessing input kata
    kalimat_bersih = simple_preprocess(kalimat)
    X_input = vectorizer.transform([kalimat_bersih])
    pred = model.predict(X_input)[0]
    return pred.capitalize()
