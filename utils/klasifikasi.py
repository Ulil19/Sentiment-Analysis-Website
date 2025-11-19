import pandas as pd
from openpyxl import load_workbook
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import datetime as dt
from werkzeug.utils import secure_filename


def run_klasifikasi_data(nama_aplikasi, file_path, img_folder):

    # 1. Load data hasil preprocessing
    df = pd.read_csv(file_path)

    # Standarisasi kolom ke lowercase
    df.columns = [col.strip().lower() for col in df.columns]

    # Drop baris dengan komentar kosong
    df = df.dropna(subset=["komentar"])
    df = df[df["komentar"].str.strip() != ""]

    # 2. TF-IDF pakai kolom "komentar" lowercase
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
    X = vectorizer.fit_transform(df["komentar"])
    y = df["label"]

    # 3. Split data dulu (tanpa SMOTE)
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Terapkan SMOTE hanya pada data latih
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 5. Train Random Forest
    model = RandomForestClassifier(random_state=42, class_weight="balanced")
    model.fit(X_train_res, y_train_res)

    # 6. Prediksi pada test set
    y_pred = model.predict(X_test)

    # 7. Evaluasi
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_dict = classification_report(
        y_test, y_pred, target_names=["negatif", "positif"], output_dict=True
    )
    conf_matrix = confusion_matrix(y_test, y_pred)

    akurasi = accuracy_score(y_test, y_pred)
    presisi = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]

    df_hasil = pd.DataFrame(
        [
            {
                "Akurasi": round(akurasi, 4),
                "Presisi": round(presisi, 4),
                "Recall": round(recall, 4),
                "F1 Score": round(f1, 4),
            }
        ]
    )
    # print(df_hasil)
    # Konversi classification report ke DataFrame
    df_class_report = (
        pd.DataFrame(report_dict)
        .transpose()
        .reset_index()
        .rename(columns={"index": "Label"})
    )

    # Konversi confusion matrix ke DataFrame
    labels = ["negatif", "positif"]
    df_conf_matrix = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    df_conf_matrix.index.name = "Actual"
    df_conf_matrix.columns.name = "Predicted"

    # 9. Gabungkan hasil prediksi ke data test asli
    df_eval = df_test.copy().reset_index(drop=True)
    df_eval["label_prediksi"] = (
        pd.Series(y_pred).map({1: "positif", 0: "negatif"})
        if y_pred.dtype != object
        else y_pred
    )

    # 10. Simpan ke CSV
    eval_csv_path = file_path
    df_eval[["komentar", "label", "label_prediksi"]].to_csv(eval_csv_path, index=False)

    # 11. Simpan classification report ke CSV
    klasifikasi_folder = os.path.join("data", "klasifikasi")
    os.makedirs(klasifikasi_folder, exist_ok=True)

    klasifikasi_file = f"{nama_aplikasi}-klasifikasi.csv"
    klasifikasi_file_path = os.path.join(klasifikasi_folder, klasifikasi_file)
    df_class_report.to_csv(klasifikasi_file_path, index=False)

    # ============== VISUALISASI ===================
    img_dir = os.path.join(img_folder, secure_filename(nama_aplikasi))
    os.makedirs(img_dir, exist_ok=True)
    # grafik confusion metrik
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["negatif", "positif"],
        yticklabels=["negatif", "positif"],
    )
    plt.xlabel("Label Prediksi")
    plt.ylabel("Label Asli")
    plt.title("Confusion Matrix")

    # Buat nama unik untuk gambar
    now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    for fname in os.listdir(img_dir):
        if fname.startswith(f"{nama_aplikasi}-confusion_matrix-") and fname.endswith(
            ".png"
        ):
            os.remove(os.path.join(img_dir, fname))
    img_confusion = os.path.join(
        img_dir, f"{nama_aplikasi}-confusion_matrix-{now_str}.png"
    )
    plt.savefig(img_confusion, bbox_inches="tight")

    # GRAFIK WORDCLOUD
    df_positif = df[df["label"] == "positif"]
    df_negatif = df[df["label"] == "negatif"]

    # Gabungkan semua teks dalam Label positif
    text_positif = " ".join(df_positif["komentar"])

    # Gabungkan semua teks dalam Label negatif
    text_negatif = " ".join(df_negatif["komentar"])

    # GRAFIK WORDCLOUD POSITIF
    plt.figure(figsize=(8, 6))
    wordcloud_pos = WordCloud(
        width=800, height=400, background_color="white", colormap="Greens"
    ).generate(text_positif)

    # Tampilkan wordcloud positif
    plt.imshow(wordcloud_pos, interpolation="bilinear")
    plt.axis("off")
    plt.title("WordCloud Positif")

    # Hapus file wordcloud positif lama jika ada
    for fname in os.listdir(img_dir):
        if fname.startswith(f"{nama_aplikasi}-wordcloud-positif-") and fname.endswith(
            ".png"
        ):
            os.remove(os.path.join(img_dir, fname))

    # Buat nama unik untuk gambar wordcloud positif
    img_wordcloud_pos = os.path.join(
        img_dir, f"{nama_aplikasi}-wordcloud-positif-{now_str}.png"
    )

    # Simpan wordcloud positif
    plt.savefig(img_wordcloud_pos, bbox_inches="tight")
    plt.close()

    # GRAFIK WORDCLOUD NEGATIF
    plt.figure(figsize=(8, 6))
    wordcloud_neg = WordCloud(
        width=800, height=400, background_color="white", colormap="Reds"
    ).generate(text_negatif)

    # Tampilkan wordcloud negatif
    plt.imshow(wordcloud_neg, interpolation="bilinear")
    plt.axis("off")
    plt.title("WordCloud Negatif")

    # Hapus file wordcloud negatif lama jika ada
    for fname in os.listdir(img_dir):
        if fname.startswith(f"{nama_aplikasi}-wordcloud-negatif-") and fname.endswith(
            ".png"
        ):
            os.remove(os.path.join(img_dir, fname))
    # Buat nama unik untuk gambar wordcloud negatif
    img_wordcloud_neg = os.path.join(
        img_dir, f"{nama_aplikasi}-wordcloud-negatif-{now_str}.png"
    )

    # Simpan wordcloud negatif
    plt.savefig(img_wordcloud_neg, bbox_inches="tight")
    plt.close()

    img_paths = [img_confusion, img_wordcloud_pos, img_wordcloud_neg]
    return img_paths, klasifikasi_file_path
