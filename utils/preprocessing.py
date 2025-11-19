import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory,
    ArrayDictionary,
)

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
import nltk
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import os
import datetime as dt

nltk.download("punkt_tab")
nltk.download("punkt")


def run_preprocess_data(file_path, nama_aplikasi):
    import pandas as pd
    import re
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
        StopWordRemoverFactory,
        ArrayDictionary,
    )
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    import nltk
    import os
    import datetime as dt
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.cm import get_cmap

    print("Preprocessing dimulai...")
    df = pd.read_csv(file_path, encoding="utf-8")

    # Standarisasi nama kolom
    df.columns = [col.strip().lower() for col in df.columns]

    # Labeling
    def label_sentiment(score):
        if score in [4, 5]:
            return "positif"
        elif score in [1, 2]:
            return "negatif"
        else:
            return None

    df["label"] = df["bintang"].apply(label_sentiment)
    df = df[df["label"].notnull()].reset_index(drop=True)

    # Visualisasi distribusi label
    img_dir = os.path.join("static", "img", nama_aplikasi)
    os.makedirs(img_dir, exist_ok=True)

    viridis = get_cmap("viridis")
    colors = [viridis(0.3), viridis(0.7)]

    plt.figure(figsize=(8, 6))
    sns.countplot(x="label", data=df, order=["negatif", "positif"], palette=colors)
    plt.title(f"Distribusi Label Sentimen {nama_aplikasi}")
    plt.xlabel("Label Sentimen")
    plt.ylabel("Jumlah")

    # Hapus gambar distribusi lama
    for fname in os.listdir(img_dir):
        if fname.startswith(f"{nama_aplikasi}-sentimen-distribusi"):
            os.remove(os.path.join(img_dir, fname))

    img_filename = f"{nama_aplikasi}-sentimen-distribusi-{dt.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    plt.savefig(os.path.join(img_dir, img_filename), bbox_inches="tight")
    plt.close()

    # === PREPROCESSING KOMENTAR ===
    factory = StopWordRemoverFactory()
    stopwords_default = factory.get_stop_words()
    additional_stopwords = [
        "saya",
        "nya",
        "aja",
        "dong",
        "nih",
        "gw",
        "gue",
        "min",
        "lu",
        "loe",
        "lo",
        "gua",
        "aku",
        "deh",
        "kok",
        "lah",
        "pun",
        "ya",
        "oh",
        "eh",
        "si",
        "dgn",
        "dg",
    ]
    stopword_dict = ArrayDictionary(stopwords_default + additional_stopwords)
    stemmer = StemmerFactory().create_stemmer()
    kamus_koreksi = {"bagu": "bagus"}

    processed_comments = []

    for komentar in df["komentar"].astype(str):
        komentar = komentar.lower()
        komentar = re.sub(r"[^\x00-\x7F]+", "", komentar)
        komentar = re.sub(r"http\S+", "", komentar)
        komentar = re.sub(r"[^a-z\s,]", "", komentar)
        komentar = re.sub(r"\s+", " ", komentar).strip()

        # Tokenisasi
        tokens = nltk.word_tokenize(komentar)

        # Stopword & koreksi kata
        tokens = [
            kamus_koreksi.get(word, word)
            for word in tokens
            if word not in stopword_dict.words and len(word) > 1
        ]

        # Stemming
        tokens_stemmed = [stemmer.stem(word) for word in tokens]

        # Simpan komentar final
        processed_comments.append(" ".join(tokens_stemmed))

    # Ganti kolom komentar dengan versi yang sudah dibersihkan
    df["komentar"] = processed_comments

    # Simpan hasil akhir tanpa kolom tambahan
    df = df[["bintang", "komentar", "label"]]
    df.to_csv(file_path, index=False, encoding="utf-8")
    print("Preprocessing selesai. File diperbarui di:", file_path)
