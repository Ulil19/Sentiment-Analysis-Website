from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    make_response,
    send_file,
)
import os
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps
from pathlib import Path
from dotenv import load_dotenv
import jwt
import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from werkzeug.utils import secure_filename
import shutil
import io
from utils.preprocessing import run_preprocess_data
from utils.klasifikasi import run_klasifikasi_data
from utils.model_kata import predict_kata, simple_preprocess

app = Flask(__name__)
# ========= LOAD ENV =========
load_dotenv(Path(__file__).with_suffix(".env"))
app.secret_key = os.getenv("SECRET_KEY", "dev-fallback")

# ========= CONFIG =========
STATE_PATH = os.path.join("data", "state.json")
UPLOAD_FOLDER = os.path.join("data", "file")
IMG_FOLDER = os.path.join("static", "img")
ALLOWED_EXTENSIONS = {"csv"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMG_FOLDER, exist_ok=True)


# ========= STATE HANDLING =========
def load_state():
    if os.path.exists(STATE_PATH) and os.path.getsize(STATE_PATH) > 0:
        with open(STATE_PATH, "r") as f:
            data = json.load(f)
        data.sort(
            key=lambda x: datetime.strptime(x["tanggal_upload"], "%Y-%m-%d %H:%M:%S"),
            reverse=True,
        )
        return data
    return []


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def update_state_for(filename, new_data):
    state = load_state()
    updated = False

    for entry in state:
        if entry["filename"] == filename:
            if "img_paths" in new_data:
                entry.setdefault("img_paths", [])
                for path in new_data["img_paths"]:
                    if path not in entry["img_paths"]:
                        entry["img_paths"].append(path)
                del new_data["img_paths"]
            entry.update(new_data)
            updated = True
            break

    if not updated:
        new_entry = {
            "filename": filename,
            "nama_aplikasi": new_data.get(
                "nama_aplikasi", filename.split("-review")[0]
            ),
        }
        if "img_paths" in new_data:
            new_entry["img_paths"] = new_data["img_paths"]
            del new_data["img_paths"]
        new_entry.update(new_data)
        state.append(new_entry)

    save_state(state)


# ========= AUTH =========
CREDS = {
    "admin@gmail.com": {
        "password_hash": generate_password_hash("admin123"),
        "username": "Admin",
    }
}


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip()
        pwd = request.form["password"].strip()
        user = CREDS.get(email)
        if user and check_password_hash(user["password_hash"], pwd):
            session["user"] = email
            session["username"] = user["username"]
            token = jwt.encode(
                {"id": email, "exp": datetime.now() + timedelta(days=1)},
                app.secret_key,
                algorithm="HS256",
            )
            resp = make_response(redirect(url_for("dashboard")))
            resp.set_cookie("token", token, httponly=True, secure=True)
            return resp
        else:
            flash("Email atau password salah!", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Anda sudah logout", "info")
    return redirect(url_for("login"))


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            flash("Silakan login dulu", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return wrapper


# ========= DASHBOARD =========
@app.route("/")
@login_required
def dashboard():
    projects = load_state()
    return render_template(
        "dashboard.html", username=session["username"], projects=projects
    )


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ========= UPLOAD ROUTE =========
@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    uploaded_file = request.files.get("file")
    nama_aplikasi = request.form.get("nama_aplikasi", "").strip().lower()

    if not nama_aplikasi:
        flash("Nama aplikasi harus diisi.", "danger")
        return redirect(url_for("dashboard"))

    if uploaded_file and allowed_file(uploaded_file.filename):
        filename = f"{secure_filename(nama_aplikasi)}-review.csv"
        save_path = os.path.join(UPLOAD_FOLDER, filename)

        # Cek apakah file sudah ada
        if os.path.exists(save_path):
            flash(
                f"File {filename} sudah ada. Ganti nama aplikasi untuk upload baru.",
                "warning",
            )
            return redirect(url_for("dashboard"))

        # Simpan CSV
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            flash("Gagal membaca file. Pastikan format CSV benar.", "danger")
            return redirect(url_for("dashboard"))

        # Normalisasi kolom
        df.columns = [col.lower().strip() for col in df.columns]

        if "bintang" not in df.columns:
            flash("Kolom 'bintang' tidak ditemukan!", "danger")
            return redirect(url_for("dashboard"))

        # Simpan file sebagai UTF-8
        df.to_csv(save_path, index=False, encoding="utf-8")

        # === VISUALISASI ===
        img_dir = os.path.join(IMG_FOLDER, secure_filename(nama_aplikasi))
        os.makedirs(img_dir, exist_ok=True)

        sns.countplot(x="bintang", data=df, order=[1, 2, 3, 4, 5], palette="viridis")
        plt.title(f"Distribusi Rating Aplikasi: {nama_aplikasi}")
        plt.xlabel("Bintang")
        plt.ylabel("Jumlah Ulasan")

        now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"{secure_filename(nama_aplikasi)}-bintang-{now_str}.png"
        img_path = os.path.join(img_dir, img_filename)
        plt.savefig(img_path, bbox_inches="tight")
        plt.close()

        web_img_path = img_path.replace("\\", "/")

        update_state_for(
            filename,
            {
                "nama_aplikasi": nama_aplikasi,
                "filename": filename,
                "filepath": save_path.replace("\\", "/"),
                "img_paths": [web_img_path],
                "status": "uploaded",
                "tanggal_upload": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

        flash("File berhasil diupload dan divisualisasikan!", "success")
    else:
        flash("File tidak valid atau bukan .csv", "danger")

    return redirect(url_for("dashboard"))


@app.route("/delete/<nama_aplikasi>", methods=["POST"])
@login_required
def delete_file(nama_aplikasi):
    # Pastikan nama valid
    safe_name = secure_filename(nama_aplikasi)

    # Hapus file CSV
    file_path = os.path.join(UPLOAD_FOLDER, f"{safe_name}-review.csv")
    if os.path.exists(file_path):
        os.remove(file_path)

    # Hapus folder visualisasi
    vis_folder = os.path.join(IMG_FOLDER, safe_name)
    if os.path.exists(vis_folder):
        shutil.rmtree(vis_folder)

    # Hapus file klasifikasi jika ada
    klasifikasi_path = os.path.join(
        "data", "klasifikasi", f"{safe_name}-klasifikasi.csv"
    )
    if os.path.exists(klasifikasi_path):
        os.remove(klasifikasi_path)

    # Update state.json
    state = load_state()
    updated_state = [item for item in state if item["nama_aplikasi"] != nama_aplikasi]
    with open(STATE_PATH, "w") as f:
        json.dump(updated_state, f, indent=4)

    flash(f"Data dan visualisasi untuk '{nama_aplikasi}' berhasil dihapus.", "success")
    return redirect(url_for("dashboard"))


@app.route("/", methods=["POST", "GET"])
@login_required
def klasifikasi_kata():
    kata = request.form.get("kata", "").strip()
    projects = load_state()
    hasil = None
    kata_bersih = kata
    if kata:
        kata_bersih = simple_preprocess(kata)
        hasil = predict_kata(kata)
    return render_template(
        "dashboard.html",
        username=session["username"],
        kata=kata_bersih, 
        hasil_klasifikasi=hasil,
        projects=projects,
    )


@app.route("/preprocess", methods=["GET", "POST"])
@login_required
def preprocess():
    # Filter status 'uploaded'
    state = [item for item in load_state() if item.get("status") == "uploaded"]
    # print(state)
    selected_app = request.form.get("nama_aplikasi")

    data_preview = None
    columns = []
    file_path = None

    if request.method == "POST" and selected_app:
        file_entry = next(
            (item for item in state if item["nama_aplikasi"] == selected_app), None
        )
        if file_entry:
            file_path = os.path.join(UPLOAD_FOLDER, file_entry["filename"])
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    data_preview = df.to_dict(orient="records")
                    # print(data_preview[:5])
                    columns = df.columns.tolist()
                except Exception as e:
                    flash(f"Error saat membaca file: {str(e)}", "error")

    return render_template(
        "preprocess.html",
        projects=state,
        selected_app=selected_app,
        data_preview=data_preview,
        columns=columns,
    )


@app.route("/preprocess/run", methods=["POST"])
@login_required
def run_preprocess():
    nama_aplikasi = request.form.get("nama_aplikasi")
    # print(nama_aplikasi)

    if not nama_aplikasi:
        flash("Nama aplikasi tidak ditemukan!", "danger")
        return redirect(url_for("preprocess"))

    # Cek di state.json
    state = load_state()
    project = next((p for p in state if p["nama_aplikasi"] == nama_aplikasi), None)
    if not project:
        flash("File aplikasi tidak ditemukan di state!", "danger")
        return redirect(url_for("preprocess"))

    file_path = project["filepath"]
    # print(file_path)

    try:
        run_preprocess_data(file_path, nama_aplikasi)
        project["status"] = "preprocessed"
        save_state(state)
        flash("Preprocessing berhasil dijalankan!", "success")
    except Exception as e:
        flash(f"Terjadi kesalahan saat preprocessing: {str(e)}", "danger")

    return redirect(url_for("preprocess"))


@app.route("/klasifikasi", methods=["GET"])
@login_required
def klasifikasi():
    state = load_state()
    projects = [item for item in state if item.get("status") == "preprocessed"]
    return render_template("klasifikasi.html", projects=projects)


@app.route("/preview-klasifikasi", methods=["POST"])
@login_required
def preview_klasifikasi():
    selected_app = request.form.get("file")

    if not selected_app:
        flash("Silakan pilih file terlebih dahulu.", "warning")
        return redirect(url_for("klasifikasi"))

    # Ambil data dari state.json
    state = load_state()
    project = next(
        (item for item in state if item["nama_aplikasi"] == selected_app), None
    )

    if not project:
        flash("Data tidak ditemukan.", "danger")
        return redirect(url_for("klasifikasi"))

    file_path = project.get("filepath")
    if not file_path or not os.path.exists(file_path):
        flash("File tidak ditemukan secara fisik.", "danger")
        return redirect(url_for("klasifikasi"))

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except Exception:
        flash("Gagal membaca file CSV.", "danger")
        return redirect(url_for("klasifikasi"))

    preview_data = df.to_dict(orient="records")

    # Ambil ulang project list
    projects = [item for item in state if item.get("status") == "preprocessed"]

    return render_template(
        "klasifikasi.html",
        projects=projects,
        selected_app=selected_app,
        preview_data=preview_data,
    )


@app.route("/klasifikasi/run", methods=["POST"])
@login_required
def run_klasifikasi():
    selected_app = request.form.get("nama_aplikasi")
    # print(selected_app)

    if not selected_app:
        flash("Nama aplikasi tidak ditemukan.", "warning")
        return redirect(url_for("klasifikasi"))

    state = load_state()
    project = next((p for p in state if p["nama_aplikasi"] == selected_app), None)

    if not project:
        flash("Data tidak ditemukan di state.json.", "danger")
        return redirect(url_for("klasifikasi"))

    file_path = project.get("filepath")
    if not file_path or not os.path.exists(file_path):
        flash("File tidak ditemukan secara fisik.", "danger")
        return redirect(url_for("klasifikasi"))

    try:
        img_paths, klasifikasi_file_path = run_klasifikasi_data(
            selected_app, file_path, IMG_FOLDER
        )
        for idx, project in enumerate(state):
            if project["nama_aplikasi"] == selected_app:
                if "img_paths" not in project:
                    project["img_paths"] = []
                project["img_paths"].extend(img_paths)
                project["klasifikasi_path"] = klasifikasi_file_path
                project["status"] = "classified"
                state[idx] = project
                break

        save_state(state)

    except Exception as e:
        flash(f"Gagal menjalankan klasifikasi: {str(e)}", "danger")

    return redirect(url_for("klasifikasi"))


@app.route("/hasil", methods=["GET", "POST"])
@login_required
def hasil():
    state = load_state()
    apps = [item for item in state if item.get("status") == "classified"]

    selected_apps = request.form.getlist("apps")  # dari checkbox input
    results = []

    if selected_apps:
        for nama_aplikasi in selected_apps:
            try:
                # File CSV klasifikasi dan review
                klasifikasi_path = f"data/klasifikasi/{nama_aplikasi}-klasifikasi.csv"
                review_path = f"data/file/{nama_aplikasi}-review.csv"

                classification_report = pd.read_csv(klasifikasi_path)
                review_data = pd.read_csv(review_path)

                # Ambil semua file gambar dari static/img/{nama_aplikasi}
                img_dir = os.path.join("static", "img", nama_aplikasi)
                images = []
                if os.path.exists(img_dir):
                    for filename in os.listdir(img_dir):
                        if filename.endswith(".png"):
                            images.append(f"img/{nama_aplikasi}/{filename}")

                results.append(
                    {
                        "nama_aplikasi": nama_aplikasi,
                        "classification_report": classification_report,
                        "review_data": review_data,
                        "images": images,
                    }
                )

            except Exception as e:
                flash(f"[ERROR] Gagal memuat data untuk {nama_aplikasi}: {e}")

    return render_template("hasil.html", apps=apps, results=results)


@app.route("/export_excel", methods=["POST"])
@login_required
def export_excel():
    nama_aplikasi = request.form.get("nama_aplikasi")
    if not nama_aplikasi:
        flash("Nama aplikasi tidak diberikan.", "error")
        return redirect(url_for("hasil"))

    # Path file CSV
    klasifikasi_path = f"data/klasifikasi/{nama_aplikasi}-klasifikasi.csv"
    review_path = f"data/file/{nama_aplikasi}-review.csv"

    # Validasi file
    if not os.path.exists(klasifikasi_path):
        flash(f"File klasifikasi untuk {nama_aplikasi} tidak ditemukan.", "error")
        return redirect(url_for("hasil"))
    if not os.path.exists(review_path):
        flash(f"File review untuk {nama_aplikasi} tidak ditemukan.", "error")
        return redirect(url_for("hasil"))

    try:
        # Baca CSV
        df_klasifikasi = pd.read_csv(klasifikasi_path)
        df_review = pd.read_csv(review_path)

        # Export ke Excel multi-sheet
        os.makedirs("temp", exist_ok=True)
        filename = (
            f"{nama_aplikasi}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        export_path = os.path.join("temp", filename)

        with pd.ExcelWriter(export_path, engine="xlsxwriter") as writer:
            df_review.to_excel(writer, index=False, sheet_name="Data Ulasan")
            df_klasifikasi.to_excel(
                writer, index=False, sheet_name="Classification Report"
            )

        # Kirim file untuk diunduh
        return send_file(
            export_path,
            as_attachment=True,
            download_name=f"{nama_aplikasi}_report.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        print(f"[ERROR] Gagal export Excel: {e}")
        flash(f"Terjadi kesalahan saat mengekspor: {e}", "error")
        return redirect(url_for("hasil"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
