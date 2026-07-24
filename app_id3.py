# app.py

import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path
import streamlit.components.v1 as components

from model_id3 import cross_validation_id3_stratified
from model_id3 import get_model_info
from model_id3 import visualize_tree
from model_id3 import get_evaluation_detail
from model_id3 import predict_single_row
from model_id3 import get_split_summary

from model_id3 import (
    get_preparation_steps,
    FITUR_X,
    TARGET_Y,
    split_data,
    train_model,
    evaluate_model,
    calculate_entropy,
    get_information_gain_detail,
    get_information_gain_ranking,
    get_recursive_node_info,
    get_node_status_info,
    get_tree_attribute_branch_summary,
    get_cross_validation_fold_tree,
    get_unformed_test_attribute_detail,
    get_cross_validation_data_split_detail,
    extract_rules_from_tree,
    visualize_tree_dot,
    visualize_tree_svg_bytes,
    visualize_tree_png_bytes
)

# =============================
# CONFIG PAGE
# =============================
st.set_page_config(
    page_title="Data Mining ID3",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================
# LOAD CUSTOM CSS
# =============================
def load_css(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

CSS_FILE = Path(__file__).parent / "assets" / "style.css"
load_css(CSS_FILE)

# =============================
# MATPLOTLIB STYLE
# =============================
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#f8fafc",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#334155",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "text.color": "#1a202c",
    "grid.color": "#e2e8f0",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CHART_COLORS = ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed", "#0891b2"]


# =============================
# LOGO & JUDUL SIDEBAR
# =============================
LOGO_PLN = Path(__file__).parent / "assets" / "logo_pln.png"

col_logo, col_title = st.sidebar.columns([0.28, 0.72])

with col_logo:
    st.image(
        str(LOGO_PLN),
        width=52
    )

with col_title:
    st.markdown("""
    <div style="padding-top: 4px;">
        <div style="
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 800;
            line-height: 1.2;
        ">
            Data Mining ID3
        </div>
        <div style="
            color: #d4e5f7;
            font-size: 0.78rem;
            font-weight: 600;
            line-height: 1.3;
            margin-top: 4px;
        ">
            PT PLN ULP Samarinda Seberang
        </div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
# =============================

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

st.sidebar.markdown("### 📂 Upload Data CICO")
uploaded = st.sidebar.file_uploader("File laporan gangguan (.xlsx)", type=["xlsx"])

if uploaded is not None:
    st.session_state.uploaded_file = uploaded

uploaded_file = st.session_state.uploaded_file

if uploaded_file:
    st.sidebar.success("✅ File terupload")
else:
    st.sidebar.markdown("""
    <div style="
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.18);
        border-left: 5px solid #60a5fa;
        border-radius: 12px;
        padding: 14px 16px;
        color: #eaf4ff;
        font-size: 0.86rem;
        line-height: 1.6;
        margin-top: 10px;
    ">
        <div style="font-weight: 800; color: #ffffff; margin-bottom: 4px;">
            ⬆️ Upload Data
        </div>
        <div style="color: #dbeafe;">
            Upload file CICO PLN untuk memulai analisis.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

def clean_columns(df):
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df

def insight_box(title, text, color="#2563eb"):
    st.markdown(f"""
    <div style="
        background:#f8fafc;
        border:1px solid #e2e8f0;
        border-left:5px solid {color};
        border-radius:12px;
        padding:14px 16px;
        margin:10px 0 16px 0;
        color:#1a202c;
        font-size:0.88rem;
        line-height:1.7;
    ">
        <div style="font-weight:800;color:#0f2540;margin-bottom:6px;">
            💡 {title}
        </div>
        <div>{text}</div>
    </div>
    """, unsafe_allow_html=True)


def buat_insight_distribusi(total, n_berat, n_ringan):
    persen_berat = (n_berat / total) * 100 if total else 0
    persen_ringan = (n_ringan / total) * 100 if total else 0

    kelas_dominan = "Berat" if n_berat > n_ringan else "Ringan"
    jumlah_dominan = max(n_berat, n_ringan)
    persen_dominan = max(persen_berat, persen_ringan)

    kondisi = "tidak seimbang" if abs(persen_berat - persen_ringan) >= 30 else "cukup seimbang"

    return f"""
    Data final berjumlah <b>{total}</b> data, terdiri dari <b>{n_berat}</b> kelas Berat
    ({persen_berat:.2f}%) dan <b>{n_ringan}</b> kelas Ringan ({persen_ringan:.2f}%).
    Kelas yang paling dominan adalah <b>{kelas_dominan}</b> sebanyak <b>{jumlah_dominan}</b> data
    ({persen_dominan:.2f}%). Kondisi ini menunjukkan bahwa distribusi kelas pada data final
    <b>{kondisi}</b>. Ketidakseimbangan ini perlu diperhatikan karena model dapat lebih mudah
    mengenali kelas yang jumlah datanya lebih banyak, sehingga dapat memengaruhi hasil evaluasi model.
    """

def buat_insight_distribusi_umum(distribusi_umum, fitur_distribusi):
    df_tmp = distribusi_umum.copy()

    if df_tmp.empty:
        return (
            f"Tabel distribusi untuk atribut <b>{fitur_distribusi}</b> belum memiliki data "
            f"yang dapat ditampilkan."
        )

    if df_tmp["Persentase"].dtype == object:
        df_tmp["Persentase"] = (
            df_tmp["Persentase"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .astype(float)
        )

    top = df_tmp.sort_values("Jumlah", ascending=False).iloc[0]

    nilai_terbanyak = top["Nilai Atribut"]
    jumlah_terbanyak = int(top["Jumlah"])
    persen_terbanyak = float(top["Persentase"])

    if persen_terbanyak >= 50:
        kalimat_awal = (
            f"Lebih dari separuh total laporan (<b>{persen_terbanyak:.2f}%</b> atau "
            f"<b>{jumlah_terbanyak}</b> data) tercatat pada nilai <b>{nilai_terbanyak}</b> "
            f"untuk atribut <b>{fitur_distribusi}</b>."
        )
    else:
        kalimat_awal = (
            f"Sebanyak <b>{persen_terbanyak:.2f}%</b> atau <b>{jumlah_terbanyak}</b> data "
            f"tercatat pada nilai <b>{nilai_terbanyak}</b> untuk atribut "
            f"<b>{fitur_distribusi}</b>."
        )

    return (
        f"{kalimat_awal} "
        f"Jumlah ini menunjukkan bahwa nilai <b>{nilai_terbanyak}</b> cukup banyak muncul "
        f"pada data gangguan. Karena jumlahnya dominan, nilai bisa berpengaruh "
        f"saat data digunakan untuk menentukan gangguan Ringan atau Berat."
    )

def buat_insight_gain(df_gains):
    top = df_gains.iloc[0]
    bottom = df_gains.iloc[-1]

    if len(df_gains) > 1:
        second = df_gains.iloc[1]
        pembanding = f" Atribut berikutnya adalah <b>{second['Atribut']}</b> dengan nilai gain <b>{second['Information Gain']:.3f}</b>."
    else:
        pembanding = ""

    return f"""
    Berdasarkan hasil perhitungan information gain, atribut <b>{top['Atribut']}</b>
    memiliki nilai gain tertinggi sebesar <b>{top['Information Gain']:.3f}</b>.
    Artinya, atribut tersebut paling baik dalam memisahkan data latih ke dalam kelas
    Ringan dan Berat pada tahap awal pembentukan pohon keputusan.{pembanding}
    Sementara itu, atribut <b>{bottom['Atribut']}</b> memiliki nilai gain terendah sebesar
    <b>{bottom['Information Gain']:.3f}</b>, sehingga pengaruhnya terhadap pemisahan kelas
    pada tahap ini relatif lebih kecil.
    """


def buat_insight_detail_atribut(df_det, attr_sel, ig_s):
    df_tmp = df_det.copy()

    total_data = df_tmp["Jumlah"].sum()
    nilai_terbesar = df_tmp.sort_values("Jumlah", ascending=False).iloc[0]
    entropy_tertinggi = df_tmp.sort_values("Entropy", ascending=False).iloc[0]

    nilai_murni = df_tmp[df_tmp["Entropy"].round(3) == 0]["Nilai"].astype(str).tolist()

    if len(nilai_murni) > 0:
        teks_murni = (
            f" Terdapat nilai atribut yang sudah homogen, contohnya "
            f"<b>{', '.join(nilai_murni[:3])}</b>. Entropy bernilai 0 berarti data pada nilai atribut tersebut "
            f"sudah mengarah pada satu kelas, sehingga tidak perlu dipisahkan lagi."
        )
    else:
        teks_murni = (
            " Pada atribut ini belum terdapat nilai atribut yang sepenuhnya homogen, "
            "sehingga beberapa bagian data masih memiliki campuran kelas Berat dan Ringan."
        )

    return f"""
    Tabel ini menampilkan perhitungan entropy untuk setiap nilai pada atribut <b>{attr_sel}</b>.
    Kolom <b>Jumlah</b> menunjukkan banyaknya data pada masing-masing nilai atribut,
    sedangkan <b>Proporsi Nilai Atribut</b> menunjukkan bagian data pada nilai atribut tersebut
    terhadap total data yang sedang dihitung.

    Pada atribut <b>{attr_sel}</b>, nilai atribut dengan jumlah data terbanyak adalah
    <b>{nilai_terbesar['Nilai']}</b> sebanyak <b>{nilai_terbesar['Jumlah']}</b> data
    dengan proporsi <b>{nilai_terbesar['Proporsi']:.3f}</b>
    atau <b>{nilai_terbesar['Proporsi'] * 100:.2f}%</b>.
    Proporsi ini digunakan sebagai bobot dalam perhitungan information gain.

    Nilai entropy tertinggi terdapat pada <b>{entropy_tertinggi['Nilai']}</b>
    sebesar <b>{entropy_tertinggi['Entropy']:.3f}</b>. Artinya, data pada nilai atribut tersebut
    masih memiliki keragaman kelas, sehingga belum langsung menjadi keputusan akhir.
    {teks_murni}

    Information gain atribut <b>{attr_sel}</b> sebesar <b>{ig_s:.3f}</b>.
    Nilai ini menunjukkan seberapa besar atribut <b>{attr_sel}</b> mengurangi
    keragaman kelas setelah data dipisahkan berdasarkan nilai atributnya.
    """

def buat_insight_crosstab(ct_table, fitur_crosstab):
    df_tmp = ct_table.copy()

    if df_tmp.empty or len(df_tmp.columns) == 0:
        return (
            f"Tabel crosstab untuk fitur <b>{fitur_crosstab}</b> belum memiliki data "
            f"yang dapat ditampilkan."
        )

    if "Nilai Atribut" not in df_tmp.columns:
        df_tmp = df_tmp.rename(columns={df_tmp.columns[0]: "Nilai Atribut"})

    if "Berat" not in df_tmp.columns:
        df_tmp["Berat"] = 0
    if "Ringan" not in df_tmp.columns:
        df_tmp["Ringan"] = 0

    df_tmp["Berat"] = df_tmp["Berat"].fillna(0).astype(int)
    df_tmp["Ringan"] = df_tmp["Ringan"].fillna(0).astype(int)
    df_tmp["Total Data"] = df_tmp["Berat"] + df_tmp["Ringan"]

    df_campur = df_tmp[
        (df_tmp["Berat"] > 0) &
        (df_tmp["Ringan"] > 0)
    ].copy()

    df_campur = df_campur.sort_values("Total Data", ascending=False).head(3)

    def gabung_daftar(daftar, penghubung="dan"):
        if len(daftar) == 1:
            return daftar[0]
        elif len(daftar) == 2:
            return f"{daftar[0]} {penghubung} {daftar[1]}"
        else:
            return f"{', '.join(daftar[:-1])}, {penghubung} {daftar[-1]}"

    if not df_campur.empty:
        daftar_campur = []

        for _, r in df_campur.iterrows():
            nilai = r["Nilai Atribut"]
            berat = int(r["Berat"])
            ringan = int(r["Ringan"])

            daftar_campur.append(
                f"<b>{nilai}</b> (<b>{berat}</b> Berat, <b>{ringan}</b> Ringan)"
            )

        if len(daftar_campur) == 1:
            kalimat_nilai = f"nilai {gabung_daftar(daftar_campur)}"
        else:
            kalimat_nilai = f"beberapa nilai seperti {gabung_daftar(daftar_campur)}"

        return (
            f"Pada fitur <b>{fitur_crosstab}</b>, {kalimat_nilai} masih muncul "
            f"pada target Berat dan Ringan. Hal ini menunjukkan bahwa kondisi tersebut "
            f"pernah muncul pada dua jenis penanganan, sehingga nilai fitur ini belum cukup kuat "
            f"untuk langsung menentukan gangguan Berat atau Ringan. Oleh karena itu, fitur "
            f"<b>{fitur_crosstab}</b> tetap perlu dibaca bersama fitur lain agar hasil klasifikasi "
            f"gangguan lebih tepat."
        )

    return (
        f"Pada fitur <b>{fitur_crosstab}</b>, tidak ada nilai yang muncul pada dua target. "
        f"Artinya, fitur ini sudah dapat membedakan gangguan Berat dan Ringan pada data "
        f"yang ditampilkan."
    )

def ambil_kode_awal_regu(nama_regu):
    if pd.isna(nama_regu):
        return None

    teks = str(nama_regu).strip()
    angka = "".join(filter(str.isdigit, teks))

    if not angka:
        return None

    kode = angka[0]

    if kode in ["7", "8"]:
        return kode

    return None


def ambil_nilai_terbanyak(series):
    data = series.dropna().astype(str).str.strip()

    if data.empty:
        return "-"

    hitung = data.value_counts()
    nilai = hitung.index[0]
    jumlah = int(hitung.iloc[0])

    return f"{nilai} ({jumlah} data)"


def buat_ringkasan_atribut_nama_regu(df_regu, fitur_X):
    df_tmp = df_regu.copy()

    df_tmp = df_tmp.dropna(subset=["Nama Regu"])
    df_tmp["Nama Regu"] = df_tmp["Nama Regu"].astype(str).str.strip()

    df_tmp["Kode Regu"] = df_tmp["Nama Regu"].apply(ambil_kode_awal_regu)
    df_tmp = df_tmp[df_tmp["Kode Regu"].isin(["7", "8"])]

    hasil_tabel = []

    for nama_regu, group in df_tmp.groupby("Nama Regu"):
        total_data = len(group)

        kode_regu = group["Kode Regu"].mode().iloc[0]

        row_tabel = {
            "Kode Regu": kode_regu,
            "Nama Regu": nama_regu,
            "Jumlah Data": total_data
        }

        for fitur in fitur_X:
            row_tabel[f"{fitur} Terbanyak"] = ambil_nilai_terbanyak(
                group[fitur]
            )

        hasil_tabel.append(row_tabel)

    df_tabel = pd.DataFrame(hasil_tabel)

    if not df_tabel.empty:
        df_tabel["Urutan Kode"] = df_tabel["Kode Regu"].map({
            "7": 1,
            "8": 2
        })

        df_tabel = df_tabel.sort_values(
            ["Urutan Kode", "Jumlah Data"],
            ascending=[True, False]
        )

        df_tabel = df_tabel.drop(columns=["Urutan Kode"])

    return df_tabel

def format_tabel_detail_id3(df_detail):
    df_tampil = df_detail.copy()

    df_tampil = df_tampil.drop(
        columns=["Proporsi"],
        errors="ignore"
    )

    if "Entropy" in df_tampil.columns:
        df_tampil["Entropy"] = df_tampil["Entropy"].apply(
            lambda x: f"{x:.3f}"
        )

    return df_tampil

def cek_indikator_risiko(input_user):
    kata_risiko = [
        "kebakaran",
        "api",
        "meledak",
        "ledakan",
        "pohon",
        "tumbang",
        "dahan",
        "ranting",
        "binatang",
        "hewan",
        "burung",
        "ular",
        "kucing",
        "bencana",
        "musibah",
        "banjir",
        "petir"
    ]

    atribut_dicek = [
        "Dampak Kerusakan",
        "Penyebab",
        "Kelompok Penyebab"
    ]

    indikator_ditemukan = []

    for atribut in atribut_dicek:
        nilai = input_user.get(atribut, "")
        teks_nilai = str(nilai).lower()

        for kata in kata_risiko:
            if kata in teks_nilai:
                indikator_ditemukan.append(
                    f"{atribut}: {nilai}"
                )
                break

    return indikator_ditemukan

# =============================
# MENU NAVIGASI
# =============================
st.sidebar.markdown("### 🧭 Navigasi")

menu = st.sidebar.radio(
    "Pilih Menu",
    ["📚 Tahapan", "⚙️ Klasifikasi Gangguan"],
    label_visibility="collapsed"
)

# =============================
# CONTENT
# =============================
if menu == "📚 Tahapan":

    tahap = st.sidebar.radio(
        "Pilih Tahapan",
        [
            "📌 Business Understanding",
            "1. Data Acquisition",
            "2. Data Preparation",
            "3. Exploratory Data Analysis",
            "4. Modeling",
            "5. Evaluation"
        ]
    )
    
    # =============================
    # SHOW / HIDE SIDEBAR SECTION
    # =============================
    SHOW_STATUS_TAHAPAN = False

    if SHOW_STATUS_TAHAPAN:
        steps_done = {
            "1": "df_final" in st.session_state or "train_df" in st.session_state,
            "2": "df_final" in st.session_state,
            "3": "train_df" in st.session_state,
            "4": "model_tree" in st.session_state,
            "5": "model_tree" in st.session_state,
        }

        status_icons = {
            True: "🟢",
            False: "🔴"
        }

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Status Tahapan**")

        for k, label in [
            ("1", "Data Acquisition"),
            ("2", "Data Preparation"),
            ("3", "EDA"),
            ("4", "Modeling"),
            ("5", "Evaluation")
        ]:
            st.sidebar.markdown(
                f"{status_icons[steps_done[k]]} {k}. {label}"
            )

    # =========================================================
    # 📌 BUSINESS UNDERSTANDING
    # =========================================================
    if tahap == "📌 Business Understanding":
        st.title("📌 Business Understanding")
        st.caption("Pemahaman konteks bisnis, permasalahan, kebutuhan solusi, tujuan penelitian, variabel penelitian, alur tahapan, dan output penelitian")
        st.markdown("---")

        # =============================
        # BANNER IDENTITAS PENELITIAN
        # =============================
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #0f2540 0%, #1a3a5c 60%, #1e4976 100%);
            border-radius: 16px;
            padding: 28px 32px;
            margin-bottom: 28px;
            box-shadow: 0 4px 16px rgba(15, 37, 64, 0.18);
        ">
            <div style="
                color: #bfdbfe;
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                margin-bottom: 10px;
            ">Tugas Akhir — Politeknik Negeri Samarinda</div>
            <div style="
                color: #ffffff;
                font-size: 1.22rem;
                font-weight: 800;
                line-height: 1.5;
                margin-bottom: 14px;
            ">
                Klasifikasi Tingkat Gangguan Menggunakan Metode Decision Tree
                Sebagai Dasar Penentuan Tim Petugas Pada PT PLN ULP Samarinda Seberang
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                <span style="background:rgba(255,255,255,0.12);color:#bfdbfe;padding:5px 14px;border-radius:999px;font-size:0.82rem;font-weight:600;">
                    🏢 PT PLN ULP Samarinda Seberang
                </span>
                <span style="background:rgba(255,255,255,0.12);color:#bfdbfe;padding:5px 14px;border-radius:999px;font-size:0.82rem;font-weight:600;">
                    🌳 Algoritma ID3 (Decision Tree)
                </span>
                <span style="background:rgba(255,255,255,0.12);color:#bfdbfe;padding:5px 14px;border-radius:999px;font-size:0.82rem;font-weight:600;">
                    🗂️ Data CICO PLN — 17–19 Juni 2025
                </span>
                <span style="background:rgba(255,255,255,0.12);color:#bfdbfe;padding:5px 14px;border-radius:999px;font-size:0.82rem;font-weight:600;">
                    🎓 Program Studi D-III Teknik Informatika
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # =============================
        # BAGIAN 1 — KONTEKS BISNIS & PERMASALAHAN
        # =============================
        st.markdown("## 🏢 Konteks Bisnis & Permasalahan")

        col_l, col_r = st.columns([1.05, 1])

        with col_l:
            st.markdown("""
            <div style="
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-left: 6px solid #2563eb;
                border-radius: 14px;
                padding: 22px 24px;
                margin-bottom: 16px;
                box-shadow: 0 2px 8px rgba(15,23,42,0.05);
            ">
                <div style="font-size:0.75rem;color:#64748b;font-weight:800;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">
                    Konteks Operasional
                </div>
                <div style="color:#1a202c;font-size:0.93rem;line-height:1.75;">
                    PT PLN Unit Layanan Pelanggan (ULP) Samarinda Seberang menerima laporan gangguan listrik dari pelanggan melalui <em>Command Center</em>. Setiap laporan perlu dialokasikan kepada tim teknis yang sesuai — tim <strong>motor</strong> untuk gangguan ringan, atau tim <strong>mobil</strong> untuk gangguan berat.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="
                background: #fff7ed;
                border: 1px solid #fed7aa;
                border-left: 6px solid #f97316;
                border-radius: 14px;
                padding: 22px 24px;
                box-shadow: 0 2px 8px rgba(15,23,42,0.05);
            ">
                <div style="font-size:0.75rem;color:#9a3412;font-weight:800;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">
                    ⚠️ Permasalahan yang Diidentifikasi
                </div>
                <div style="display:grid;gap:8px;">
                    <div style="background:#ffffff;border:1px solid #fed7aa;border-radius:10px;padding:10px 14px;display:flex;gap:10px;align-items:flex-start;">
                        <span style="font-size:1rem;min-width:22px;">❌</span>
                        <span style="color:#1a202c;font-size:0.88rem;line-height:1.55;">
                            <strong>Proses pemeriksaan manual</strong> — setiap laporan gangguan masih perlu ditinjau terlebih dahulu oleh Command Center sebelum dialokasikan ke tim teknis.
                        </span>
                    </div>
                    <div style="background:#ffffff;border:1px solid #fed7aa;border-radius:10px;padding:10px 14px;display:flex;gap:10px;align-items:flex-start;">
                        <span style="font-size:1rem;min-width:22px;">❌</span>
                        <span style="color:#1a202c;font-size:0.88rem;line-height:1.55;">
                            <strong>Kesalahan penugasan</strong> — laporan gangguan berpotensi dialokasikan kepada tim yang kurang sesuai.
                        </span>
                    </div>
                    <div style="background:#ffffff;border:1px solid #fed7aa;border-radius:10px;padding:10px 14px;display:flex;gap:10px;align-items:flex-start;">
                        <span style="font-size:1rem;min-width:22px;">❌</span>
                        <span style="color:#1a202c;font-size:0.88rem;line-height:1.55;">
                            <strong>Belum berbasis data historis</strong> — proses alokasi belum menggunakan pendekatan klasifikasi berbasis data historis untuk membantu menentukan tingkat gangguan.
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_r:
            st.markdown("""
                <div style="background:#ffffff;border:1px solid #e2e8f0;border-left:6px solid #16a34a;border-radius:14px;padding:22px 24px;height:100%;box-shadow:0 2px 8px rgba(15,23,42,0.05);">
                <div style="font-size:0.75rem;color:#64748b;font-weight:800;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:14px;">Alur Penanganan Gangguan Saat Ini</div>
                <div style="display:grid;gap:10px;">
                <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;padding:12px 14px;display:flex;gap:12px;align-items:center;">
                <div style="min-width:32px;height:32px;background:#0284c7;color:white;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:0.85rem;flex-shrink:0;">1</div>
                <div><div style="font-weight:700;color:#0f2540;font-size:0.88rem;">Laporan Masuk</div>
                <div style="color:#64748b;font-size:0.80rem;line-height:1.4;">Pelanggan melapor melalui call center atau PLN Mobile</div></div>
                </div>
                <div style="text-align:center;color:#94a3b8;font-size:1.1rem;font-weight:700;">↓</div>
                <div style="background:#fef9c3;border:1px solid #fde68a;border-radius:10px;padding:12px 14px;display:flex;gap:12px;align-items:center;">
                <div style="min-width:32px;height:32px;background:#ca8a04;color:white;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:0.85rem;flex-shrink:0;">2</div>
                <div><div style="font-weight:700;color:#0f2540;font-size:0.88rem;">Command Center <span style="background:#fbbf24;color:#78350f;padding:2px 8px;border-radius:999px;font-size:0.72rem;font-weight:800;">Manual</span></div>
                <div style="color:#64748b;font-size:0.80rem;line-height:1.4;">Laporan ditinjau secara manual sebelum dialokasikan ke tim motor atau tim mobil</div></div>
                </div>
                <div style="text-align:center;color:#94a3b8;font-size:1.1rem;font-weight:700;">↓</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:10px 12px;text-align:center;">
                <div style="font-size:1.3rem;">🛵</div>
                <div style="font-weight:700;color:#166534;font-size:0.82rem;margin-top:4px;">Tim Motor</div>
                <div style="color:#64748b;font-size:0.75rem;">Gangguan Ringan</div>
                </div>
                <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:10px 12px;text-align:center;">
                <div style="font-size:1.3rem;">🛻</div>
                <div style="font-weight:700;color:#991b1b;font-size:0.82rem;margin-top:4px;">Tim Mobil</div>
                <div style="color:#64748b;font-size:0.75rem;">Gangguan Berat</div>
                </div>
                </div>
                </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # =============================
        # BAGIAN 2 — KEBUTUHAN SOLUSI & PENDEKATAN
        # =============================
        st.markdown("## 💡 Kebutuhan Solusi & Pendekatan")

        st.markdown("""
        <div style="
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 18px;
            box-shadow: 0 1px 4px rgba(15,23,42,0.05);
        ">
            <div style="font-size:0.75rem;color:#0369a1;font-weight:800;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">
                Kebutuhan Solusi
            </div>
            <div style="color:#1a202c;font-size:0.93rem;line-height:1.75;">
                Diperlukan suatu sistem berbasis data yang mampu mengklasifikasikan tingkat gangguan listrik berdasarkan atribut yang tercatat pada laporan gangguan. Sistem ini diharapkan dapat menjadi <strong>dasar pendukung keputusan</strong> bagi <em>Command Center</em> dalam proses alokasi tim teknis, sehingga proses penentuan tim dapat dilakukan secara lebih terstruktur dan berbasis data..
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_sol1, col_sol2, col_sol3 = st.columns(3)

        with col_sol1:
            st.markdown("""
            <div style="
                background:#ffffff;border:1px solid #e2e8f0;border-top:5px solid #2563eb;
                border-radius:14px;padding:18px 16px;height:180px;
                box-shadow:0 2px 8px rgba(15,23,42,0.05);
            ">
                <div style="font-size:1.6rem;margin-bottom:8px;">🌳</div>
                <div style="font-weight:800;color:#1e3a5f;font-size:0.9rem;margin-bottom:6px;">Metode</div>
                <div style="font-size:1.1rem;font-weight:800;color:#2563eb;margin-bottom:4px;">Decision Tree</div>
                <div style="color:#64748b;font-size:0.82rem;line-height:1.5;">
                    Metode klasifikasi yang menghasilkan aturan keputusan yang mudah dipahami dan dijelaskan.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_sol2:
            st.markdown("""
            <div style="
                background:#ffffff;border:1px solid #e2e8f0;border-top:5px solid #7c3aed;
                border-radius:14px;padding:18px 16px;height:180px;
                box-shadow:0 2px 8px rgba(15,23,42,0.05);
            ">
                <div style="font-size:1.6rem;margin-bottom:8px;">⚙️</div>
                <div style="font-weight:800;color:#1e3a5f;font-size:0.9rem;margin-bottom:6px;">Algoritma</div>
                <div style="font-size:1.1rem;font-weight:800;color:#7c3aed;margin-bottom:4px;">ID3 (Iterative Dichotomiser 3)</div>
                <div style="color:#64748b;font-size:0.82rem;line-height:1.5;">
                    Algoritma berbasis entropy dan information gain untuk data kategorikal.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_sol3:
            st.markdown("""
            <div style="
                background:#ffffff;border:1px solid #e2e8f0;border-top:5px solid #16a34a;
                border-radius:14px;padding:18px 16px;height:180px;
                box-shadow:0 2px 8px rgba(15,23,42,0.05);
            ">
                <div style="font-size:1.6rem;margin-bottom:8px;">🗂️</div>
                <div style="font-weight:800;color:#1e3a5f;font-size:0.9rem;margin-bottom:6px;">Sumber Data</div>
                <div style="font-size:1.1rem;font-weight:800;color:#16a34a;margin-bottom:4px;">Data Historis CICO PLN</div>
                <div style="color:#64748b;font-size:0.82rem;line-height:1.5;">
                    Laporan gangguan tercatat dari Check-In Check-Out PLN ULP Samarinda Seberang.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # =============================
        # BAGIAN 3 — TUJUAN Penelitian
        # =============================
        st.markdown("## 🎯 Tujuan Penelitian")

        st.markdown("""
        <div style="
            background:#ffffff;border:1px solid #e2e8f0;border-top:5px solid #2563eb;
            border-radius:14px;padding:20px 22px;
            box-shadow:0 2px 8px rgba(15,23,42,0.05);
        ">
            <div style="font-size:0.75rem;color:#64748b;font-weight:800;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:14px;">
                🎯 Tujuan Penelitian
            </div>
            <div style="display:grid;gap:10px;">
                <div style="display:flex;gap:10px;align-items:flex-start;">
                    <div style="min-width:26px;height:26px;background:#dbeafe;color:#1d4ed8;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:0.78rem;">1</div>
                    <div style="color:#1a202c;font-size:0.88rem;line-height:1.6;">Membangun dan mengevaluasi model klasifikasi tingkat gangguan listrik menggunakan metode Decision Tree dengan algoritma ID3 berdasarkan data historis di PT PLN ULP Samarinda Seberang.</div>
                </div>
                <div style="display:flex;gap:10px;align-items:flex-start;">
                    <div style="min-width:26px;height:26px;background:#dbeafe;color:#1d4ed8;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:0.78rem;">2</div>
                    <div style="color:#1a202c;font-size:0.88rem;line-height:1.6;">Mengidentifikasi atribut yang paling berpengaruh dalam menentukan tingkat gangguan berdasarkan struktur pohon keputusan yang terbentuk.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # =============================
        # BAGIAN 4 — VARIABEL PENELITIAN
        # =============================
        st.markdown("## 📊 Variabel Penelitian")

        col_var_l, col_var_r = st.columns([1.1, 1])

        with col_var_l:
            st.markdown("""
            <div style="
                background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
                padding:22px 24px;box-shadow:0 2px 8px rgba(15,23,42,0.05);
            ">
                <div style="font-size:0.75rem;color:#64748b;font-weight:800;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:14px;">
                    Variabel Independen (Fitur Input — X)
                </div>
                <div style="display:grid;gap:8px;">
                    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:10px 14px;display:flex;gap:10px;align-items:center;">
                        <span style="background:#2563eb;color:white;border-radius:6px;padding:2px 8px;font-size:0.75rem;font-weight:800;min-width:24px;text-align:center;">X1</span>
                        <div>
                            <div style="font-weight:700;color:#1e3a5f;font-size:0.88rem;">Fasilitas</div>
                            <div style="color:#64748b;font-size:0.78rem;">Jenis fasilitas kelistrikan yang terdampak gangguan</div>
                        </div>
                    </div>
                    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:10px 14px;display:flex;gap:10px;align-items:center;">
                        <span style="background:#2563eb;color:white;border-radius:6px;padding:2px 8px;font-size:0.75rem;font-weight:800;min-width:24px;text-align:center;">X2</span>
                        <div>
                            <div style="font-weight:700;color:#1e3a5f;font-size:0.88rem;">Peralatan</div>
                            <div style="color:#64748b;font-size:0.78rem;">Peralatan yang mengalami gangguan atau kerusakan</div>
                        </div>
                    </div>
                    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:10px 14px;display:flex;gap:10px;align-items:center;">
                        <span style="background:#2563eb;color:white;border-radius:6px;padding:2px 8px;font-size:0.75rem;font-weight:800;min-width:24px;text-align:center;">X3</span>
                        <div>
                            <div style="font-weight:700;color:#1e3a5f;font-size:0.88rem;">Dampak Kerusakan</div>
                            <div style="color:#64748b;font-size:0.78rem;">Jenis kerusakan yang ditimbulkan akibat gangguan</div>
                        </div>
                    </div>
                    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:10px 14px;display:flex;gap:10px;align-items:center;">
                        <span style="background:#2563eb;color:white;border-radius:6px;padding:2px 8px;font-size:0.75rem;font-weight:800;min-width:24px;text-align:center;">X4</span>
                        <div>
                            <div style="font-weight:700;color:#1e3a5f;font-size:0.88rem;">Penyebab</div>
                            <div style="color:#64748b;font-size:0.78rem;">Faktor penyebab utama terjadinya gangguan listrik</div>
                        </div>
                    </div>
                    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:10px 14px;display:flex;gap:10px;align-items:center;">
                        <span style="background:#2563eb;color:white;border-radius:6px;padding:2px 8px;font-size:0.75rem;font-weight:800;min-width:24px;text-align:center;">X5</span>
                        <div>
                            <div style="font-weight:700;color:#1e3a5f;font-size:0.88rem;">Kelompok Penyebab</div>
                            <div style="color:#64748b;font-size:0.78rem;">Pengelompokan kategori penyebab gangguan</div>
                        </div>
                    </div>
                    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:10px 14px;display:flex;gap:10px;align-items:center;">
                        <span style="background:#2563eb;color:white;border-radius:6px;padding:2px 8px;font-size:0.75rem;font-weight:800;min-width:24px;text-align:center;">X6</span>
                        <div>
                            <div style="font-weight:700;color:#1e3a5f;font-size:0.88rem;">Cuaca</div>
                            <div style="color:#64748b;font-size:0.78rem;">Kondisi cuaca saat gangguan listrik terjadi</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_var_r:
            st.markdown("""
            <div style="
                background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
                padding:22px 24px;box-shadow:0 2px 8px rgba(15,23,42,0.05);
                margin-bottom:14px;
            ">
                <div style="font-size:0.75rem;color:#64748b;font-weight:800;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:14px;">
                    Variabel Dependen (Target Klasifikasi — Y)
                </div>
                <div style="display:grid;gap:10px;">
                    <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:12px;padding:16px;text-align:center;">
                        <div style="font-size:1.8rem;margin-bottom:6px;">🚨</div>
                        <div style="font-weight:800;color:#991b1b;font-size:1rem;">BERAT</div>
                        <div style="color:#64748b;font-size:0.82rem;margin-top:4px;line-height:1.5;">
                            Ditangani tim <strong>mobil</strong>.<br>
                            Kode regu diawali angka <strong>8</strong>.
                        </div>
                    </div>
                    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;padding:16px;text-align:center;">
                        <div style="font-size:1.8rem;margin-bottom:6px;">✅</div>
                        <div style="font-weight:800;color:#166534;font-size:1rem;">RINGAN</div>
                        <div style="color:#64748b;font-size:0.82rem;margin-top:4px;line-height:1.5;">
                            Ditangani tim <strong>motor</strong>.<br>
                            Kode regu diawali angka <strong>7</strong>.
                        </div>
                    </div>
                </div>
            </div>

            <div style="
                background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
                padding:16px;box-shadow:0 1px 4px rgba(15,23,42,0.04);
            ">
                <div style="font-size:0.75rem;color:#64748b;font-weight:800;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">
                    Metrik Evaluasi Model
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:8px 10px;font-size:0.82rem;color:#334155;font-weight:600;">🔢 Confusion Matrix</div>
                    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:8px 10px;font-size:0.82rem;color:#334155;font-weight:600;">📊 Akurasi</div>
                    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:8px 10px;font-size:0.82rem;color:#334155;font-weight:600;">🎯 Presisi</div>
                    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:8px 10px;font-size:0.82rem;color:#334155;font-weight:600;">📌 Recall</div>
                    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:8px 10px;font-size:0.82rem;color:#334155;font-weight:600;">🏅 F1-Score</div>
                    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:8px 10px;font-size:0.82rem;color:#334155;font-weight:600;">🔁 Stratified 5-Fold Cross Validation</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # =============================
        # BAGIAN 5 — OUTPUT SISTEM & ALUR TAHAPAN
        # =============================
        st.markdown("## 📦 Alur Tahapan Penelitian & Output Penelitian")

        col_steps, col_out = st.columns([1.1, 1])

        with col_steps:
            steps_data = [
                ("1", "#2563eb", "Data Acquisition",
                "Mengumpulkan data historis laporan gangguan (CICO) dari PT PLN ULP Samarinda Seberang."),
                ("2", "#7c3aed", "Data Preparation",
                "Cleaning data, pelabelan variabel target, seleksi fitur, case folding, dan stratified split 80:20."),
                ("3", "#0891b2", "Exploratory Data Analysis",
                "Mengeksplorasi distribusi target, distribusi atribut, hubungan fitur dengan target, dan ringkasan atribut berdasarkan regu sebagai pemahaman awal sebelum tahap modeling."),
                ("4", "#16a34a", "Modeling (ID3)",
                "Pembentukan pohon keputusan menggunakan algoritma ID3 berbasis entropy dan information gain."),
                ("5", "#dc2626", "Evaluation",
                "Pengujian model dengan confusion matrix, akurasi, presisi, recall, F1-score, dan 5-fold cross validation."),
            ]

            # Bangun HTML lengkap dalam satu variabel Python,
            # lalu panggil st.markdown() SATU KALI saja.
            # Jangan pernah split <div> buka/tutup di dua st.markdown() berbeda.
            html_steps = []
            html_steps.append('<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;padding:20px 22px;box-shadow:0 2px 8px rgba(15,23,42,0.05);">')
            html_steps.append('<div style="font-size:0.75rem;color:#64748b;font-weight:800;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:14px;">Alur Tahapan Penelitian</div>')

            for idx, (num, color, title, desc) in enumerate(steps_data):
                is_last = idx == len(steps_data) - 1
                mb = "0" if is_last else "10px"
                pb = "0" if is_last else "10px"
                html_steps.append(
                    f'<div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:{mb};">'
                    f'<div style="min-width:30px;height:30px;background:{color};color:white;border-radius:50%;'
                    f'display:flex;align-items:center;justify-content:center;font-weight:800;font-size:0.85rem;'
                    f'flex-shrink:0;box-shadow:0 2px 6px rgba(0,0,0,0.15);">{num}</div>'
                    f'<div style="border-left:2px dashed #e2e8f0;padding-left:12px;padding-bottom:{pb};width:100%;">'
                    f'<div style="font-weight:700;color:#0f2540;font-size:0.88rem;">{title}</div>'
                    f'<div style="color:#64748b;font-size:0.80rem;line-height:1.5;margin-top:3px;">{desc}</div>'
                    f'</div></div>'
                )

            html_steps.append('</div>')

            st.markdown("".join(html_steps), unsafe_allow_html=True)

        with col_out:
            st.markdown("""
            <div style="
                background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
                padding:20px 22px;box-shadow:0 2px 8px rgba(15,23,42,0.05);
            ">
                <div style="font-size:0.75rem;color:#64748b;font-weight:800;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:14px;">
                    Output yang Dihasilkan
                </div>
                <div style="display:grid;gap:9px;">
                    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:10px 14px;display:flex;gap:10px;align-items:center;">
                        <span style="font-size:1.1rem;">🏷️</span>
                        <div>
                            <div style="font-weight:700;color:#166534;font-size:0.86rem;">Klasifikasi Gangguan</div>
                            <div style="color:#64748b;font-size:0.78rem;">Menghasilkan prediksi tingkat gangguan, yaitu Ringan atau Berat.</div>
                        </div>
                    </div>
                    <div style="background:#fff1f2;border:1px solid #fecdd3;border-radius:10px;padding:10px 14px;display:flex;gap:10px;align-items:center;">
                        <span style="font-size:1.1rem;">📋</span>
                        <div>
                            <div style="font-weight:700;color:#9f1239;font-size:0.86rem;">Evaluasi Performa Model</div>
                            <div style="color:#64748b;font-size:0.78rem;">Menampilkan hasil evaluasi model menggunakan confusion matrix, akurasi, presisi, recall, F1-score, dan cross validation.</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # =============================
        # FOOTER RINGKASAN
        # =============================
        # st.markdown("<br>", unsafe_allow_html=True)
        # st.markdown("""
        # <div style="
        #     background: linear-gradient(135deg, #f0f9ff 0%, #f0fdf4 100%);
        #     border: 1px solid #bae6fd;
        #     border-radius: 14px;
        #     padding: 18px 24px;
        #     box-shadow: 0 1px 4px rgba(15,23,42,0.04);
        # ">
        #     <div style="font-size:0.75rem;color:#0369a1;font-weight:800;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">
        #         ℹ️ Catatan Penggunaan Aplikasi
        #     </div>
        #     <div style="color:#334155;font-size:0.88rem;line-height:1.75;">
        #         Untuk memulai analisis, unggah file data CICO PLN (<code>.xlsx</code>) melalui panel kiri, 
        #         lalu ikuti tahapan secara berurutan mulai dari <strong>Data Acquisition</strong> hingga <strong>Evaluation</strong>. 
        #         Setelah model terbentuk, fitur <strong>Klasifikasi Gangguan</strong> pada menu sebelah kiri dapat digunakan 
        #         untuk melakukan prediksi tingkat gangguan secara langsung.
        #     </div>
        # </div>
        # """, unsafe_allow_html=True)
        
    # =========================================================
    # 📥 DATA ACQUISITION
    # =========================================================
    elif tahap == "1. Data Acquisition":
        st.title("📥 Tahap 1 — Data Acquisition")
        st.caption("Mengumpulkan data laporan gangguan dari sistem CICO PT PLN ULP Samarinda Seberang")
        st.markdown("---")

        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file, header=8, skiprows=[9])
            df = clean_columns(df)
            df = df[df["No Laporan"].notna()]

            # =============================
            # DASHBOARD CARD DATA ACQUISITION
            # =============================
            total_baris_mentah = len(df)
            jumlah_kolom_asli = len(df.columns)
            sumber_data = "CICO PLN"
            periode_data = "17–19 Juni 2025"   # bisa diganti "Data CICO" kalau tidak mau hardcode

            components.html(f"""
            <div style="
                font-family: Arial, sans-serif;
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 14px;
                width: 100%;
                margin: 8px 0 18px 0;
            ">

                <div style="
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-top: 5px solid #2563eb;
                    border-radius: 14px;
                    padding: 18px 16px;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                ">
                    <div style="font-size: 1.7rem;">📄</div>
                    <div style="
                        font-size: 0.76rem;
                        color: #64748b;
                        font-weight: 800;
                        text-transform: uppercase;
                        margin-top: 8px;
                        letter-spacing: 0.04em;
                    ">
                        Data Mentah
                    </div>
                    <div style="
                        font-size: 2rem;
                        font-weight: 900;
                        color: #2563eb;
                        margin-top: 4px;
                    ">
                        {total_baris_mentah}
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8;">
                        total baris laporan
                    </div>
                </div>

                <div style="
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-top: 5px solid #16a34a;
                    border-radius: 14px;
                    padding: 18px 16px;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                ">
                    <div style="font-size: 1.7rem;">📊</div>
                    <div style="
                        font-size: 0.76rem;
                        color: #64748b;
                        font-weight: 800;
                        text-transform: uppercase;
                        margin-top: 8px;
                        letter-spacing: 0.04em;
                    ">
                        Kolom
                    </div>
                    <div style="
                        font-size: 2rem;
                        font-weight: 900;
                        color: #16a34a;
                        margin-top: 4px;
                    ">
                        {jumlah_kolom_asli}
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8;">
                        Jumlah Kolom
                    </div>
                </div>

                <div style="
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-top: 5px solid #7c3aed;
                    border-radius: 14px;
                    padding: 18px 16px;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                ">
                    <div style="font-size: 1.7rem;">🏢</div>
                    <div style="
                        font-size: 0.76rem;
                        color: #64748b;
                        font-weight: 800;
                        text-transform: uppercase;
                        margin-top: 8px;
                        letter-spacing: 0.04em;
                    ">
                        Sumber Data
                    </div>
                    <div style="
                        font-size: 1.55rem;
                        font-weight: 900;
                        color: #7c3aed;
                        margin-top: 8px;
                    ">
                        {sumber_data}
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8;">
                        laporan gangguan
                    </div>
                </div>

                <div style="
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-top: 5px solid #f97316;
                    border-radius: 14px;
                    padding: 18px 16px;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                ">
                    <div style="font-size: 1.7rem;">🗓️</div>
                    <div style="
                        font-size: 0.76rem;
                        color: #64748b;
                        font-weight: 800;
                        text-transform: uppercase;
                        margin-top: 8px;
                        letter-spacing: 0.04em;
                    ">
                        Periode Data
                    </div>
                    <div style="
                        font-size: 1.35rem;
                        font-weight: 900;
                        color: #f97316;
                        margin-top: 10px;
                    ">
                        {periode_data}
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8;">
                        file CICO
                    </div>
                </div>

            </div>
            """, height=180, scrolling=False)

            tab_raw, tab_info = st.tabs(["📄 Isi File Mentah", "ℹ️ Info Kolom"])

            with tab_raw:
                st.info("Tabel di bawah adalah isi file Excel mentah apa adanya — belum ada pelabelan, cleaning, atau pemilihan kolom. Semua pengolahan baru dilakukan di Tahap 2 — Data Preparation.")
                search_raw = st.text_input("🔍 Cari di seluruh kolom:", key="search_raw")
                df_show = df if not search_raw else df[
                    df.apply(
                        lambda r: r.astype(str).str.contains(
                            search_raw,
                            case=False,
                            na=False,
                            regex=False
                        ).any(),
                        axis=1
                    )
                ]
                st.dataframe(df_show, use_container_width=True, hide_index=True)
                st.caption(f"Menampilkan {len(df_show)} dari {len(df)} baris mentah | {len(df.columns)} kolom asli | File: {uploaded_file.name}")

            with tab_info:
                kolom = df.columns.tolist()

                keterangan_dict = {
                    "No Laporan": "Nomor unik laporan gangguan",
                    "Fasilitas": "Jenis fasilitas",
                    "Peralatan": "Peralatan terdampak",
                    "Dampak Kerusakan": "Dampak gangguan",
                    "Penyebab": "Penyebab gangguan",
                    "Kelompok Penyebab": "Kategori penyebab",
                    "Cuaca": "Kondisi cuaca",
                    "Nama Regu": "Kode regu (sumber label target)"
                }

                total_baris = len(df)

                def ambil_contoh_data(df, kolom, jumlah=3):
                    contoh = (
                        df[kolom]
                        .dropna()
                        .astype(str)
                        .str.strip()
                    )

                    contoh = contoh[contoh != ""]
                    contoh = contoh.drop_duplicates().head(jumlah).tolist()

                    if len(contoh) == 0:
                        return "—"

                    return ", ".join(contoh)

                df_info = pd.DataFrame({
                    "No": range(1, len(kolom) + 1),
                    "Nama Kolom": kolom,
                    # "Keterangan": [
                    #     keterangan_dict.get(col, "—")
                    #     for col in kolom
                    # ],
                    "Contoh Data": [
                        ambil_contoh_data(df, col, 3)
                        for col in kolom
                    ],
                    "Terisi": [
                        f"{df[col].notna().sum()}/{total_baris}"
                        for col in kolom
                    ]
                })

                st.dataframe(
                    df_info,
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.markdown("""
            <div style="background:#fefce8;border:1px dashed #ca8a04;border-radius:10px;
                        padding:2rem;text-align:center;color:#92400e;margin-bottom:1.5rem;">
                <div style="font-size:2rem;">📂</div>
                <div style="font-weight:700;font-size:1.1rem;margin:8px 0;">Silakan upload file CICO PLN terlebih dahulu melalui panel kiri.</div>
                <div style="font-size:0.87rem;">Format: <code>.xlsx</code> | Limit 200 MB</div>
            </div>""", unsafe_allow_html=True)

    # =========================================================
    # 🧹 DATA PREPARATION
    # =========================================================
    elif tahap == "2. Data Preparation":
        st.title("🧹 Tahap 2 — Data Preparation")
        st.caption("Membersihkan data, pelabelan berdasarkan kode regu, dan pembagian data latih-uji")
        st.markdown("---")

        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file, header=8, skiprows=[9])
            df = clean_columns(df)
            df = df[df["No Laporan"].notna()]

            prep = get_preparation_steps(df)

            df_before = prep["df_before"]
            df_no_dup = prep["df_no_dup"]
            df_clean_pre = prep["df_clean_pre"]
            df_final = prep["df_final"]
            fitur_X = prep["fitur_X"]
            target_Y = prep["target_Y"]

            train_df, test_df = split_data(df_final)

            st.session_state.train_df = train_df
            st.session_state.test_df = test_df
            st.session_state.df_final = df_final

            kolom_eda_regu = ["Nama Regu"] + FITUR_X + [TARGET_Y]

            kolom_eda_regu = [
                kolom for kolom in kolom_eda_regu
                if kolom in df_clean_pre.columns
            ]

            st.session_state.df_eda_regu = df_clean_pre[kolom_eda_regu].copy()

            # =============================
            # EXPORT HASIL SPLITTING KE EXCEL
            # =============================
            # train_df.reset_index(drop=True).to_excel(
            #     "data_latih_stratify.xlsx",
            #     index=False
            # )
            # test_df.reset_index(drop=True).to_excel(
            #     "data_uji_stratify.xlsx",
            #     index=False
            # )

            # =============================
            # DASHBOARD CARD DATA PREPARATION
            # =============================
            total_raw = len(df)
            total_duplicate = len(df) - len(df_no_dup)
            total_clean = len(df_no_dup)
            total_train = len(train_df)
            total_test = len(test_df)

            components.html(f"""
            <div style="
                font-family: Arial, sans-serif;
                display: grid;
                grid-template-columns: repeat(5, 1fr);
                gap: 14px;
                width: 100%;
                margin-bottom: 12px;
            ">

                <div style="
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-top: 5px solid #2563eb;
                    border-radius: 14px;
                    padding: 18px 16px;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                ">
                    <div style="font-size: 1.6rem;">📄</div>
                    <div style="font-size: 0.78rem; color: #64748b; font-weight: 700; text-transform: uppercase; margin-top: 8px;">
                        Data Mentah
                    </div>
                    <div style="font-size: 1.9rem; font-weight: 800; color: #0f2540; margin-top: 4px;">
                        {total_raw}
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8;">
                        record awal
                    </div>
                </div>

                <div style="
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-top: 5px solid #dc2626;
                    border-radius: 14px;
                    padding: 18px 16px;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                ">
                    <div style="font-size: 1.6rem;">🔁</div>
                    <div style="font-size: 0.78rem; color: #64748b; font-weight: 700; text-transform: uppercase; margin-top: 8px;">
                        Data Duplikat
                    </div>
                    <div style="font-size: 1.9rem; font-weight: 800; color: #dc2626; margin-top: 4px;">
                        {total_duplicate}
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8;">
                        dihapus
                    </div>
                </div>

                <div style="
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-top: 5px solid #16a34a;
                    border-radius: 14px;
                    padding: 18px 16px;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                ">
                    <div style="font-size: 1.6rem;">🧹</div>
                    <div style="font-size: 0.78rem; color: #64748b; font-weight: 700; text-transform: uppercase; margin-top: 8px;">
                        Setelah Cleaning
                    </div>
                    <div style="font-size: 1.9rem; font-weight: 800; color: #16a34a; margin-top: 4px;">
                        {total_clean}
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8;">
                        record bersih
                    </div>
                </div>

                <div style="
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-top: 5px solid #7c3aed;
                    border-radius: 14px;
                    padding: 18px 16px;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                ">
                    <div style="font-size: 1.6rem;">📘</div>
                    <div style="font-size: 0.78rem; color: #64748b; font-weight: 700; text-transform: uppercase; margin-top: 8px;">
                        Data Latih
                    </div>
                    <div style="font-size: 1.9rem; font-weight: 800; color: #7c3aed; margin-top: 4px;">
                        {total_train}
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8;">
                        80% dataset
                    </div>
                </div>

                <div style="
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-top: 5px solid #f97316;
                    border-radius: 14px;
                    padding: 18px 16px;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                ">
                    <div style="font-size: 1.6rem;">📕</div>
                    <div style="font-size: 0.78rem; color: #64748b; font-weight: 700; text-transform: uppercase; margin-top: 8px;">
                        Data Uji
                    </div>
                    <div style="font-size: 1.9rem; font-weight: 800; color: #f97316; margin-top: 4px;">
                        {total_test}
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8;">
                        20% dataset
                    </div>
                </div>

            </div>
            """, height=175, scrolling=False)

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📄 Data Sebelum Cleaning",
                "🧹 Data Sesudah Cleaning",
                "📊 Perbandingan Visual",
                "🏷️ Pelabelan",
                "📦 Data Final",
                "✂️ Split Data"
            ])

            with tab1:
                st.subheader("📄 Data Mentah CICO — Sebelum Cleaning")
                st.warning(f"Data ini mengandung **{len(df) - len(df_no_dup)} duplikat** dan belum dilabeli. Kolom Jenis Gangguan belum ada pada tahap ini.")
                search1 = st.text_input("🔍 Cari", key="s1")
                df_s1 = df if not search1 else df[
                    df.apply(
                        lambda r: r.astype(str).str.contains(
                            search1,
                            case=False,
                            na=False,
                            regex=False
                        ).any(),
                        axis=1
                    )
                ]
                st.dataframe(df_s1, use_container_width=True, hide_index=True)
                st.caption(f"Menampilkan {len(df_s1)} dari {len(df)} record")

            
            with tab2:

                st.subheader("🧹 Data Sesudah Cleaning")

                st.success(
                    f"""
                    Data berhasil dibersihkan dengan menghapus
                    {len(df) - len(df_no_dup)} data duplikat
                    berdasarkan kolom No Laporan.
                    """
                )
                
                # =============================
                # CONTOH DATA DUPLIKAT
                # =============================
                st.markdown("### 🔁 Contoh Data Duplikat")

                df_duplicate = df[
                    df.duplicated(
                        subset=["No Laporan"],
                        keep=False
                    )
                ]

                if not df_duplicate.empty:

                    st.info(
                        "Berikut contoh data dengan "
                        "No Laporan yang terdeteksi duplikat."
                    )

                    kolom_duplikat = [
                        "Nama Regu",
                        "No Laporan",
                        "Fasilitas",
                        "Peralatan",
                        "Dampak Kerusakan",
                        "Penyebab",
                        "Kelompok Penyebab",
                        "Cuaca"
                    ]

                    st.dataframe(
                        df_duplicate[kolom_duplikat].head(5),
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.success(
                        "Tidak ditemukan data duplikat."
                    )

                # col1, col2 = st.columns(2)

                # with col1:

                #     missing_clean = (
                #         df_no_dup.isnull()
                #         .sum()
                #         .reset_index()
                #     )

                #     missing_clean.columns = [
                #         "Kolom",
                #         "Jumlah Missing"
                #     ]

                #     st.markdown("### ⚠️ Missing Value")

                #     st.dataframe(
                #         missing_clean,
                #         use_container_width=True,
                #         hide_index=True
                #     )

                # with col2:

                #     st.markdown("### 📊 Visualisasi Missing Value")

                #     missing_vis = missing_clean[
                #         missing_clean["Jumlah Missing"] > 0
                #     ]

                #     if not missing_vis.empty:

                #         fig, ax = plt.subplots(figsize=(5,3.5))

                #         bars = ax.barh(
                #             missing_vis["Kolom"],
                #             missing_vis["Jumlah Missing"],
                #             color="#dc2626",
                #             edgecolor="white"
                #         )

                #         ax.bar_label(
                #             bars,
                #             fontsize=9,
                #             padding=3
                #         )

                #         ax.set_xlabel("Jumlah Missing")

                #         ax.set_title(
                #             "Jumlah Missing Value per Kolom",
                #             fontweight="bold"
                #         )

                #         plt.tight_layout()

                #         st.pyplot(fig)

                #     else:

                #         st.success(
                #             "Tidak ada missing value."
                #         )
                # # =============================
                # # PENJELASAN OTOMATIS MISSING VALUE
                # # =============================
                # total_missing_umum = missing_clean["Jumlah Missing"].sum()

                # missing_model_check = (
                #     df_final[fitur_X + [target_Y]]
                #     .isnull()
                #     .sum()
                # )

                # total_missing_model = missing_model_check.sum()

                # kolom_missing_umum = missing_clean[
                #     missing_clean["Jumlah Missing"] > 0
                # ]["Kolom"].tolist()

                # kolom_missing_model = missing_model_check[
                #     missing_model_check > 0
                # ].index.tolist()

                # if total_missing_umum == 0:
                #     st.success(
                #         """
                #         Tidak ditemukan missing value pada data setelah proses penghapusan duplikat.
                #         Data dapat dilanjutkan ke tahap pelabelan, seleksi fitur, dan case folding.
                #         """
                #     )

                # elif total_missing_model == 0:
                #     st.info(
                #         f"""
                #         Missing value ditemukan pada beberapa kolom data umum,
                #         yaitu: **{", ".join(kolom_missing_umum)}**.

                #         Namun setelah dilakukan pelabelan dan seleksi fitur,
                #         tidak terdapat missing value pada kolom yang digunakan sebagai
                #         fitur model (X) maupun target (Y).

                #         Dengan demikian, data yang masuk ke proses modeling ID3 tetap aman digunakan.
                #         """
                #     )

                # else:
                #     st.warning(
                #         f"""
                #         Missing value masih ditemukan pada kolom yang digunakan untuk modeling,
                #         yaitu: **{", ".join(kolom_missing_model)}**.

                #         Data pada kolom tersebut perlu diperiksa kembali karena dapat memengaruhi
                #         proses pembentukan model Decision Tree ID3.
                #         """
                #     )
                
                # =============================
                # DATA SETELAH CLEANING
                # =============================
                st.markdown("### 📄 Data Setelah Cleaning")

                kolom_tampil = [
                    "Nama Regu",
                    "No Laporan",
                    "Fasilitas",
                    "Peralatan",
                    "Dampak Kerusakan",
                    "Penyebab",
                    "Kelompok Penyebab",
                    "Cuaca",
                ]

                df_cleaning_tampil = df_no_dup[kolom_tampil].copy()

                search_cleaning = st.text_input(
                    "Cari data setelah cleaning",
                    placeholder="Cari No Laporan, nama regu, fasilitas, peralatan, dampak kerusakan, penyebab, atau cuaca...",
                    key="search_data_setelah_cleaning"
                )

                if search_cleaning:
                    mask_cleaning = df_cleaning_tampil.astype(str).apply(
                        lambda row: row.str.contains(
                            search_cleaning,
                            case=False,
                            na=False
                        ).any(),
                        axis=1
                    )

                    df_cleaning_tampil = df_cleaning_tampil[mask_cleaning]

                st.dataframe(
                    df_cleaning_tampil,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "Data di atas merupakan data setelah penghapusan duplikat berdasarkan No Laporan."
                )

            with tab3:
                st.subheader("📊 Perbandingan Visual Sebelum & Sesudah Cleaning")
                col_tbl, col_chart = st.columns([1, 1])
                with col_tbl:
                    df_comp = pd.DataFrame({
                        "Tahap": ["Sebelum Cleaning", "Setelah Cleaning", "Selisih"],
                        "Jumlah Data": [len(df), len(df_no_dup), len(df) - len(df_no_dup)]
                    })
                    st.dataframe(df_comp, use_container_width=True, hide_index=True)
                with col_chart:
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    bars = ax.bar(["Sebelum","Sesudah"], [len(df), len(df_no_dup)],
                                  color=["#dc2626","#16a34a"], width=0.45, edgecolor="white")
                    ax.bar_label(bars, fontweight="bold", fontsize=11)
                    ax.set_title("Jumlah Data Sebelum vs Sesudah", fontweight="bold")
                    ax.set_ylabel("Record")
                    plt.tight_layout()
                    st.pyplot(fig)

            with tab4:
                st.subheader("🏷️ Pelabelan — Variabel Target")
                
                col_tbl, col_chart = st.columns([1, 1])
                
                with col_tbl:
                    df_aturan = pd.DataFrame({"Kode Regu":["7","8"],
                                              "Jenis Kendaraan":["Motor","Mobil"],
                                              "Jenis Gangguan":["Ringan","Berat"]})
                    st.markdown("**Aturan Penentuan Jenis Gangguan:**")
                    st.dataframe(df_aturan, use_container_width=True, hide_index=True)
                    st.info("Kode 7 → Ringan (motor), Kode 8 → Berat (mobil).")
                    
                    # =============================
                    # VALIDASI DATA TIDAK TERLABEL
                    # =============================
                    data_tidak_terlabel = len(df_no_dup) - len(df_clean_pre)

                    if data_tidak_terlabel > 0:
                        st.warning(
                            f"""
                            Terdapat **{data_tidak_terlabel} data** yang tidak dapat diberi label
                            karena kode pada kolom **Nama Regu** tidak termasuk kode 7 atau 8.
                            Data tersebut tidak digunakan pada proses modeling.
                            """
                        )

                        df_tidak_terlabel = df_no_dup[
                            df_no_dup["Jenis Gangguan"].isna()
                        ]

                        with st.expander("📄 Lihat Data Tidak Terlabel"):
                            st.dataframe(
                                df_tidak_terlabel[
                                    [
                                        "No Laporan",
                                        "Nama Regu",
                                        "Fasilitas",
                                        "Peralatan",
                                        "Dampak Kerusakan",
                                        "Penyebab",
                                        "Kelompok Penyebab",
                                        "Cuaca"
                                    ]
                                ],
                                use_container_width=True,
                                hide_index=True
                            )
                    else:
                        st.success(
                            "Semua data setelah cleaning berhasil diberi label berdasarkan kode regu 7 dan 8."
                        )

                with col_chart:
                    total_berhasil_label = len(df_clean_pre)
                    total_tidak_terlabel = data_tidak_terlabel

                    components.html(f"""
                    <div style="
                        font-family: Arial, sans-serif;
                        background: #ffffff;
                        border: 1px solid #e2e8f0;
                        border-radius: 14px;
                        padding: 18px 20px;
                        box-shadow: 0 1px 4px rgba(15,23,42,0.06);
                        width: 100%;
                        box-sizing: border-box;
                    ">
                        <div style="
                            color: #0f2540;
                            font-size: 17px;
                            font-weight: 800;
                            margin-bottom: 14px;
                        ">
                            Ringkasan Pelabelan
                        </div>

                        <div style="
                            background: #f0fdf4;
                            border: 1px solid #bbf7d0;
                            border-left: 5px solid #16a34a;
                            border-radius: 10px;
                            padding: 12px 14px;
                            margin-bottom: 10px;
                        ">
                            <div style="color:#166534;font-size:13px;font-weight:700;">
                                Data Berhasil Diberi Label
                            </div>
                            <div style="color:#166534;font-size:26px;font-weight:800;margin-top:4px;">
                                {total_berhasil_label} data
                            </div>
                        </div>

                        <div style="
                            background: #fff7ed;
                            border: 1px solid #fed7aa;
                            border-left: 5px solid #f97316;
                            border-radius: 10px;
                            padding: 12px 14px;
                            margin-bottom: 12px;
                        ">
                            <div style="color:#9a3412;font-size:13px;font-weight:700;">
                                Data Tidak Terlabel
                            </div>
                            <div style="color:#9a3412;font-size:26px;font-weight:800;margin-top:4px;">
                                {total_tidak_terlabel} data
                            </div>
                        </div>
                    </div>
                    """, height=380, scrolling=False)
                    
                    # with st.expander("🧾 Lihat Ringkasan Pelabelan per Regu"):
                    #     df_regu_label = df_no_dup.dropna(subset=["Jenis Gangguan"]).copy()

                    #     df_regu_label["Nama Regu"] = (
                    #         df_regu_label["Nama Regu"]
                    #         .astype(str)
                    #         .str.strip()
                    #     )

                    #     df_regu_label["Kode Regu"] = df_regu_label["Nama Regu"].apply(ambil_kode_awal_regu)

                    #     ringkasan_regu = (
                    #         df_regu_label
                    #         .groupby(["Kode Regu", "Nama Regu", "Jenis Gangguan"])
                    #         .size()
                    #         .reset_index(name="Jumlah Data")
                    #     )

                    #     ringkasan_regu["Persentase"] = (
                    #         ringkasan_regu["Jumlah Data"] / len(df_clean_pre) * 100
                    #     ).round(2).astype(str) + "%"

                    #     ringkasan_regu = ringkasan_regu.sort_values(
                    #         ["Kode Regu", "Jumlah Data"],
                    #         ascending=[True, False]
                    #     )

                    #     filter_gangguan = st.selectbox(
                    #         "Filter Jenis Gangguan",
                    #         ["Semua", "Ringan", "Berat"],
                    #         key="filter_regu_pelabelan"
                    #     )

                    #     if filter_gangguan != "Semua":
                    #         ringkasan_regu = ringkasan_regu[
                    #             ringkasan_regu["Jenis Gangguan"] == filter_gangguan
                    #         ]

                    #     # Sembunyikan Persentase
                    #     ringkasan_regu_tampil = ringkasan_regu.drop(columns=["Persentase"])

                    #     st.dataframe(
                    #         ringkasan_regu_tampil,
                    #         use_container_width=True,
                    #         hide_index=True
                    #     )

                st.markdown("### 📄 Contoh Hasil Pelabelan")
                df_label = df_no_dup[
                    [
                        "No Laporan",
                        "Nama Regu",
                        "Jenis Gangguan"
                    ]
                ].copy()

                st.dataframe(
                    df_label.head(10),
                    use_container_width=True,
                    hide_index=True
                )
            
            with tab5:
                st.subheader("📦 Data Final")

                df_model_sel = df_clean_pre[fitur_X + [target_Y]].copy()

                tab5a, tab5b, tab5c = st.tabs([
                    "📄 Seleksi Fitur",
                    "🔡 Case Folding",
                    "✅ Data Siap Modeling"
                ])
                
                with tab5a:
                    df_fitur = pd.DataFrame({
                        "Variabel": ["X"]*len(fitur_X) + ["Y"],
                        "Nama Variabel": fitur_X + [target_Y],
                        "Jenis Variabel": ["Variabel Fitur"]*len(fitur_X) + ["Variabel Target"]
                    })
                    st.dataframe(df_fitur, use_container_width=True, hide_index=True)
                    st.info(
                        """
                        Seleksi fitur dilakukan secara manual menggunakan atribut
                        yang menggambarkan kondisi gangguan listrik,
                        yaitu fasilitas terdampak,
                        peralatan yang mengalami gangguan,
                        dampak kerusakan,
                        penyebab gangguan,
                        kelompok penyebab,
                        serta kondisi cuaca saat gangguan terjadi.

                        Atribut tersebut digunakan karena tersedia pada data laporan gangguan PLN
                        dan memiliki hubungan langsung dengan proses penanganan gangguan di lapangan.
                        Selain itu, atribut yang digunakan berupa data kategorikal
                        sehingga sesuai dengan algoritma Decision Tree ID3.
                        """
                    )
                    
                    # =============================
                    # STATUS PENGGUNAAN KOLOM
                    # =============================
                    st.markdown("### 📋 Status Penggunaan Kolom")

                    df_status = pd.DataFrame({
                        "Kolom": [
                            "No Laporan",
                            "Nama Regu",
                            "Fasilitas",
                            "Peralatan",
                            "Dampak Kerusakan",
                            "Penyebab",
                            "Kelompok Penyebab",
                            "Cuaca",
                            "Jenis Gangguan"
                        ],

                        "Status": [
                            "Digunakan untuk deduplikasi",
                            "Digunakan untuk pelabelan",
                            "Digunakan sebagai fitur",
                            "Digunakan sebagai fitur",
                            "Digunakan sebagai fitur",
                            "Digunakan sebagai fitur",
                            "Digunakan sebagai fitur",
                            "Digunakan sebagai fitur",
                            "Digunakan sebagai target"
                        ],

                        "Alasan": [
                            "Hanya sebagai identitas laporan dan aturan penghapusan data duplikat",
                            "Digunakan untuk menentukan label Ringan atau Berat",
                            "Menunjukkan jenis fasilitas yang terdampak gangguan",
                            "Menunjukkan peralatan yang mengalami gangguan",
                            "Menunjukkan dampak kerusakan akibat gangguan",
                            "Menunjukkan penyebab utama gangguan",
                            "Mengelompokkan penyebab gangguan",
                            "Menunjukkan kondisi cuaca saat gangguan terjadi",
                            "Menjadi kelas target yang akan diklasifikasikan oleh model"
                        ]
                    })

                    st.dataframe(
                        df_status,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.markdown("**Data Setelah Seleksi Fitur:**")
                    st.dataframe(df_model_sel, use_container_width=True, hide_index=True)
                    
                with tab5b:
                    st.markdown("**Case Folding — semua nilai teks diubah ke huruf kecil:**")
                    
                    st.info(
                        """
                        Case Folding dilakukan dengan mengubah seluruh teks
                        menjadi huruf kecil
                        untuk menjaga konsistensi data kategorikal
                        dan menghindari perbedaan penulisan data,
                        seperti penggunaan huruf besar dan kecil.
                        """
                    )
                    
                    df_norm = df_final.copy()
                    st.markdown("### 🔄 Contoh Perubahan Case Folding")

                    # =============================
                    # CONTOH Case Folding
                    # =============================
                    df_before_after = pd.DataFrame({
                        "Sebelum": ["Padam", "GARDU", "PETIR"],
                        "Sesudah": ["padam", "gardu", "petir"]
                    })

                    st.dataframe(
                        df_before_after,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # =============================
                    # HASIL Case Folding
                    # =============================
                    st.markdown("### 📄 Data Setelah Case Folding")
                    st.dataframe(df_norm, use_container_width=True, hide_index=True)
                    
                with tab5c:

                    st.info(
                        """
                        Dataset pada tahap ini merupakan hasil akhir preprocessing
                        yang telah melalui proses cleaning,
                        pelabelan,
                        seleksi fitur,
                        dan case folding.

                        Dataset tersebut selanjutnya digunakan
                        pada proses pembentukan model Decision Tree ID3.
                        """
                    )

                    # =============================
                    # RINGKASAN DATASET FINAL
                    # =============================
                    # kelas_target = ", ".join(
                    #     sorted(df_final[target_Y].unique())
                    # )

                    # col1, col2 = st.columns(2)
                    # col1.metric("Jumlah Fitur", len(fitur_X))
                    # col2.metric("Kelas Target", kelas_target)

                    # # =============================
                    # # VALIDASI DATA SIAP MODELING
                    # # =============================
                    # st.markdown("### 🧪 Validasi Data Siap Modeling")

                    # missing_model = (
                    #     df_final[fitur_X + [target_Y]]
                    #     .isnull()
                    #     .sum()
                    #     .reset_index()
                    # )

                    # missing_model.columns = [
                    #     "Kolom Model",
                    #     "Jumlah Missing"
                    # ]

                    # st.dataframe(
                    #     missing_model,
                    #     use_container_width=True,
                    #     hide_index=True
                    # )

                    # total_missing_model = missing_model["Jumlah Missing"].sum()

                    # if total_missing_model == 0:
                    #     st.success(
                    #         "Tidak terdapat missing value pada fitur dan target yang digunakan untuk modeling ID3."
                    #     )
                    # else:
                    #     st.warning(
                    #         f"Terdapat **{total_missing_model} missing value** pada kolom yang digunakan untuk modeling."
                    #     )
                    
                    # =============================
                    # DATA FINAL
                    # =============================
                    st.markdown("### 📄 Dataset Final")

                    df_final_tampil = df_final.copy()

                    search_final = st.text_input(
                        "Cari data final / data siap modeling",
                        placeholder="Cari fasilitas, peralatan, dampak kerusakan, penyebab, kelompok penyebab, cuaca, atau jenis gangguan...",
                        key="search_dataset_final_modeling"
                    )

                    if search_final:
                        mask_final = df_final_tampil.astype(str).apply(
                            lambda row: row.str.contains(
                                search_final,
                                case=False,
                                na=False
                            ).any(),
                            axis=1
                        )

                        df_final_tampil = df_final_tampil[mask_final]

                    st.dataframe(
                        df_final_tampil,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Dataset final hanya berisi atribut yang digunakan "
                        "pada proses klasifikasi ID3."
                    )

                    st.success(
                        f"✅ Total data siap digunakan: **{len(df_final)}** record"
                    )

            with tab6:
                st.subheader("✂️ Data Splitting — Stratified 80/20")
                train_summary, test_summary, gap = get_split_summary(train_df, test_df)
                
                st.info(
                    """
                    Pembagian data dilakukan dengan rasio 80:20 menggunakan stratified split.

                    Metode ini digunakan untuk menjaga proporsi kelas target
                    antara data latih dan data uji tetap terjaga,
                    sehingga distribusi kategori Ringan dan Berat
                    pada kedua dataset tidak berbeda jauh.

                    """
                )
                
                # =============================
                # VISUAL FLOW SPLIT DATA
                # =============================
                st.markdown("### 🔄 Alur Pembagian Data")

                total_final = len(df_final)
                total_train = len(train_df)
                total_test = len(test_df)

                persen_train = (total_train / total_final) * 100
                persen_test = (total_test / total_final) * 100

                components.html(f"""
                <div style="
                    font-family: Arial, sans-serif;
                    display: grid;
                    grid-template-columns: 1fr 70px 1fr 70px 1fr;
                    gap: 12px;
                    align-items: center;
                    width: 100%;
                ">

                    <div style="
                        background: #f8fafc;
                        border: 1px solid #cbd5e1;
                        border-radius: 14px;
                        padding: 18px;
                        text-align: center;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
                    ">
                        <div style="font-size: 1.8rem;">📦</div>
                        <div style="font-weight: 800; color: #0f2540; margin-top: 6px;">
                            Dataset Final
                        </div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #2563eb;">
                            {total_final}
                        </div>
                        <div style="font-size: 0.85rem; color: #64748b;">
                            data siap modeling
                        </div>
                    </div>

                    <div style="
                        text-align: center;
                        font-size: 2rem;
                        color: #94a3b8;
                        font-weight: 800;
                    ">
                        →
                    </div>

                    <div style="
                        background: #eff6ff;
                        border: 1px solid #bfdbfe;
                        border-radius: 14px;
                        padding: 18px;
                        text-align: center;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
                    ">
                        <div style="font-size: 1.8rem;">✂️</div>
                        <div style="font-weight: 800; color: #1e40af; margin-top: 6px;">
                            Stratified Split
                        </div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #1d4ed8;">
                            80 : 20
                        </div>
                        <div style="font-size: 0.85rem; color: #475569;">
                            proporsi kelas dijaga
                        </div>
                        
                        <div style="
                            display: flex;
                            justify-content: center;
                            gap: 8px;
                            margin-top: 10px;
                            flex-wrap: wrap;
                        ">
                            <span style="
                                background: #dbeafe;
                                color: #1e40af;
                                padding: 5px 10px;
                                border-radius: 999px;
                                font-size: 0.72rem;
                                font-weight: 700;
                            ">
                                stratify = Jenis Gangguan
                            </span>

                            
                        </div>
                    </div>

                    <div style="
                        text-align: center;
                        font-size: 2rem;
                        color: #94a3b8;
                        font-weight: 800;
                    ">
                        →
                    </div>

                    <div style="display: grid; gap: 10px;">
                        <div style="
                            background: #f0fdf4;
                            border: 1px solid #bbf7d0;
                            border-left: 5px solid #16a34a;
                            border-radius: 12px;
                            padding: 14px 16px;
                        ">
                            <div style="font-weight: 800; color: #166534;">
                                📘 Data Latih
                            </div>
                            <div style="font-size: 1.3rem; font-weight: 800; color: #15803d;">
                                {total_train} data
                            </div>
                            <div style="font-size: 0.85rem; color: #64748b;">
                                {persen_train:.2f}% dari dataset final
                            </div>
                        </div>

                        <div style="
                            background: #fff7ed;
                            border: 1px solid #fed7aa;
                            border-left: 5px solid #f97316;
                            border-radius: 12px;
                            padding: 14px 16px;
                        ">
                            <div style="font-weight: 800; color: #9a3412;">
                                📕 Data Uji
                            </div>
                            <div style="font-size: 1.3rem; font-weight: 800; color: #ea580c;">
                                {total_test} data
                            </div>
                            <div style="font-size: 0.85rem; color: #64748b;">
                                {persen_test:.2f}% dari dataset final
                            </div>
                        </div>
                    </div>

                </div>
                """, height=250, scrolling=False)

                col_tbl, col_chart = st.columns([1, 1])
                with col_tbl:
                    df_summary = pd.DataFrame([
                        {
                            "Jenis Data": "Data Latih",
                            "Total": train_summary["Total Data"],
                            "Berat": train_summary["Jumlah Berat"],
                            "Ringan": train_summary["Jumlah Ringan"],
                            "Entropy": f"{train_summary['Entropy']:.3f}"
                        },
                        {
                            "Jenis Data": "Data Uji",
                            "Total": test_summary["Total Data"],
                            "Berat": test_summary["Jumlah Berat"],
                            "Ringan": test_summary["Jumlah Ringan"],
                            "Entropy": f"{test_summary['Entropy']:.3f}"
                        },
                    ])
                    st.markdown("**Ringkasan Distribusi:**")
                    st.dataframe(df_summary, use_container_width=True, hide_index=True)
                    gap_val = gap["Gap Entropy"]
                    st.success(f"Selisih entropy latih-uji: **{gap_val:.3f}** ({gap_val*100:.2f}%)")
                    
                with col_chart:
                    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
                    for i, (label, summary) in enumerate([("Latih", train_summary), ("Uji", test_summary)]):
                        vals = [summary["Jumlah Berat"], summary["Jumlah Ringan"]]
                        axes[i].pie(vals, labels=["Berat","Ringan"], autopct="%1.2f%%",
                                    colors=["#dc2626","#16a34a"],
                                    wedgeprops={"edgecolor":"white","linewidth":2})
                        axes[i].set_title(f"Data {label}", fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig)

                # =============================
                # TABEL DATA LATIH
                # =============================
                st.subheader("📘 Data Latih")
                st.caption(
                    "Dataset berikut digunakan pada proses pelatihan model ID3."
                )

                search_train = st.text_input(
                    "🔍 Cari pada Data Latih",
                    placeholder="Contoh: mcb, petir, berat, ringan...",
                    key="search_train_split"
                )

                if search_train:
                    mask_train = train_df.astype(str).apply(
                        lambda row: row.str.contains(
                            search_train,
                            case=False,
                            na=False,
                            regex=False
                        ).any(),
                        axis=1
                    )

                    train_tampil = train_df[mask_train]

                else:
                    train_tampil = train_df

                st.dataframe(
                    train_tampil,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    f"Menampilkan {len(train_tampil)} dari {len(train_df)} data latih."
                )


                # =============================
                # TABEL DATA UJI 
                # =============================
                st.subheader("📕 Data Uji")
                st.caption(
                    "Dataset berikut digunakan pada proses pengujian model ID3."
                )

                search_test = st.text_input(
                    "🔍 Cari pada Data Uji",
                    placeholder="Contoh: mcb, petir, berat, ringan...",
                    key="search_test_split"
                )

                if search_test:
                    mask_test = test_df.astype(str).apply(
                        lambda row: row.str.contains(
                            search_test,
                            case=False,
                            na=False,
                            regex=False
                        ).any(),
                        axis=1
                    )

                    test_tampil = test_df[mask_test]

                else:
                    test_tampil = test_df

                st.dataframe(
                    test_tampil,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    f"Menampilkan {len(test_tampil)} dari {len(test_df)} data uji."
                )

        else:
            st.warning("⬆️ Upload file terlebih dahulu melalui panel kiri.")

    # =========================================================
    # 🔍 EXPLORATORY DATA ANALYSIS
    # =========================================================
    elif tahap == "3. Exploratory Data Analysis":
        st.title("🔍 Tahap 3 — Exploratory Data Analysis")
        st.caption(
            "Mengeksplorasi distribusi target, distribusi atribut, hubungan fitur dengan target, "
            "dan ringkasan atribut berdasarkan regu sebagai pemahaman awal karakteristik data."
        )
        st.markdown("---")

        if "df_final" in st.session_state:
            df = st.session_state.df_final.copy()
            fitur_X = FITUR_X.copy()
            target_Y = TARGET_Y

            total = len(df)
            n_berat = len(df[df[target_Y] == "Berat"])
            n_ringan = len(df[df[target_Y] == "Ringan"])

            # =============================
            # 📊 Distribusi Kelas Target (Data Final)
            # =============================
            st.markdown("### 📊 Distribusi Kelas Target (Data Final)")
            col_tbl_dist, col_chart_dist = st.columns([1,1])

            with col_tbl_dist:

                df_dist = pd.DataFrame({
                    "Jenis Gangguan": ["Berat", "Ringan"],
                    "Jumlah": [n_berat, n_ringan],
                    "Persentase": [
                        f"{(n_berat/total)*100:.2f}%",
                        f"{(n_ringan/total)*100:.2f}%"
                    ]
                })

                st.dataframe(
                    df_dist,
                    use_container_width=True,
                    hide_index=True
                )

                # st.info(
                #     """
                #     Distribusi kelas target pada data latih
                #     digunakan untuk melihat keseimbangan data
                #     sebelum proses pembentukan pohon keputusan ID3.
                #     """
                # )
                
                insight_box(
                    "Penjelasan Distribusi Kelas Target",
                    buat_insight_distribusi(total, n_berat, n_ringan),
                    color="#7c3aed"
                )

            with col_chart_dist:

                fig, ax = plt.subplots(figsize=(4.5,4))

                ax.pie(
                    [n_berat, n_ringan],
                    labels=["Berat", "Ringan"],
                    autopct="%1.2f%%",
                    colors=["#dc2626", "#16a34a"],
                    wedgeprops={
                        "edgecolor": "white",
                        "linewidth": 2
                    }
                )

                ax.set_title(
                    "Distribusi Target Data Final",
                    fontweight="bold"
                )

                ax.axis("off")

                st.pyplot(fig)

            # =============================
            # DISTRIBUSI UMUM SETIAP ATRIBUT
            # =============================
            st.subheader("📌 Distribusi Umum Setiap Atribut")
            st.caption(
                "Bagian ini digunakan untuk melihat sebaran nilai pada setiap atribut "
                "tanpa mengaitkannya terlebih dahulu dengan target Ringan atau Berat."
            )

            fitur_distribusi = st.selectbox(
                "Pilih atribut untuk melihat distribusi umum",
                fitur_X,
                key="eda_distribusi_umum"
            )

            distribusi_umum = (
                df[fitur_distribusi]
                .value_counts()
                .reset_index()
            )

            distribusi_umum.columns = ["Nilai Atribut", "Jumlah"]
            distribusi_umum["Persentase"] = (
                distribusi_umum["Jumlah"] / distribusi_umum["Jumlah"].sum() * 100
            )

            distribusi_tampil = distribusi_umum.copy()
            distribusi_tampil["Persentase"] = distribusi_tampil["Persentase"].apply(
                lambda x: f"{x:.2f}%"
            )

            nilai_terbanyak = distribusi_umum.iloc[0]["Nilai Atribut"]
            jumlah_terbanyak = distribusi_umum.iloc[0]["Jumlah"]
            persen_terbanyak = distribusi_umum.iloc[0]["Persentase"]

            col_umum_tbl, col_umum_chart = st.columns([1, 1])

            with col_umum_tbl:
                st.dataframe(
                    distribusi_tampil,
                    use_container_width=True,
                    hide_index=True
                )

                insight_box(
                    "Insight Distribusi Umum",
                    buat_insight_distribusi_umum(distribusi_umum, fitur_distribusi),
                    color="#0891b2"
                )

            with col_umum_chart:
                fig, ax = plt.subplots(figsize=(5.5, 4))

                bars = ax.bar(
                    distribusi_umum["Nilai Atribut"].astype(str),
                    distribusi_umum["Jumlah"],
                    color="#0891b2",
                    edgecolor="white"
                )

                ax.bar_label(bars, fontsize=9, padding=3)
                ax.set_title(
                    f"Distribusi Nilai Atribut: {fitur_distribusi}",
                    fontweight="bold"
                )
                ax.set_ylabel("Jumlah Data")
                plt.xticks(rotation=35, ha="right", fontsize=8)
                plt.tight_layout()

                st.pyplot(fig)

            # =============================
            # CROSSTAB (HUBUNAGN FITUR DENGAN TARGET)
            # =============================
            st.subheader("🔗 Hubungan Fitur dengan Target")
            fitur_crosstab = st.selectbox(
                "Pilih atribut",
                fitur_X,
                key="crosstab"
            )

            # Crosstab untuk chart
            ct = pd.crosstab(
                df[fitur_crosstab],
                df[target_Y]
            )

            ct.index.name = fitur_crosstab

            # Crosstab untuk tabel
            ct_table = ct.reset_index()
            ct_table = ct_table.rename(columns={
                fitur_crosstab: "Nilai Atribut"
            })

            # Tambahkan total data per nilai atribut
            ct_table["Total Data"] = ct_table.select_dtypes(include="number").sum(axis=1)

            col_tbl, col_chart = st.columns([1, 1])

            with col_tbl:
                st.info(
                    "Tabel ini digunakan untuk melihat hubungan antara nilai atribut "
                    "dengan kategori target pada data final."
                )

                st.dataframe(
                    ct_table,
                    use_container_width=True,
                    hide_index=True
                )

            with col_chart:
                fig, ax = plt.subplots(figsize=(5.5, 4))

                ct.plot(
                    kind="bar",
                    ax=ax,
                    color=["#dc2626", "#16a34a"],
                    edgecolor="white"
                )

                ax.set_title(
                    f"Hubungan {fitur_crosstab} vs Target",
                    fontweight="bold"
                )

                ax.set_ylabel("Jumlah")
                ax.legend(title="Jenis Gangguan")

                plt.xticks(rotation=35, ha="right", fontsize=8)
                plt.tight_layout()

                st.pyplot(fig)

            insight_box(
                f"Penjelasan Hubungan {fitur_crosstab} dengan Target",
                buat_insight_crosstab(ct_table, fitur_crosstab),
                color="#0891b2"
            )

            st.subheader("🧾 Ringkasan Atribut Berdasarkan Regu")
            df_regu = st.session_state.get("df_eda_regu")

            if df_regu is not None and "Nama Regu" in df_regu.columns:
                df_ringkasan_regu = buat_ringkasan_atribut_nama_regu(
                    df_regu,
                    fitur_X
                )
                df_ringkasan_regu = df_ringkasan_regu.drop(
                    columns=["Gangguan"],
                    errors="ignore"
                )

                st.dataframe(
                    df_ringkasan_regu,
                    use_container_width=True,
                    hide_index=True
                )
                insight_box(
                    "Penjelasan Ringkasan Atribut per Regu",
                    "Tabel di atas menunjukkan nilai yang paling sering muncul pada setiap regu "
                    "berdasarkan enam atribut yang digunakan. Dari tabel di atas dapat dilihat regu mana saja "
                    "yang sering menangani gangguan berdasarkan jumlah data, serta kondisi gangguan apa saja "
                    "yang paling banyak ditangani oleh setiap regu selama periode data penelitian.",
                    color="#0891b2"
                )

                st.markdown("---")
                st.markdown("""
                <div style="
                    background-color: #f8fafc;
                    border: 1px solid #e2e8f0;
                    border-left: 6px solid #10b981;
                    padding: 20px;
                    border-radius: 12px;
                    margin-top: 10px;
                ">
                    <h4 style="color: #065f46; margin-top: 0;">🎯 Kesimpulan EDA</h4>
                    <p style="color: #334155; margin-bottom: 0; line-height: 1.7;">
                        Berdasarkan hasil eksplorasi data, data final menunjukkan distribusi target yang tidak seimbang,
                        dengan kelas Berat lebih dominan dibandingkan kelas Ringan. Hubungan fitur dengan target
                        juga menunjukkan bahwa beberapa nilai atribut masih muncul pada kedua kelas, sehingga
                        klasifikasi tingkat gangguan tidak cukup ditentukan dari satu atribut saja. Selain itu,
                        ringkasan atribut berdasarkan regu memberikan gambaran operasional mengenai gangguan
                        yang paling sering muncul pada setiap regu. Dengan demikian, tahap EDA memberikan
                        pemahaman awal terhadap karakteristik data sebelum dilanjutkan ke tahap Modeling ID3
                        untuk membentuk pohon keputusan klasifikasi gangguan Ringan dan Berat.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.info(
                    "Data nama regu belum tersedia. Jalankan tahap Data Preparation terlebih dahulu."
                )

    # =========================================================
    # 🌳 MODELING ID3
    # =========================================================
    elif tahap == "4. Modeling":
        st.title("🌳 Tahap 4 — Modeling ID3")
        st.caption("Membangun pohon keputusan ID3 dan menampilkan visualisasi pohon")
        st.markdown("---")

        if "train_df" in st.session_state:
            import numpy as np
            train_df = st.session_state.train_df
            fitur_X = FITUR_X.copy()
            target_Y = TARGET_Y

            info = get_model_info(train_df, fitur_X, target_Y)
            model_tree = train_model(train_df)
            st.session_state.model_tree = model_tree

            root = info["root"]
            root_gain = info["gains"][root]
            n = len(train_df)
            n_berat = len(train_df[train_df[target_Y]=="Berat"])
            n_ringan = len(train_df[train_df[target_Y]=="Ringan"])
            entropy_val = calculate_entropy(train_df[target_Y])

            # =============================
            # DASHBOARD CARD MODELING
            # =============================
            jumlah_fitur = len(fitur_X)
            target_prediksi = TARGET_Y
            kelas_target = "Berat / Ringan"
            root_node = root
            root_gain_label = f"IG = {root_gain:.3f}"

            components.html(f"""
            <div style="
                font-family: Arial, sans-serif;
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 14px;
                width: 100%;
                margin: 8px 0 18px 0;
            ">

                <div style="
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-top: 5px solid #2563eb;
                    border-radius: 14px;
                    padding: 18px 16px;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                ">
                    <div style="font-size: 1.6rem;">🌳</div>
                    <div style="font-size: 0.78rem; color: #64748b; font-weight: 700; text-transform: uppercase; margin-top: 8px;">
                        Algoritma
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #0f2540; margin-top: 4px;">
                        ID3
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8;">
                        Decision Tree
                    </div>
                </div>

                <div style="
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-top: 5px solid #16a34a;
                    border-radius: 14px;
                    padding: 18px 16px;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                ">
                    <div style="font-size: 1.6rem;">📊</div>
                    <div style="font-size: 0.78rem; color: #64748b; font-weight: 700; text-transform: uppercase; margin-top: 8px;">
                        Fitur Input
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #16a34a; margin-top: 4px;">
                        {jumlah_fitur}
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8;">
                        atribut model
                    </div>
                </div>

                <div style="
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-top: 5px solid #7c3aed;
                    border-radius: 14px;
                    padding: 18px 16px;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                ">
                    <div style="font-size: 1.6rem;">🎯</div>
                    <div style="font-size: 0.78rem; color: #64748b; font-weight: 700; text-transform: uppercase; margin-top: 8px;">
                        Target & Kelas
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #7c3aed; margin-top: 8px; line-height: 1.3;">
                        {target_prediksi}
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8;">
                        {kelas_target}
                    </div>
                </div>

                <div style="
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-top: 5px solid #f97316;
                    border-radius: 14px;
                    padding: 18px 16px;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                ">
                    <div style="font-size: 1.6rem;">🌱</div>
                    <div style="font-size: 0.78rem; color: #64748b; font-weight: 700; text-transform: uppercase; margin-top: 8px;">
                        Root Node
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #f97316; margin-top: 8px; line-height: 1.3;">
                        {root_node}
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8;">
                        {root_gain_label}
                    </div>
                </div>

            </div>
            """, height=180, scrolling=False)

            tab1, tab2 = st.tabs(["🌳 Pembentukan Model","🌲 Pohon Keputusan"])

            with tab1:
                # =============================
                # RINGKASAN DATA LATIH
                # =============================
                df_gain = get_information_gain_ranking(
                    train_df,
                    fitur_X,
                    target_Y
                )

                df_gain["Information Gain"] = df_gain["Information Gain"].apply(
                    lambda x: f"{x:.3f}"
                )

                df_ringkasan_latih = pd.DataFrame({
                    "Komponen": [
                        "Total Data",
                        "Jumlah Berat",
                        "Jumlah Ringan"
                    ],
                    "Nilai": [
                        n,
                        n_berat,
                        n_ringan
                    ]
                })

                st.markdown("### Ringkasan Data Latih")
                st.dataframe(
                    df_ringkasan_latih,
                    use_container_width=True,
                    hide_index=True
                )

                insight_box(
                    "Penjelasan Ringkasan Data Latih",
                    f"Data latih merupakan data yang digunakan untuk membentuk pohon keputusan ID3. "
                    f"Pada penelitian ini, data latih berjumlah <b>{n}</b> data, terdiri dari "
                    f"<b>{n_berat}</b> data kelas Berat dan <b>{n_ringan}</b> data kelas Ringan. "
                    f"Komposisi kelas ini digunakan sebagai dasar untuk menghitung entropy data latih "
                    f"pada tahap berikutnya.",
                    color="#2563eb"
                )

                # =============================
                # PERHITUNGAN ENTROPY DATA LATIH
                # =============================
                st.markdown("### Perhitungan Entropy Data Latih")
                st.markdown(
                    f"""
                    **Entropy(S)** = -(({n_berat}/{n}) × log₂({n_berat}/{n})) - (({n_ringan}/{n}) × log₂({n_ringan}/{n}))

                    **Entropy(S)** = **{entropy_val:.3f}**
                    """
                )
                
                df_entropy_latih = pd.DataFrame({
                    "Komponen": [
                        "Entropy Data Latih"
                    ],
                    "Nilai": [
                        f"{entropy_val:.3f}"
                    ]
                })

                st.info(
                    f"Nilai entropy data latih adalah **{entropy_val:.3f}**. "
                    "Nilai ini digunakan sebagai entropy awal dalam perhitungan information gain setiap atribut."
                )

                # =============================
                # PERHITUNGAN DETAIL PER ATRIBUT
                # =============================
                # with st.expander("🔍 Lihat Perhitungan Detail per Atribut"):
                st.markdown("---")
                st.subheader("🔍 Detail Perhitungan per Atribut")
                atribut_pilih = st.selectbox("Pilih Atribut", fitur_X)
                
                hasil_ig = get_information_gain_detail(
                    train_df,
                    atribut_pilih,
                    target_Y
                )

                df_entropy = hasil_ig["detail_df"]

                gain = hasil_ig["information_gain"]
                
                col_t, col_c = st.columns([1,1])
                with col_t:
                    df_entropy_tampil = format_tabel_detail_id3(df_entropy)

                    st.dataframe(
                        df_entropy_tampil,
                        use_container_width=True,
                        hide_index=True
                    )

                    insight_box(
                        "Penjelasan Detail Perhitungan Atribut",
                        f"Tabel menunjukkan setiap nilai yang ada pada atribut tersebut, jumlah datanya, serta "
                        f"pembagian kelas Ringan dan Berat. Nilai entropy digunakan untuk melihat apakah data "
                        f"pada nilai atribut tersebut masih bercampur atau sudah mengarah pada satu kelas. "
                        f"Semakin kecil nilai entropy, maka data pada nilai tersebut semakin jelas mengarah "
                        f"ke salah satu kelas. ",
                        color="#2563eb"
                    )

                with col_c:
                    fig, ax = plt.subplots(figsize=(5,3.5))
                    
                    bars = ax.bar(
                        df_entropy["Nilai"].astype(str),
                        df_entropy["Entropy"], 
                        color="#2563eb",
                        edgecolor="white"
                    )

                    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)

                    ax.set_title(f"Entropy per Nilai: {atribut_pilih}", fontweight="bold")
                    ax.set_ylabel("Entropy"); plt.xticks(rotation=30, ha="right", fontsize=8); plt.tight_layout()
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
                    st.pyplot(fig)

                # =============================
                # RANKING INFORMATION GAIN DATA LATIH
                # =============================
                st.markdown("---")
                st.subheader("Ranking Information Gain Data Latih")

                df_gain_tampil = df_gain.copy()

                df_gain_tampil.insert(
                    0,
                    "Ranking",
                    [str(i) for i in range(1, len(df_gain_tampil) + 1)]
                )

                col_rank_tabel, col_rank_chart = st.columns([1, 1])

                with col_rank_tabel:
                    st.dataframe(
                        df_gain_tampil.style.set_properties(
                            subset=["Ranking"],
                            **{"text-align": "center"}
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                with col_rank_chart:
                    df_gain_chart = df_gain_tampil.copy()

                    df_gain_chart["Information Gain"] = df_gain_chart["Information Gain"].astype(float)

                    fig, ax = plt.subplots(figsize=(5.5, 3.8))

                    bars = ax.barh(
                        df_gain_chart["Atribut"],
                        df_gain_chart["Information Gain"],
                        color="#16a34a",
                        edgecolor="white"
                    )

                    ax.bar_label(
                        bars,
                        fmt="%.3f",
                        fontsize=9,
                        padding=3
                    )

                    ax.set_title(
                        "Ranking Information Gain",
                        fontweight="bold"
                    )

                    ax.set_xlabel("Information Gain")
                    ax.invert_yaxis()
                    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))

                    plt.tight_layout()
                    st.pyplot(fig)

                st.success(
                    f"Root node terpilih adalah **{root}** karena memiliki nilai Information Gain "
                    f"tertinggi sebesar **{root_gain:.3f}**."
                )

                insight_box(
                    "Penjelasan Ranking Information Gain",
                    f"Tabel dan grafik ranking information gain menunjukkan hasil perhitungan gain "
                    f"pada seluruh atribut data latih. Nilai information gain diperoleh dari entropy "
                    f"data latih yang dikurangi dengan total bobot entropy cabang pada masing-masing atribut. "
                    f"Semakin besar nilai information gain, semakin baik atribut tersebut dalam memisahkan "
                    f"data ke dalam kelas Ringan dan Berat. Berdasarkan hasil perhitungan, atribut "
                    f"<b>{root}</b> berada pada ranking pertama dengan nilai information gain sebesar "
                    f"<b>{root_gain:.3f}</b>, sehingga atribut tersebut dipilih sebagai root node pada "
                    f"pohon keputusan ID3.",
                    color="#16a34a"
                )
                
                # =============================
                # RECURSIVE ID3 EXPLORER
                # =============================
                SHOW_RECURSIVE_ID3 = True

                # Untuk demo, detail yang tampil langsung atribut dengan Information Gain tertinggi.
                # Ubah ke True kalau ingin memilih atribut detail lewat dropdown.
                SHOW_DETAIL_DROPDOWN_LEVEL_1 = False
                SHOW_DETAIL_DROPDOWN_LEVEL_2 = False

                # Level 3 tidak ditampilkan agar halaman modeling tidak terlalu panjang.
                SHOW_LEVEL_3 = False
                
                # =============================
                # STATUS KEMURNIAN CABANG ROOT
                # =============================
                # st.subheader("📌 Status Kemurnian Cabang Root")

                root_attr = root

                data_cabang_root = []

                for nilai_cabang in sorted(train_df[root_attr].dropna().unique()):
                    subset_cabang = train_df[
                        train_df[root_attr] == nilai_cabang
                    ]

                    target_cabang = subset_cabang[target_Y].astype(str).str.lower()

                    jumlah_data = len(subset_cabang)
                    jumlah_berat = (target_cabang == "berat").sum()
                    jumlah_ringan = (target_cabang == "ringan").sum()

                    entropy_cabang = calculate_entropy(
                        subset_cabang[target_Y]
                    )

                    data_cabang_root.append({
                        "Nilai Cabang Root": nilai_cabang,
                        "Jumlah Data": jumlah_data,
                        "Berat": jumlah_berat,
                        "Ringan": jumlah_ringan,
                        "Entropy": round(entropy_cabang, 3),
                        "Status": "Belum Murni" if entropy_cabang > 0 else "Murni / Leaf Node"
                    })

                df_cabang_root = pd.DataFrame(data_cabang_root)

                df_cabang_root_tampil = df_cabang_root.copy()

                df_cabang_root_tampil["Entropy"] = df_cabang_root_tampil["Entropy"].apply(
                    lambda x: f"{x:.3f}"
                )

                root_values_belum_murni = df_cabang_root[
                    df_cabang_root["Entropy"] > 0
                ]["Nilai Cabang Root"].tolist()

                df_cabang_belum_murni = df_cabang_root[
                    df_cabang_root["Entropy"] > 0
                ].copy()

                if not df_cabang_belum_murni.empty:
                    df_cabang_belum_murni_tampil = df_cabang_belum_murni.copy()

                    df_cabang_belum_murni_tampil["Entropy"] = df_cabang_belum_murni_tampil["Entropy"].apply(
                        lambda x: f"{x:.3f}"
                    )

                    with st.expander("📌 Lihat Cabang Root yang Belum Murni", expanded=False):
                        st.info(
                            "Tabel ini hanya menampilkan cabang pada root node yang masih memiliki entropy lebih dari 0, "
                            "sehingga perlu dilanjutkan ke proses pembentukan node berikutnya."
                        )

                        st.dataframe(
                            df_cabang_belum_murni_tampil,
                            use_container_width=True,
                            hide_index=True
                        )

                else:
                    st.success(
                        "Semua cabang pada root node sudah murni, sehingga tidak ada cabang root yang perlu diproses lebih lanjut."
                    )

                if SHOW_RECURSIVE_ID3:

                    st.markdown("---")
                    st.subheader("🌿 Recursive Node Explorer")

                    st.info(
                        """
                        Bagian ini menampilkan proses pembentukan node lanjutan pada algoritma ID3
                        secara bertahap.

                        Jika suatu node sudah murni, maka node tersebut menjadi leaf node dan
                        proses rekursif pada cabang tersebut dihentikan.
                        """
                    )

                    # =============================
                    # LEVEL 1
                    # =============================
                    st.markdown("## 🌳 Level 1")

                    root_attr = root

                    # =============================
                    # PILIH CABANG ROOT LEVEL 1
                    # =============================

                    # NOTE:
                    # Kode lama di bawah ini menampilkan SEMUA cabang root.
                    # Jika ingin menampilkan semua cabang lagi, uncomment bagian ini
                    # lalu comment blok filter cabang belum murni di bawahnya.

                    # root_values = sorted(
                    #     train_df[root_attr].dropna().unique()
                    # )
                    #
                    # root_value = st.selectbox(
                    #     f"Pilih Cabang Root ({root_attr})",
                    #     root_values
                    # )

                    if len(root_values_belum_murni) == 0:
                        st.success(
                            """
                            🍃 Semua cabang pada root node sudah murni.
                            Proses rekursif tidak perlu dilanjutkan ke Level 1.
                            """
                        )

                        root_value = None

                    else:
                        root_value = st.selectbox(
                            f"Pilih Cabang Root yang Belum Murni ({root_attr})",
                            root_values_belum_murni
                        )

                        # st.caption("Dropdown ini hanya menampilkan cabang root yang masih memiliki entropy lebih dari 0.")

                    # =============================
                    # PROSES LEVEL 1
                    # =============================
                    if root_value is not None:

                        path_filters_level_1 = [
                            (root_attr, root_value)
                        ]

                        recursive_1 = get_recursive_node_info(
                            train_df,
                            fitur_X,
                            target_Y,
                            path_filters_level_1
                        )

                        if recursive_1 is not None:

                            # =============================
                            # PATH NODE LEVEL 1
                            # =============================
                            components.html(f"""
                            <div style="
                                font-family: Arial, sans-serif;
                                background: #ffffff;
                                border: 1px solid #dbeafe;
                                border-left: 6px solid #2563eb;
                                border-radius: 14px;
                                padding: 14px 18px;
                                margin: 12px 0 22px 0;
                                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                            ">
                                <div style="
                                    font-size: 0.75rem;
                                    color: #64748b;
                                    font-weight: 800;
                                    letter-spacing: 0.06em;
                                    text-transform: uppercase;
                                    margin-bottom: 6px;
                                ">
                                    Path Node Level 1
                                </div>

                                <div style="
                                    font-size: 1.05rem;
                                    color: #0f2540;
                                    font-weight: 700;
                                    line-height: 1.5;
                                ">
                                    {root_attr}
                                    <span style="color:#64748b;">=</span>
                                    <span style="
                                        background:#eff6ff;
                                        color:#1d4ed8;
                                        padding: 4px 10px;
                                        border-radius: 999px;
                                        font-weight: 800;
                                    ">
                                        {root_value}
                                    </span>
                                </div>
                            </div>
                            """, height=105, scrolling=False)

                            # =============================
                            # CEK STATUS NODE LEVEL 1
                            # =============================
                            path_text_1 = f"{root_attr} = {root_value}"

                            node_status_1 = get_node_status_info(
                                df_node=recursive_1["subset_df"],
                                target_col=target_Y,
                                nama_level="Level 1",
                                path_node=path_text_1
                            )

                            if node_status_1["should_stop"]:

                                if node_status_1["is_empty"]:
                                    st.warning(
                                        "Cabang ini tidak memiliki data, sehingga proses rekursif tidak dapat dilanjutkan."
                                    )

                                elif node_status_1["is_pure"]:
                                    st.success(
                                        f"🍃 {node_status_1['nama_level']} sudah menjadi leaf node / node akhir."
                                    )

                                    st.info(
                                        f"""
                                        Semua data pada node ini masuk ke kelas **{node_status_1['kelas_final']}**.

                                        **Path node:** `{node_status_1['path_node']}`

                                        Karena node sudah murni, entropy bernilai **{node_status_1['entropy']:.3f}**
                                        dan proses rekursif tidak perlu dilanjutkan ke Level 2.
                                        """
                                    )

                                    st.dataframe(
                                        node_status_1["distribusi_df"],
                                        use_container_width=True,
                                        hide_index=True
                                    )

                            else:

                                # =============================
                                # RINGKASAN NODE & RANKING GAIN LEVEL 1
                                # =============================
                                next_attr_1 = recursive_1["best_attr"]

                                # st.markdown("### 📊 Ringkasan Node & Ranking Information Gain Level 1")

                                target_level_1 = recursive_1["subset_df"][target_Y].astype(str).str.lower()

                                jumlah_data_level_1 = len(recursive_1["subset_df"])
                                jumlah_berat_level_1 = (target_level_1 == "berat").sum()
                                jumlah_ringan_level_1 = (target_level_1 == "ringan").sum()
                                entropy_level_1 = recursive_1["entropy"]

                                df_ringkasan_level_1 = pd.DataFrame({
                                    "Komponen": [
                                        "Jumlah Data",
                                        "Jumlah Berat",
                                        "Jumlah Ringan",
                                        "Entropy Node"
                                    ],
                                    "Nilai": [
                                        jumlah_data_level_1,
                                        jumlah_berat_level_1,
                                        jumlah_ringan_level_1,
                                        f"{entropy_level_1:.3f}"
                                    ]
                                })

                                col_summary_1, col_rank_1 = st.columns([0.9, 1.6])

                                with col_summary_1:
                                    st.markdown("#### 🧾 Ringkasan Node Level 1")

                                    st.dataframe(
                                        df_ringkasan_level_1,
                                        use_container_width=True,
                                        hide_index=True
                                    )

                                    st.caption(
                                        "Ringkasan ini menunjukkan kondisi data pada node Level 1 sebelum pemilihan node berikutnya."
                                    )

                                with col_rank_1:
                                    st.markdown("#### 📈 Ranking Information Gain")

                                    gain_df_1_tampil = recursive_1["gain_df"].copy()
                                    gain_df_1_tampil["Information Gain"] = gain_df_1_tampil["Information Gain"].apply(
                                        lambda x: f"{x:.3f}"
                                    )

                                    st.dataframe(
                                        gain_df_1_tampil,
                                        use_container_width=True,
                                        hide_index=True
                                    )

                                st.success(
                                    f"Node berikutnya: **{next_attr_1}**"
                                )

                                # =============================
                                # DETAIL PERHITUNGAN LEVEL 1
                                # =============================
                                st.markdown("### 🔍 Detail Perhitungan per Atribut Level 1")

                                atribut_detail_1_list = recursive_1["gain_df"]["Atribut"].tolist()

                                default_index_1 = (
                                    atribut_detail_1_list.index(next_attr_1)
                                    if next_attr_1 in atribut_detail_1_list
                                    else 0
                                )

                                if SHOW_DETAIL_DROPDOWN_LEVEL_1:
                                    atribut_detail_1 = st.selectbox(
                                        "Pilih Atribut untuk Detail Perhitungan Level 1",
                                        atribut_detail_1_list,
                                        index=default_index_1,
                                        key=f"detail_level_1_{root_attr}_{root_value}"
                                    )
                                else:
                                    atribut_detail_1 = next_attr_1
                                    st.caption(
                                        f"Detail perhitungan Level 1 otomatis menampilkan atribut dengan "
                                        f"Information Gain tertinggi, yaitu **{next_attr_1}**."
                                    )

                                df_detail_1 = recursive_1["detail_tables"][atribut_detail_1]

                                gain_detail_1 = recursive_1["gain_df"].loc[
                                    recursive_1["gain_df"]["Atribut"] == atribut_detail_1,
                                    "Information Gain"
                                ].iloc[0]

                                col_detail_1, col_chart_1 = st.columns([1, 1])

                                with col_detail_1:
                                    df_detail_1_tampil = format_tabel_detail_id3(df_detail_1)

                                    st.dataframe(
                                        df_detail_1_tampil,
                                        use_container_width=True,
                                        hide_index=True
                                    )

                                with col_chart_1:
                                    fig, ax = plt.subplots(figsize=(5, 3.5))

                                    bars = ax.bar(
                                        df_detail_1["Nilai"].astype(str),
                                        df_detail_1["Entropy"],
                                        color="#2563eb",
                                        edgecolor="white"
                                    )

                                    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)

                                    ax.set_title(
                                        f"Entropy per Nilai: {atribut_detail_1}",
                                        fontweight="bold"
                                    )

                                    ax.set_ylabel("Entropy")
                                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

                                    max_entropy_1 = df_detail_1["Entropy"].max()

                                    if max_entropy_1 == 0:
                                        ax.set_ylim(0, 0.050)
                                    else:
                                        ax.set_ylim(0, max_entropy_1 + 0.050)

                                    plt.xticks(rotation=30, ha="right", fontsize=8)
                                    plt.tight_layout()

                                    st.pyplot(fig)

                                st.success(
                                    f"Information Gain ({atribut_detail_1}): **{gain_detail_1:.3f}**"
                                )

                                # st.caption(
                                #     f"Catatan: dropdown ini hanya untuk melihat detail perhitungan setiap atribut. "
                                #     f"Node berikutnya tetap dipilih berdasarkan Information Gain tertinggi, yaitu {next_attr_1}."
                                # )

                                # =============================
                                # CEK KELANJUTAN KE LEVEL 2
                                # =============================
                                entropy_not_zero_level_2 = recursive_1[
                                    "detail_tables"
                                ][next_attr_1]

                                valid_values_level_2 = entropy_not_zero_level_2[
                                    entropy_not_zero_level_2["Entropy"] != 0
                                ]["Nilai"].tolist()

                                if len(valid_values_level_2) > 0:
                                    st.info(
                                        f"""
                                        Atribut dengan Information Gain tertinggi pada Level 1 adalah **{next_attr_1}**.

                                        Namun, masih terdapat cabang pada atribut **{next_attr_1}**
                                        yang memiliki entropy lebih dari 0, sehingga cabang tersebut belum murni.

                                        Oleh karena itu, proses ID3 dilanjutkan ke **Level 2**
                                        untuk menghitung Information Gain pada atribut yang tersisa.
                                        """
                                    )

                                    with st.expander("📌 Lihat Cabang yang Belum Murni pada Level 1"):
                                        df_belum_murni_1 = entropy_not_zero_level_2[
                                            entropy_not_zero_level_2["Entropy"] != 0
                                        ].copy()

                                        df_belum_murni_1["Entropy"] = df_belum_murni_1["Entropy"].apply(
                                            lambda x: f"{x:.3f}"
                                        )

                                        st.dataframe(
                                            df_belum_murni_1,
                                            use_container_width=True,
                                            hide_index=True
                                        )

                                else:
                                    st.success(
                                        f"""
                                        🍃 Setelah node Level 1 dipecah berdasarkan atribut **{next_attr_1}**,
                                        seluruh cabang sudah memiliki entropy 0.

                                        Artinya setiap cabang sudah murni dan proses rekursif tidak perlu dilanjutkan.
                                        """
                                    )

                                # =============================
                                # LEVEL 2
                                # =============================
                                if len(valid_values_level_2) > 0:

                                    # st.markdown("---")
                                    st.markdown("## 🌿 Level 2")

                                    next_value_1 = st.selectbox(
                                        f"Pilih Cabang ({next_attr_1})",
                                        valid_values_level_2,
                                        key=f"level_2_branch_{root_attr}_{root_value}_{next_attr_1}"
                                    )

                                    path_filters_level_2 = [
                                        (root_attr, root_value),
                                        (next_attr_1, next_value_1)
                                    ]

                                    recursive_2 = get_recursive_node_info(
                                        train_df,
                                        fitur_X,
                                        target_Y,
                                        path_filters_level_2
                                    )

                                    if recursive_2 is not None:

                                        # =============================
                                        # PATH NODE LEVEL 2
                                        # =============================
                                        components.html(f"""
                                        <div style="
                                            font-family: Arial, sans-serif;
                                            background: #ffffff;
                                            border: 1px solid #dbeafe;
                                            border-left: 6px solid #2563eb;
                                            border-radius: 14px;
                                            padding: 14px 18px;
                                            margin: 12px 0 22px 0;
                                            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                                        ">
                                            <div style="
                                                font-size: 0.75rem;
                                                color: #64748b;
                                                font-weight: 800;
                                                letter-spacing: 0.06em;
                                                text-transform: uppercase;
                                                margin-bottom: 6px;
                                            ">
                                                Path Node Level 2
                                            </div>

                                            <div style="
                                                font-size: 1.05rem;
                                                color: #0f2540;
                                                font-weight: 700;
                                                line-height: 1.7;
                                            ">
                                                {root_attr}
                                                <span style="color:#64748b;">=</span>
                                                <span style="
                                                    background:#eff6ff;
                                                    color:#1d4ed8;
                                                    padding: 4px 10px;
                                                    border-radius: 999px;
                                                    font-weight: 800;
                                                ">
                                                    {root_value}
                                                </span>

                                                <span style="color:#94a3b8; margin: 0 8px;">→</span>

                                                {next_attr_1}
                                                <span style="color:#64748b;">=</span>
                                                <span style="
                                                    background:#f0fdf4;
                                                    color:#15803d;
                                                    padding: 4px 10px;
                                                    border-radius: 999px;
                                                    font-weight: 800;
                                                ">
                                                    {next_value_1}
                                                </span>
                                            </div>
                                        </div>
                                        """, height=115, scrolling=False)

                                        # =============================
                                        # CEK STATUS NODE LEVEL 2
                                        # =============================
                                        path_text_2 = (
                                            f"{root_attr} = {root_value} → "
                                            f"{next_attr_1} = {next_value_1}"
                                        )

                                        node_status_2 = get_node_status_info(
                                            df_node=recursive_2["subset_df"],
                                            target_col=target_Y,
                                            nama_level="Level 2",
                                            path_node=path_text_2
                                        )

                                        if node_status_2["should_stop"]:

                                            if node_status_2["is_empty"]:
                                                st.warning(
                                                    "Cabang ini tidak memiliki data, sehingga proses rekursif tidak dapat dilanjutkan."
                                                )

                                            elif node_status_2["is_pure"]:
                                                st.success(
                                                    f"🍃 {node_status_2['nama_level']} sudah menjadi leaf node / node akhir."
                                                )

                                                st.info(
                                                    f"""
                                                    Semua data pada node ini masuk ke kelas **{node_status_2['kelas_final']}**.

                                                    **Path node:** `{node_status_2['path_node']}`

                                                    Karena node sudah murni, entropy bernilai **{node_status_2['entropy']:.3f}**
                                                    dan proses rekursif tidak perlu dilanjutkan.
                                                    """
                                                )

                                                st.dataframe(
                                                    node_status_2["distribusi_df"],
                                                    use_container_width=True,
                                                    hide_index=True
                                                )

                                        else:

                                            # =============================
                                            # RINGKASAN NODE & RANKING GAIN LEVEL 2
                                            # =============================
                                            next_attr_2 = recursive_2["best_attr"]

                                            # st.markdown("### 📊 Ringkasan Node & Ranking Information Gain Level 2")

                                            target_level_2 = recursive_2["subset_df"][target_Y].astype(str).str.lower()

                                            jumlah_data_level_2 = len(recursive_2["subset_df"])
                                            jumlah_berat_level_2 = (target_level_2 == "berat").sum()
                                            jumlah_ringan_level_2 = (target_level_2 == "ringan").sum()
                                            entropy_level_2 = recursive_2["entropy"]

                                            df_ringkasan_level_2 = pd.DataFrame({
                                                "Komponen": [
                                                    "Jumlah Data",
                                                    "Jumlah Berat",
                                                    "Jumlah Ringan",
                                                    "Entropy Node"
                                                ],
                                                "Nilai": [
                                                    jumlah_data_level_2,
                                                    jumlah_berat_level_2,
                                                    jumlah_ringan_level_2,
                                                    f"{entropy_level_2:.3f}"
                                                ]
                                            })

                                            col_summary_2, col_rank_2 = st.columns([0.9, 1.6])

                                            with col_summary_2:
                                                st.markdown("#### 🧾 Ringkasan Node Level 2")

                                                st.dataframe(
                                                    df_ringkasan_level_2,
                                                    use_container_width=True,
                                                    hide_index=True
                                                )

                                                st.caption(
                                                    "Ringkasan ini menunjukkan kondisi data pada node Level 2 sebelum pemilihan node berikutnya."
                                                )

                                            with col_rank_2:
                                                st.markdown("#### 📈 Ranking Information Gain")

                                                gain_df_2_tampil = recursive_2["gain_df"].copy()
                                                gain_df_2_tampil["Information Gain"] = gain_df_2_tampil["Information Gain"].apply(
                                                    lambda x: f"{x:.3f}"
                                                )

                                                st.dataframe(
                                                    gain_df_2_tampil,
                                                    use_container_width=True,
                                                    hide_index=True
                                                )

                                            st.success(
                                                f"Node berikutnya: **{next_attr_2}**"
                                            )

                                            # =============================
                                            # DETAIL PERHITUNGAN LEVEL 2
                                            # =============================
                                            st.markdown("### 🔍 Detail Perhitungan per Atribut Level 2")

                                            atribut_detail_2_list = recursive_2["gain_df"]["Atribut"].tolist()

                                            default_index_2 = (
                                                atribut_detail_2_list.index(next_attr_2)
                                                if next_attr_2 in atribut_detail_2_list
                                                else 0
                                            )

                                            if SHOW_DETAIL_DROPDOWN_LEVEL_2:
                                                atribut_detail_2 = st.selectbox(
                                                    "Pilih Atribut untuk Detail Perhitungan Level 2",
                                                    atribut_detail_2_list,
                                                    index=default_index_2,
                                                    key=f"detail_level_2_{root_attr}_{root_value}_{next_attr_1}_{next_value_1}"
                                                )
                                            else:
                                                atribut_detail_2 = next_attr_2
                                                st.caption(
                                                    f"Detail perhitungan Level 2 otomatis menampilkan atribut dengan "
                                                    f"Information Gain tertinggi, yaitu **{next_attr_2}**."
                                                )

                                            df_detail_2 = recursive_2["detail_tables"][atribut_detail_2]

                                            gain_detail_2 = recursive_2["gain_df"].loc[
                                                recursive_2["gain_df"]["Atribut"] == atribut_detail_2,
                                                "Information Gain"
                                            ].iloc[0]

                                            col_detail_2, col_chart_2 = st.columns([1, 1])

                                            with col_detail_2:
                                                df_detail_2_tampil = df_detail_2.copy()

                                                st.dataframe(
                                                    df_detail_2_tampil,
                                                    use_container_width=True,
                                                    hide_index=True
                                                )

                                            with col_chart_2:
                                                fig, ax = plt.subplots(figsize=(5, 3.5))

                                                bars = ax.bar(
                                                    df_detail_2["Nilai"].astype(str),
                                                    df_detail_2["Entropy"],
                                                    color="#2563eb",
                                                    edgecolor="white"
                                                )

                                                ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)

                                                ax.set_title(
                                                    f"Entropy per Nilai: {atribut_detail_2}",
                                                    fontweight="bold"
                                                )

                                                ax.set_ylabel("Entropy")
                                                ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

                                                max_entropy_2 = df_detail_2["Entropy"].max()

                                                if max_entropy_2 == 0:
                                                    ax.set_ylim(0, 0.050)
                                                else:
                                                    ax.set_ylim(0, max_entropy_2 + 0.050)

                                                plt.xticks(rotation=30, ha="right", fontsize=8)
                                                plt.tight_layout()

                                                st.pyplot(fig)

                                            st.success(
                                                f"Information Gain ({atribut_detail_2}): **{gain_detail_2:.3f}**"
                                            )

                                            # st.caption(
                                            #     f"Catatan: dropdown ini hanya untuk melihat detail perhitungan setiap atribut. "
                                            #     f"Node berikutnya tetap dipilih berdasarkan Information Gain tertinggi, yaitu {next_attr_2}."
                                            # )

                                            # =============================
                                            # CEK APAKAH PERLU LANJUT KE LEVEL 3
                                            # =============================
                                            entropy_not_zero_level_3 = recursive_2[
                                                "detail_tables"
                                            ][next_attr_2]

                                            valid_values_level_3 = entropy_not_zero_level_3[
                                                entropy_not_zero_level_3["Entropy"] != 0
                                            ]["Nilai"].tolist()

                                            if len(valid_values_level_3) == 0:
                                                st.success(
                                                    f"""
                                                    🍃 Setelah node Level 2 dipecah berdasarkan atribut **{next_attr_2}**,
                                                    seluruh cabang sudah memiliki entropy 0.
                                                    Artinya setiap cabang sudah murni dan proses rekursif tidak perlu
                                                    dilanjutkan.
                                                    """
                                                )

                                            else:
                                                st.info(
                                                    f"""
                                                    Node pada **Level 2** belum sepenuhnya selesai karena masih terdapat
                                                    cabang pada atribut **{next_attr_2}** yang memiliki entropy lebih dari 0.

                                                    **Path node:** `{node_status_2['path_node']}`

                                                    Oleh karena itu, secara konsep algoritma ID3 masih dapat dilanjutkan
                                                    ke Level 3 pada cabang yang belum murni.
                                                    """
                                                )

                                                # =============================
                                                # LEVEL 3 - OPSIONAL / TERSEMBUNYI
                                                # =============================
                                                if SHOW_LEVEL_3:

                                                    st.markdown("---")
                                                    st.markdown("## 🌱 Level 3")

                                                    st.warning(
                                                        """
                                                        Level 3 bersifat opsional dan hanya ditampilkan jika masih ada cabang
                                                        yang belum murni pada hasil pemecahan Level 2.
                                                        """
                                                    )

                                                    next_value_2 = st.selectbox(
                                                        f"Pilih Cabang ({next_attr_2})",
                                                        valid_values_level_3,
                                                        key=f"level_3_branch_{root_attr}_{root_value}_{next_attr_1}_{next_value_1}_{next_attr_2}"
                                                    )

                                                    path_filters_level_3 = [
                                                        (root_attr, root_value),
                                                        (next_attr_1, next_value_1),
                                                        (next_attr_2, next_value_2)
                                                    ]

                                                    recursive_3 = get_recursive_node_info(
                                                        train_df,
                                                        fitur_X,
                                                        target_Y,
                                                        path_filters_level_3
                                                    )

                                                    if recursive_3 is not None:

                                                        st.markdown(
                                                            f"""
                                                            ### Path Node Level 3:
                                                            {root_attr} = {root_value}
                                                            → {next_attr_1} = {next_value_1}
                                                            → {next_attr_2} = {next_value_2}
                                                            """
                                                        )

                                                        st.success(
                                                            f"""
                                                            Entropy Node:
                                                            **{recursive_3['entropy']:.3f}**
                                                            """
                                                        )

                                                        # =============================
                                                        # CEK STATUS NODE LEVEL 3
                                                        # =============================
                                                        path_text_3 = (
                                                            f"{root_attr} = {root_value} → "
                                                            f"{next_attr_1} = {next_value_1} → "
                                                            f"{next_attr_2} = {next_value_2}"
                                                        )

                                                        node_status_3 = get_node_status_info(
                                                            df_node=recursive_3["subset_df"],
                                                            target_col=target_Y,
                                                            nama_level="Level 3",
                                                            path_node=path_text_3
                                                        )

                                                        if node_status_3["should_stop"]:

                                                            if node_status_3["is_empty"]:
                                                                st.warning(
                                                                    "Cabang ini tidak memiliki data, sehingga proses rekursif tidak dapat dilanjutkan."
                                                                )

                                                            elif node_status_3["is_pure"]:
                                                                st.success(
                                                                    f"🍃 {node_status_3['nama_level']} sudah menjadi leaf node / node akhir."
                                                                )

                                                                st.info(
                                                                    f"""
                                                                    Semua data pada node ini masuk ke kelas **{node_status_3['kelas_final']}**.

                                                                    **Path node:** `{node_status_3['path_node']}`

                                                                    Karena node sudah murni, entropy bernilai **{node_status_3['entropy']:.3f}**
                                                                    dan proses rekursif tidak perlu dilanjutkan ke level berikutnya.
                                                                    """
                                                                )

                                                                st.dataframe(
                                                                    node_status_3["distribusi_df"],
                                                                    use_container_width=True,
                                                                    hide_index=True
                                                                )

                                                        else:

                                                            st.markdown("### 📊 Ranking Information Gain Level 3")

                                                            gain_df_3_tampil = recursive_3["gain_df"].copy()
                                                            gain_df_3_tampil["Information Gain"] = gain_df_3_tampil["Information Gain"].apply(
                                                                lambda x: f"{x:.3f}"
                                                            )

                                                            st.dataframe(
                                                                gain_df_3_tampil,
                                                                use_container_width=True,
                                                                hide_index=True
                                                            )

                                                            next_attr_3 = recursive_3["best_attr"]

                                                            st.success(
                                                                f"""
                                                                Node berikutnya:
                                                                **{next_attr_3}**
                                                                """
                                                            )

                                                            # =============================
                                                            # DETAIL PERHITUNGAN LEVEL 3
                                                            # =============================
                                                            st.markdown("### 🔍 Detail Perhitungan per Atribut Level 3")

                                                            atribut_detail_3_list = recursive_3["gain_df"]["Atribut"].tolist()

                                                            default_index_3 = (
                                                                atribut_detail_3_list.index(next_attr_3)
                                                                if next_attr_3 in atribut_detail_3_list
                                                                else 0
                                                            )

                                                            atribut_detail_3 = st.selectbox(
                                                                "Pilih Atribut untuk Detail Perhitungan Level 3",
                                                                atribut_detail_3_list,
                                                                index=default_index_3,
                                                                key=f"detail_level_3_{root_attr}_{root_value}_{next_attr_1}_{next_value_1}_{next_attr_2}_{next_value_2}"
                                                            )

                                                            df_detail_3 = recursive_3["detail_tables"][atribut_detail_3]

                                                            gain_detail_3 = recursive_3["gain_df"].loc[
                                                                recursive_3["gain_df"]["Atribut"] == atribut_detail_3,
                                                                "Information Gain"
                                                            ].iloc[0]

                                                            col_detail_3, col_chart_3 = st.columns([1, 1])

                                                            with col_detail_3:
                                                                df_detail_3_tampil = format_tabel_detail_id3(df_detail_3)

                                                                st.dataframe(
                                                                    df_detail_3_tampil,
                                                                    use_container_width=True,
                                                                    hide_index=True
                                                                )

                                                            with col_chart_3:
                                                                fig, ax = plt.subplots(figsize=(5, 3.5))

                                                                bars =ax.bar(
                                                                    df_detail_3["Nilai"].astype(str),
                                                                    df_detail_3["Entropy"],
                                                                    color="#2563eb",
                                                                    edgecolor="white"
                                                                )

                                                                ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)

                                                                ax.set_title(
                                                                    f"Entropy per Nilai: {atribut_detail_3}",
                                                                    fontweight="bold"
                                                                )

                                                                ax.set_ylabel("Entropy")
                                                                ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
                                                                plt.xticks(rotation=30, ha="right", fontsize=8)
                                                                plt.tight_layout()

                                                                st.pyplot(fig)

                                                            st.success(
                                                                f"Information Gain ({atribut_detail_3}): **{gain_detail_3:.3f}**"
                                                            )

                                                            st.caption(
                                                                f"Catatan: Level 3 ini hanya tampilan opsional. "
                                                                f"Node berikutnya tetap dipilih berdasarkan Information Gain tertinggi, yaitu {next_attr_3}."
                                                            )

                # =============================
                # RUMUS ALGORITMA ID3
                # =============================
                st.markdown("---")

                with st.expander("📐 Lihat Rumus Algoritma ID3", expanded=False):
                    st.markdown("### Rumus Entropy")

                    st.markdown(
                        """
                        **Entropy(S)** = -Σ pᵢ × log₂(pᵢ)

                        Keterangan:

                        **S** = himpunan data atau dataset pada suatu node  
                        **pᵢ** = proporsi data pada kelas ke-i  
                        **i** = kelas target, yaitu Ringan dan Berat
                        """
                    )

                    st.markdown("---")

                    st.markdown("### Rumus Information Gain")

                    st.markdown(
                        """
                        **Gain(S, A)** = Entropy(S) - Σ ((|Sᵥ| / |S|) × Entropy(Sᵥ))

                        Keterangan:

                        **S** = himpunan data pada node yang sedang dihitung  
                        **A** = atribut yang diuji  
                        **v** = nilai yang terdapat pada atribut A  
                        **Sᵥ** = subset data yang memiliki nilai v pada atribut A  
                        **|Sᵥ|** = jumlah data pada subset Sᵥ  
                        **|S|** = jumlah seluruh data pada node S
                        """
                    )

                    st.info(
                        "Pada algoritma ID3, entropy digunakan untuk mengukur ketidakpastian kelas pada data. "
                        "Information gain digunakan untuk menentukan atribut terbaik. Atribut dengan nilai "
                        "information gain tertinggi dipilih sebagai node pada pohon keputusan."
                    )

                    # =============================
                    # CONTOH ANGKA INFORMATION GAIN ATRIBUT FASILITAS
                    # =============================
                    st.markdown("---")
                    st.markdown("### Contoh Perhitungan Information Gain Atribut Fasilitas")

                    hasil_contoh_fasilitas = get_information_gain_detail(
                        train_df,
                        "Fasilitas",
                        target_Y
                    )

                    df_contoh_fasilitas = hasil_contoh_fasilitas["detail_df"].copy()
                    gain_contoh_fasilitas = hasil_contoh_fasilitas["information_gain"]

                    df_contoh_tampil = df_contoh_fasilitas.copy()

                    st.dataframe(
                        df_contoh_tampil,
                        use_container_width=True,
                        hide_index=True
                    )

                    weighted_entropy_fasilitas = (
                        df_contoh_fasilitas["Proporsi"] * df_contoh_fasilitas["Entropy"]
                    ).sum()

                    daftar_bobot_entropy = []

                    for _, row in df_contoh_fasilitas.iterrows():
                        daftar_bobot_entropy.append(
                            f"({int(row['Jumlah'])}/{n} × {row['Entropy']:.3f})"
                        )

                    teks_bobot_entropy = " + ".join(daftar_bobot_entropy)

                    st.markdown(
                        f"""
                        **Substitusi ke rumus Information Gain atribut Fasilitas:**

                        **Gain(S, Fasilitas)** = Entropy(S) - Σ ((|Sᵥ| / |S|) × Entropy(Sᵥ))

                        **Gain(S, Fasilitas)** = {entropy_val:.3f} - [{teks_bobot_entropy}]

                        **Gain(S, Fasilitas)** = {entropy_val:.3f} - {weighted_entropy_fasilitas:.3f}

                        **Gain(S, Fasilitas)** = **{gain_contoh_fasilitas:.3f}**
                        """
                    )

                    st.success(
                        f"Berdasarkan contoh perhitungan tersebut, nilai Information Gain atribut "
                        f"**Fasilitas** adalah **{gain_contoh_fasilitas:.3f}**."
                    )

            with tab2:
                st.subheader("🌲 Struktur Pohon Keputusan")

                # =============================
                # VISUAL POHON LAMA (PNG)
                # =============================
                # img_path = visualize_tree(model_tree)
                # st.image(img_path, use_container_width=True)

                # =============================
                # VISUAL POHON BARU (SVG / GRAPHVIZ)
                # =============================
                st.graphviz_chart(
                    visualize_tree_dot(model_tree),
                    use_container_width=True
                )

                col_svg_id, col_svg_en = st.columns(2)
                with col_svg_id:
                    st.download_button(
                        label="⬇️ Download SVG (Indonesia)",
                        data=visualize_tree_svg_bytes(model_tree, lang="id"),
                        file_name="pohon_keputusan_model_utama_indonesia.svg",
                        mime="image/svg+xml"
                    )

                with col_svg_en:
                    st.download_button(
                        label="⬇️ Download SVG (English)",
                        data=visualize_tree_svg_bytes(model_tree, lang="en"),
                        file_name="decision_tree_main_model_english.svg",
                        mime="image/svg+xml"
                    )

                st.markdown("---")
                st.subheader("🖼️ Pohon Keputusan Versi PNG")
                col_tree_id, col_tree_en = st.columns(2)
                with col_tree_id:

                    st.download_button(
                        label="⬇️ Download PNG (Indonesia)",
                        data=visualize_tree_png_bytes(model_tree, lang="id"),
                        file_name="pohon_keputusan_model_utama_indonesia.png",
                        mime="image/png"
                    )
                with col_tree_en:
                    st.download_button(
                        label="⬇️ Download PNG (English)",
                        data=visualize_tree_png_bytes(model_tree, lang="en"),
                        file_name="decision_tree_main_model_english.png",
                        mime="image/png"
                    )

                # =============================
                # RINGKASAN ATRIBUT DAN CABANG POHON
                # =============================
                st.markdown("---")
                st.subheader("📌 Ringkasan Atribut dan Cabang Pohon Keputusan")

                df_tree_summary = get_tree_attribute_branch_summary(
                    model_tree,
                    train_df,
                    fitur_X
                )

                total_atribut = len(fitur_X)
                atribut_terbentuk = len(
                    df_tree_summary[
                        df_tree_summary["Status Node"] == "Terbentuk"
                    ]
                )
                atribut_tidak_terbentuk = len(
                    df_tree_summary[
                        df_tree_summary["Status Node"] == "Tidak Terbentuk"
                    ]
                )

                cabang_tidak_terbentuk = df_tree_summary[
                    "Cabang Tidak Terbentuk"
                ].sum()

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Total Atribut", total_atribut)
                col2.metric("Atribut Terbentuk", atribut_terbentuk)
                col3.metric("Atribut Tidak Terbentuk", atribut_tidak_terbentuk)
                col4.metric("Cabang Tidak Terbentuk", cabang_tidak_terbentuk)

                st.info(
                    """
                    Tabel ini menampilkan ringkasan atribut yang digunakan sebagai node
                    pada pohon keputusan ID3, sekaligus nilai cabang yang terbentuk dan
                    tidak terbentuk.

                    Kolom **Nilai Cabang Tidak Terbentuk** menunjukkan nilai atribut
                    yang ada pada data latih, tetapi tidak muncul sebagai cabang pada
                    pohon keputusan.
                    """
                )

                df_tree_summary_tampil = df_tree_summary.copy()
                df_tree_summary_tampil.insert(
                    0,
                    "No",
                    range(1, len(df_tree_summary_tampil) + 1)
                )

                st.dataframe(
                    df_tree_summary_tampil,
                    use_container_width=True,
                    hide_index=True
                )

                # =============================
                # ATURAN IF-THEN DARI POHON
                # =============================
                with st.expander("📜 Lihat Aturan IF-THEN dari Pohon Keputusan", expanded=False):

                    df_rules = extract_rules_from_tree(
                        model_tree,
                        target_name=target_Y
                    )

                    st.markdown("### 📋 Tabel Aturan Keputusan")

                    st.dataframe(
                        df_rules,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.markdown("### 🧾 Format IF-THEN")

                    for _, rule in df_rules.iterrows():
                        kelas = str(rule["Kelas"]).lower()

                        if kelas == "berat":
                            warna_bg = "#fef2f2"
                            warna_border = "#dc2626"
                            warna_text = "#991b1b"
                            icon = "🚨"
                        else:
                            warna_bg = "#f0fdf4"
                            warna_border = "#16a34a"
                            warna_text = "#166534"
                            icon = "✅"

                        html_rule = f"""
            <div style="background:{warna_bg};border:1px solid {warna_border};border-left:6px solid {warna_border};border-radius:12px;padding:14px 16px;margin-bottom:12px;box-shadow:0 1px 4px rgba(15,23,42,0.05);">
                <div style="font-weight:800;color:{warna_text};margin-bottom:8px;font-size:0.95rem;">
                    {icon} Rule {int(rule["No"])}
                </div>
                <div style="color:#0f172a;font-size:0.92rem;line-height:1.7;">
                    <b>IF</b> {rule["IF"]}<br>
                    <b>THEN</b>
                    <span style="background:#ffffff;color:{warna_text};padding:4px 10px;border-radius:999px;font-weight:800;">
                        {rule["THEN"]}
                    </span>
                </div>
            </div>
            """

                        st.markdown(
                            html_rule,
                            unsafe_allow_html=True
                        )

                    st.success(
                        f"Total aturan yang terbentuk dari pohon keputusan: **{len(df_rules)} aturan**."
                    )

        else:
            st.warning("⚠️ Silakan lakukan Data Preparation terlebih dahulu.")

    # =========================================================
    # 📋 EVALUATION
    # =========================================================
    elif tahap == "5. Evaluation":
        st.title("📋 Tahap 5 — Evaluation & Validation")
        st.caption("Menguji model dengan confusion matrix, akurasi, presisi, recall, F1-score, dan cross validation")
        st.markdown("---")

        if st.session_state.get("model_tree") is not None:
            import numpy as np
            model_tree = st.session_state.model_tree
            train_df = st.session_state.train_df
            test_df = st.session_state.test_df
            accuracy, precision, recall, f1, cm, y_test, y_pred = evaluate_model(test_df, model_tree)
            tn, fp, fn, tp = cm.ravel()

            # =============================
            # DASHBOARD METRIK EVALUASI UTAMA
            # =============================
            components.html(f"""
            <div style="
                font-family: Arial, sans-serif;
                display: grid;
                grid-template-columns: 1.45fr 1fr 1fr 1fr;
                gap: 14px;
                width: 100%;
                margin: 8px 0 20px 0;
            ">

                <div style="
                    background: linear-gradient(135deg, #1d4ed8, #2563eb);
                    border-radius: 16px;
                    padding: 22px 20px;
                    color: white;
                    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.28);
                    position: relative;
                    overflow: hidden;
                ">
                    <div style="
                        position: absolute;
                        top: 12px;
                        right: 14px;
                        background: rgba(255,255,255,0.18);
                        padding: 5px 10px;
                        border-radius: 999px;
                        font-size: 0.72rem;
                        font-weight: 800;
                    ">
                    </div>

                    <div style="font-size: 2rem; margin-bottom: 8px;">🎯</div>

                    <div style="
                        font-size: 0.82rem;
                        font-weight: 800;
                        letter-spacing: 0.06em;
                        text-transform: uppercase;
                        opacity: 0.9;
                    ">
                        F1-Score
                    </div>

                    <div style="
                        font-size: 2.45rem;
                        font-weight: 900;
                        margin-top: 6px;
                        line-height: 1;
                    ">
                        {f1*100:.2f}%
                    </div>

                    <div style="
                        font-size: 0.86rem;
                        margin-top: 12px;
                        line-height: 1.55;
                        opacity: 0.95;
                    ">
                        Digunakan sebagai acuan utama karena data kelas tidak seimbang
                        dan kelas Berat lebih dominan.
                    </div>
                </div>

                <div style="
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-top: 5px solid #16a34a;
                    border-radius: 16px;
                    padding: 18px 16px;
                    height: 170px;
                    box-sizing: border-box;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                ">
                    <div style="font-size: 1.6rem;">✅</div>
                    <div style="font-size: 0.78rem; color: #64748b; font-weight: 800; text-transform: uppercase; margin-top: 8px;">
                        Presisi
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: #16a34a; margin-top: 6px;">
                        {precision*100:.2f}%
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">
                        ketepatan prediksi Berat
                    </div>
                </div>

                <div style="
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-top: 5px solid #dc2626;
                    border-radius: 16px;
                    padding: 18px 16px;
                    height: 170px;
                    box-sizing: border-box;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                ">
                    <div style="font-size: 1.6rem;">📌</div>
                    <div style="font-size: 0.78rem; color: #64748b; font-weight: 800; text-transform: uppercase; margin-top: 8px;">
                        Recall
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: #dc2626; margin-top: 6px;">
                        {recall*100:.2f}%
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">
                        kemampuan mengenali Berat
                    </div>
                </div>

                <div style="
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-top: 5px solid #f97316;
                    border-radius: 16px;
                    padding: 18px 16px;
                    height: 170px;
                    box-sizing: border-box;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                ">
                    <div style="font-size: 1.6rem;">📊</div>
                    <div style="font-size: 0.78rem; color: #64748b; font-weight: 800; text-transform: uppercase; margin-top: 8px;">
                        Akurasi
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: #f97316; margin-top: 6px;">
                        {accuracy*100:.2f}%
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">
                        prediksi benar dari data uji
                    </div>
                </div>

            </div>
            """, height=250, scrolling=False)

            tab1, tab2 = st.tabs(["📏 Evaluasi Model","🔁 Cross Validation"])

            with tab1:
                st.subheader("Confusion Matrix")
                col_chart, col_tbl = st.columns([1,1])
                with col_chart:
                    fig, ax = plt.subplots(figsize=(5,4))
                    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ringan","Berat"]).plot(
                        ax=ax, colorbar=False, cmap="Blues")
                    ax.set_title(f"Confusion Matrix — Data Uji (n={len(test_df)})", fontweight="bold")
                    plt.tight_layout(); st.pyplot(fig)
                with col_tbl:
                    st.markdown("**Detail Confusion Matrix:**")
                    df_cm_detail = pd.DataFrame({
                        "Notasi":["TP","TN","FP","FN"],
                        "Arti": [
                            "Aktual Berat, Prediksi Berat",
                            "Aktual Ringan, Prediksi Ringan",
                            "Aktual Ringan, Prediksi Berat",
                            "Aktual Berat, Prediksi Ringan "
                        ],
                        "Nilai":[tp,tn,fp,fn]
                    })
                    st.dataframe(df_cm_detail, use_container_width=True, hide_index=True)
                    st.markdown("**Rumus Evaluasi:**")
                    df_rumus = pd.DataFrame({
                        "Metrik": [
                            "Akurasi",
                            "Presisi",
                            "Recall",
                            "F1-Score"
                        ],

                        "Rumus": [
                            "(TP+TN)/(TP+TN+FP+FN)",
                            "TP/(TP+FP)",
                            "TP/(TP+FN)",
                            "2×P×R/(P+R)"
                        ],

                        "Nilai": [
                            f"{accuracy*100:.2f}%",
                            f"{precision*100:.2f}%",
                            f"{recall*100:.2f}%",
                            f"{f1*100:.2f}%"
                        ]
                    })
                    st.dataframe(df_rumus, use_container_width=True, hide_index=True)

                root_attr = model_tree.get("attr","—") if isinstance(model_tree, dict) else "—"
                st.success(f"✅ **Kesimpulan:** Model ID3 berhasil mengklasifikasikan **{tp+tn} dari {len(test_df)}** data uji dengan benar. F1-score: **{f1    *100:.2f}%** | Root node paling berpengaruh: **{root_attr}**")

                # =============================
                # TABEL KLASIFIKASI DATA UJI
                # =============================
                st.markdown("---")
                st.subheader("📊 Tabel Klasifikasi Data Uji")

                df_eval = get_evaluation_detail(
                    test_df,
                    train_df,
                    model_tree
                )

                df_eval_tampil = df_eval.copy()

                df_eval_tampil.insert(
                    0,
                    "No Data Uji",
                    range(1, len(df_eval_tampil) + 1)
                )

                df_eval_tampil["Status Prediksi"] = df_eval_tampil["Status"].apply(
                    lambda x: "Benar" if x else "Salah"
                )

                kolom_eval = [
                    "No Data Uji",
                    "Fasilitas",
                    "Peralatan",
                    "Dampak Kerusakan",
                    "Penyebab",
                    "Kelompok Penyebab",
                    "Cuaca",
                    "Jenis Gangguan",
                    "Prediksi",
                    "Status Prediksi",
                    "Cabang Prediksi",
                    "Atribut Tidak Terbentuk",
                    "Nilai Atribut Tidak Terbentuk",
                    "Path Terakhir Terbentuk",
                    "Kelas Majority Fallback"
                ]

                df_eval_tampil = df_eval_tampil[
                    [kolom for kolom in kolom_eval if kolom in df_eval_tampil.columns]
                ]

                st.caption(
                    "Tabel di bawah ini menampilkan hasil klasifikasi data uji berdasarkan data aktual, "
                    "hasil prediksi model, status prediksi, serta informasi cabang prediksi."
                )

                search_eval = st.text_input(
                    "Cari data pada tabel klasifikasi data uji",
                    placeholder="Cari No Data, fasilitas, dampak kerusakan, aktual, prediksi, status, atau cabang...",
                    key="search_tabel_klasifikasi_data_uji"
                )

                df_eval_tampil_filtered = df_eval_tampil.copy()

                if search_eval:
                    mask_eval = df_eval_tampil_filtered.astype(str).apply(
                        lambda row: row.str.contains(
                            search_eval,
                            case=False,
                            na=False
                        ).any(),
                        axis=1
                    )

                    df_eval_tampil_filtered = df_eval_tampil_filtered[mask_eval]

                    st.caption(
                        f"Menampilkan {len(df_eval_tampil_filtered)} dari {len(df_eval_tampil)} data uji."
                    )

                def highlight_eval(row):
                    style = [""] * len(row)

                    cabang_prediksi = row.get("Cabang Prediksi", "")
                    status_prediksi = row.get("Status Prediksi", "")

                    # Warna kuning: cabang tidak terbentuk / majority fallback
                    if cabang_prediksi == "Tidak Terbentuk (Majority Fallback)":
                        style = ["background-color:#fef9c3;color:#713f12"] * len(row)

                    # Warna merah: prediksi salah
                    elif status_prediksi == "Salah":
                        style = ["background-color:#fee2e2;color:#991b1b"] * len(row)

                    # Warna hijau: prediksi benar
                    else:
                        style = ["background-color:#f0fdf4;color:#166534"] * len(row)

                    return style

                st.dataframe(
                    df_eval_tampil_filtered.style.apply(
                        highlight_eval,
                        axis=1
                    ),
                    use_container_width=True,
                    height=430,
                    hide_index=True
                )

                jumlah_salah = len(
                    df_eval_tampil[
                        df_eval_tampil["Status Prediksi"] == "Salah"
                    ]
                )

                jumlah_benar = len(
                    df_eval_tampil[
                        df_eval_tampil["Status Prediksi"] == "Benar"
                    ]
                )

                if "Cabang Prediksi" in df_eval_tampil.columns:
                    jumlah_fallback = len(
                        df_eval_tampil[
                            df_eval_tampil["Cabang Prediksi"] == "Tidak Terbentuk (Majority Fallback)"
                        ]
                    )
                else:
                    jumlah_fallback = 0

                st.info(
                    f"Dari {len(df_eval_tampil)} data uji, terdapat {jumlah_benar} data dengan prediksi benar "
                    f"dan {jumlah_salah} data dengan prediksi salah. Data yang menggunakan majority fallback "
                    f"berjumlah {jumlah_fallback} data."
                )


                # =============================
                # TABEL KHUSUS NILAI ATRIBUT TIDAK TERBENTUK DATA UJI
                # =============================
                st.markdown("---")
                st.subheader("🟨 Detail Nilai Atribut Tidak Terbentuk pada Data Uji")

                df_unformed_test = get_unformed_test_attribute_detail(
                    test_df,
                    model_tree
                )

                if df_unformed_test.empty:
                    st.success(
                        "Tidak ditemukan nilai atribut tidak terbentuk pada data uji. Semua data uji dapat melewati cabang pohon keputusan yang tersedia."
                    )

                else:
                    st.info(
                        """
                        Tabel ini menampilkan data uji yang memiliki nilai atribut tidak terbentuk
                        pada jalur pohon keputusan. Prediksi pada data tersebut tetap dihasilkan
                        menggunakan teknik majority fallback, yaitu mengambil kelas mayoritas
                        pada node terakhir yang masih sesuai dengan jalur keputusan.
                        """
                    )

                    kolom_unformed = [
                        "No Data Uji",
                        "Path Terakhir Terbentuk",
                        "Atribut Tidak Terbentuk",
                        "Nilai Atribut Tidak Terbentuk",
                        "Kelas Majority Fallback",
                        "Aktual",
                        "Prediksi",
                        "Status Prediksi",
                        "Fasilitas",
                        "Peralatan",
                        "Dampak Kerusakan",
                        "Penyebab",
                        "Kelompok Penyebab",
                        "Cuaca"
                    ]

                    df_unformed_test = df_unformed_test[
                        [kolom for kolom in kolom_unformed if kolom in df_unformed_test.columns]
                    ]

                    st.dataframe(
                        df_unformed_test.style.apply(
                            lambda row: ["background-color:#fef9c3;color:#713f12"] * len(row),
                            axis=1
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

            with tab2:
                st.subheader("🔁 Stratified 5-Fold Cross Validation")
                summary, cv_df, unseen_df, unseen_attr_df = cross_validation_id3_stratified(st.session_state.df_final)

                # =============================
                # DASHBOARD METRIK CROSS VALIDATION
                # =============================
                components.html(f"""
                <div style="
                    font-family: Arial, sans-serif;
                    display: grid;
                    grid-template-columns: 1.25fr 1fr 1fr 1fr;
                    gap: 10px;
                    width: 100%;
                    margin: 6px 0 14px 0;
                ">

                    <div style="
                        background: linear-gradient(135deg, #6d28d9, #7c3aed);
                        border-radius: 14px;
                        padding: 14px 16px;
                        color: white;
                        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.28);
                        position: relative;
                        overflow: hidden;
                    ">
                        <div style="
                            position: absolute;
                            top: 12px;
                            right: 14px;
                            background: rgba(255,255,255,0.18);
                            padding: 5px 10px;
                            border-radius: 999px;
                            font-size: 0.72rem;
                            font-weight: 800;
                        ">
                        </div>

                        <div style="font-size: 2rem; margin-bottom: 8px;">🎯</div>

                        <div style="
                            font-size: 0.82rem;
                            font-weight: 800;
                            letter-spacing: 0.06em;
                            text-transform: uppercase;
                            opacity: 0.9;
                        ">
                            F1-Score Cross Validation
                        </div>

                        <div style="
                            font-size: 1.95rem;
                            font-weight: 900;
                            margin-top: 6px;
                            line-height: 1;
                        ">
                            {summary['f1']*100:.2f}%
                        </div>

                        <div style="
                            font-size: 0.86rem;
                            margin-top: 12px;
                            line-height: 1.55;
                            opacity: 0.95;
                        ">
                            Rata-rata F1-Score dari 5 fold. Nilai ini lebih diprioritaskan
                            karena data kelas tidak seimbang.
                        </div>
                    </div>

                    <div style="
                        background: #ffffff;
                        border: 1px solid #e2e8f0;
                        border-top: 5px solid #16a34a;
                        border-radius: 14px;
                        padding: 12px 14px;
                        height: 150px;
                        box-sizing: border-box;
                        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                    ">
                        <div style="font-size: 1.6rem;">✅</div>
                        <div style="font-size: 0.78rem; color: #64748b; font-weight: 800; text-transform: uppercase; margin-top: 8px;">
                            Presisi CV
                        </div>
                        <div style="font-size: 1.45rem; font-weight: 900; color: #16a34a; margin-top: 6px;">
                            {summary['precision']*100:.2f}%
                        </div>
                        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">
                            rata-rata Presisi
                        </div>
                    </div>

                    <div style="
                        background: #ffffff;
                        border: 1px solid #e2e8f0;
                        border-top: 5px solid #dc2626;
                        border-radius: 14px;
                        padding: 12px 14px;
                        height: 150px;
                        box-sizing: border-box;
                        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                    ">
                        <div style="font-size: 1.6rem;">📌</div>
                        <div style="font-size: 0.78rem; color: #64748b; font-weight: 800; text-transform: uppercase; margin-top: 8px;">
                            Recall CV
                        </div>
                        <div style="font-size: 1.45rem; font-weight: 900; color: #dc2626; margin-top: 6px;">
                            {summary['recall']*100:.2f}%
                        </div>
                        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">
                            rata-rata recall
                        </div>
                    </div>

                    <div style="
                        background: #ffffff;
                        border: 1px solid #e2e8f0;
                        border-top: 5px solid #f97316;
                        border-radius: 14px;
                        padding: 12px 14px;
                        height: 150px;
                        box-sizing: border-box;
                        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
                    ">
                        <div style="font-size: 1.6rem;">📊</div>
                        <div style="font-size: 0.78rem; color: #64748b; font-weight: 800; text-transform: uppercase; margin-top: 8px;">
                            Akurasi CV
                        </div>
                        <div style="font-size: 1.45rem; font-weight: 900; color: #f97316; margin-top: 6px;">
                            {summary['accuracy']*100:.2f}%
                        </div>
                        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">
                            prediksi benar dari data uji
                        </div>
                    </div>

                </div>
                """, height=220, scrolling=False)
                st.info(
                        "Cross validation digunakan agar pengujian model lebih adil dan tidak hanya bergantung "
                        "pada satu pembagian data saja. Pada 5-fold cross validation, semua data mendapat giliran "
                        "menjadi data uji secara bergantian, sehingga hasil evaluasi model lebih mewakili "
                        "keseluruhan data."
                    )

                st.markdown("---")
                col_l, col_r = st.columns([1,1])
                
                with col_l:
                    st.subheader("🌳 Root Node Tiap Fold")
                    st.dataframe(cv_df[["Fold","Root"]], use_container_width=True, hide_index=True)
                    
                with col_r:
                    st.subheader("📊 Evaluasi per Fold")
                    df_detail_fold = cv_df[["Fold","Accuracy","Precision","Recall","F1"]].copy()
                    for col_m in ["Accuracy","Precision","Recall","F1"]:
                        df_detail_fold[col_m] = df_detail_fold[col_m].apply(lambda x: f"{x*100:.2f}%")
                    st.dataframe(df_detail_fold, use_container_width=True, hide_index=True)

                # =============================
                # POHON KEPUTUSAN PER FOLD
                # =============================
                st.markdown("---")
                st.subheader("🌲 Pohon Keputusan per Fold")

                selected_fold = st.selectbox(
                    "Pilih fold untuk melihat pohon keputusan:",
                    cv_df["Fold"].astype(int).tolist(),
                    key="selected_cv_fold_tree"
                )

                fold_tree_info = get_cross_validation_fold_tree(
                    st.session_state.df_final,
                    selected_fold=selected_fold,
                    n_splits=5
                )

                if fold_tree_info is not None:

                    selected_fold_row = cv_df[
                        cv_df["Fold"] == selected_fold
                    ].iloc[0]

                    components.html(f"""
                    <div style="
                        font-family: Arial, sans-serif;
                        background: #f8fafc;
                        border: 1px solid #dbeafe;
                        border-left: 6px solid #2563eb;
                        border-radius: 14px;
                        padding: 16px 18px;
                        margin: 10px 0 18px 0;
                        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.05);
                    ">
                        <div style="
                            font-size: 0.78rem;
                            color: #64748b;
                            font-weight: 800;
                            letter-spacing: 0.06em;
                            text-transform: uppercase;
                            margin-bottom: 10px;
                        ">
                            Ringkasan Pohon Fold {selected_fold}
                        </div>

                        <div style="
                            display: flex;
                            flex-wrap: wrap;
                            gap: 10px;
                            align-items: center;
                            margin-bottom: 10px;
                        ">
                            <span style="
                                background: #dbeafe;
                                color: #1e40af;
                                padding: 6px 12px;
                                border-radius: 999px;
                                font-weight: 800;
                                font-size: 0.85rem;
                            ">
                                Fold {selected_fold}
                            </span>

                            <span style="
                                background: #ede9fe;
                                color: #6d28d9;
                                padding: 6px 12px;
                                border-radius: 999px;
                                font-weight: 800;
                                font-size: 0.85rem;
                            ">
                                Root Node: {fold_tree_info["root"]}
                            </span>

                            <span style="
                                background: #f0fdf4;
                                color: #166534;
                                padding: 6px 12px;
                                border-radius: 999px;
                                font-weight: 800;
                                font-size: 0.85rem;
                            ">
                                Data Latih: {len(fold_tree_info["train_df"])}
                            </span>

                            <span style="
                                background: #fff7ed;
                                color: #9a3412;
                                padding: 6px 12px;
                                border-radius: 999px;
                                font-weight: 800;
                                font-size: 0.85rem;
                            ">
                                Data Uji: {len(fold_tree_info["test_df"])}
                            </span>

                            <span style="
                                background: #ecfeff;
                                color: #155e75;
                                padding: 6px 12px;
                                border-radius: 999px;
                                font-weight: 800;
                                font-size: 0.85rem;
                            ">
                                F1-Score: {selected_fold_row["F1"] * 100:.2f}%
                            </span>
                        </div>

                        <div style="
                            color: #475569;
                            font-size: 0.9rem;
                            line-height: 1.6;
                        ">
                            Pohon keputusan ini dibentuk menggunakan data latih pada Fold {selected_fold}.
                            Data uji pada fold ini digunakan untuk mengukur performa model, bukan untuk membentuk pohon.
                        </div>
                    </div>
                    """, height=150, scrolling=False)

                    st.graphviz_chart(
                        fold_tree_info["dot_source"],
                        use_container_width=True
                    )
                    
                    st.download_button(
                        label=f"⬇️ Download Pohon Fold {selected_fold} (SVG)",
                        data=fold_tree_info["svg_bytes"],
                        file_name=f"pohon_keputusan_fold_{selected_fold}.svg",
                        mime="image/svg+xml"
                    )

                    with st.expander(
                        f"📜 Lihat Aturan IF-THEN Fold {selected_fold}",
                        expanded=False
                    ):

                        df_rules_fold = extract_rules_from_tree(
                            fold_tree_info["model_tree"],
                            target_name=TARGET_Y
                        )

                        st.dataframe(
                            df_rules_fold,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.success(
                            f"Total aturan pada Fold {selected_fold}: **{len(df_rules_fold)} aturan**."
                        )
                
                st.markdown("---")
                st.subheader("📊 Ringkasan Data Latih & Uji per Fold")
                rows = []
                for _, fold in cv_df.iterrows():
                    fold_num = int(fold["Fold"]); gap_val = fold["Gap Entropy"]
                    rows.append({"Fold":fold_num,"Jenis Data":"Data Latih","Total":fold["Data Latih"],
                                 "Berat":fold["Latih Berat"],"Ringan":fold["Latih Ringan"],
                                 "Entropy": f"{fold['Entropy Latih']:.3f}","Gap Entropy":f"{gap_val:.3f} ({gap_val*100:.2f}%)"})
                    
                    rows.append({"Fold":"","Jenis Data":"Data Uji","Total":fold["Data Uji"],
                                 "Berat":fold["Uji Berat"],"Ringan":fold["Uji Ringan"],
                                 "Entropy": f"{fold['Entropy Uji']:.3f}","Gap Entropy":""})

                col_tbl, col_chart = st.columns([1,1])
                with col_tbl:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    
                with col_chart:
                    df_plot = cv_df[["Accuracy","Precision","Recall","F1"]].copy()
                    x = np.arange(len(cv_df)); width = 0.18
                    offsets = [-1.5,-0.5,0.5,1.5]
                    fig, ax = plt.subplots(figsize=(6,4))
                    for idx,(metric,color,label) in enumerate(zip(["Accuracy","Precision","Recall","F1"],
                            ["#2563eb","#16a34a","#dc2626","#d97706"],["Akurasi","Presisi","Recall","F1-Score"])):
                        bars = ax.bar(x+offsets[idx]*width, df_plot[metric], width, label=label, color=color, edgecolor="white")
                        ax.bar_label(bars, labels=[f"{v*100:.2f}%" for v in bars.datavalues], padding=2, fontsize=7, rotation=55)

                    ax.set_xticks(x)
                    ax.set_xticklabels([f"Fold {i+1}" for i in range(len(cv_df))])

                    ax.set_ylim(0, 1.55)
                    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))

                    ax.set_ylabel("Persentase")
                    ax.set_title("Metrik per Fold", fontweight="bold")

                    ax.legend(fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)

                df_final_cv = st.session_state.get("df_final")

                if df_final_cv is not None:
                    cv_split_detail_df = get_cross_validation_data_split_detail(
                        df_final_cv,
                        n_splits=5
                    )

                    st.markdown("#### 🧾 Detail Pembagian Data per Fold")

                    st.caption(
                        "Tabel di bawah ini merupakan data final setelah tahap data preparation beserta status pembagian data pada setiap fold."
                    )

                    search_cv_detail = st.text_input(
                        "Cari data pada tabel pembagian fold",
                        placeholder="Cari No Data, fasilitas, peralatan, dampak kerusakan, penyebab, atau target...",
                        key="search_detail_pembagian_fold"
                    )

                    cv_split_tampil = cv_split_detail_df.copy()

                    if search_cv_detail:
                        mask_cv_detail = cv_split_tampil.astype(str).apply(
                            lambda row: row.str.contains(
                                search_cv_detail,
                                case=False,
                                na=False
                            ).any(),
                            axis=1
                        )

                        cv_split_tampil = cv_split_tampil[mask_cv_detail]

                    def highlight_cv_status(val):
                        if val == "Uji":
                            return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"
                        elif val == "Latih":
                            return "background-color: #dcfce7; color: #166534;"
                        return ""

                    fold_cols = [f"Fold {i}" for i in range(1, 6)]

                    styled_cv_split = cv_split_tampil.style.map(
                        highlight_cv_status,
                        subset=fold_cols
                    )

                    st.dataframe(
                        styled_cv_split,
                        use_container_width=True,
                        height=430,
                        hide_index=True
                    )

                else:
                    st.warning("⚠️ Data final belum tersedia. Silakan lakukan Data Preparation terlebih dahulu.")
                
                # st.markdown("---")
                # st.subheader("📊 Visualisasi K-Fold Cross Validation")
                # n_folds = len(cv_df)
                # fig, ax = plt.subplots(figsize=(10,4)); ax.set_facecolor("#f8fafc")
                # for i in range(n_folds):
                #     f1_v = cv_df.iloc[i]["F1"]
                #     for j in range(n_folds):
                #         if j==i:
                #             ax.add_patch(plt.Rectangle((j,n_folds-i-1),1,0.75,color="#f59e0b",alpha=0.9,linewidth=0))
                #             ax.text(j+0.5,n_folds-i-1+0.375,f"Uji\nF1:{f1_v*100:.2f}%",ha="center",va="center",fontsize=8,fontweight="bold")
                #         else:
                #             ax.add_patch(plt.Rectangle((j,n_folds-i-1),1,0.75,color="#3b82f6",alpha=0.8,linewidth=0))
                #             ax.text(j+0.5,n_folds-i-1+0.375,"Latih",ha="center",va="center",fontsize=8,color="white")
                # for i in range(n_folds):
                #     ax.text(-0.35,n_folds-i-1+0.35,str(i+1),ha="center",va="center",fontsize=12,fontweight="bold",color="#1e3a5f")
                # ax.set_xlim(-0.6,n_folds); ax.set_ylim(0,n_folds); ax.axis("off")
                # plt.tight_layout(); st.pyplot(fig)

                st.markdown("---")
                
                col_u1, col_u2 = st.columns([1,1])
                with col_u1:
                    st.subheader("🚫 Cabang Unseen")
                    if unseen_df.empty: st.success("Tidak ada unseen value pada semua fold.")
                    else: st.dataframe(unseen_df, use_container_width=True, hide_index=True)
                    
                with col_u2:
                    st.subheader("📊 Atribut Sering Unseen")
                    if unseen_attr_df.empty: st.success("Tidak ada")
                    else: st.dataframe(unseen_attr_df, use_container_width=True, hide_index=True)

        else:
            st.warning("⚠️ Silakan lakukan Modeling terlebih dahulu.")

# =========================================================
# ⚙️ KLASIFIKASI GANGGUAN
# =========================================================
elif menu == "⚙️ Klasifikasi Gangguan":
    st.title("📊 Klasifikasi Gangguan Berdasarkan Model Decision Tree ID3")
    st.caption(
        "Menu ini menggunakan model Decision Tree ID3 yang telah dibangun pada tahap Modeling "
        "untuk memprediksi tingkat gangguan berdasarkan kombinasi nilai atribut input."
    )
    st.markdown("---")

    if "model_tree" in st.session_state and "df_final" in st.session_state:
        model_tree = st.session_state.model_tree
        fitur_X = FITUR_X.copy()

        st.info("Pilih nilai untuk setiap atribut di bawah, lalu klik tombol **Klasifikasi**.")

        input_user = {}
        cols_input = st.columns(2)
        
        for i, f in enumerate(fitur_X):
            opsi = sorted(st.session_state.df_final[f].unique())
            with cols_input[i % 2]:
                input_user[f] = st.selectbox(f"**{f}**", opsi)

        st.markdown("")
        
        if st.button("🔍 Klasifikasi Gangguan"):
            hasil = predict_single_row(input_user, model_tree)
            indikator_risiko = cek_indikator_risiko(input_user)
            
            if hasil.lower() == "berat":
                st.markdown(f"""
                <div style="
                    background:#fef2f2;
                    border:2px solid #dc2626;
                    border-radius:12px;
                    padding:20px;
                    text-align:center;
                    margin-top:16px;
                ">
                    <div style="font-size:2.5rem;">🚨</div>
                    <div style="
                        font-size:1.4rem;
                        font-weight:800;
                        color:#dc2626;
                        margin-top:8px;
                    ">
                        Hasil Prediksi Model ID3: BERAT
                    </div>
                    <div style="
                        color:#7f1d1d;
                        margin-top:6px;
                        font-size:0.9rem;
                        line-height:1.6;
                    ">
                        Berdasarkan model ID3, kombinasi atribut ini diprediksi sebagai gangguan berat.
                        Hasil ini dapat digunakan sebagai pendukung keputusan awal dalam penentuan tim teknis.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown(f"""
                <div style="
                    background:#f0fdf4;
                    border:2px solid #16a34a;
                    border-radius:12px;
                    padding:20px;
                    text-align:center;
                    margin-top:16px;
                ">
                    <div style="font-size:2.5rem;">✅</div>
                    <div style="
                        font-size:1.4rem;
                        font-weight:800;
                        color:#16a34a;
                        margin-top:8px;
                    ">
                        Hasil Prediksi Model ID3: RINGAN
                    </div>
                    <div style="
                        color:#14532d;
                        margin-top:6px;
                        font-size:0.9rem;
                        line-height:1.6;
                    ">
                        Berdasarkan model ID3, kombinasi atribut ini diprediksi sebagai gangguan ringan.
                        Hasil ini digunakan sebagai pendukung keputusan awal dan tetap dapat diverifikasi
                        dengan kondisi aktual di lapangan.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if hasil.lower() == "ringan" and indikator_risiko:
                indikator_teks = ", ".join(indikator_risiko)

                st.markdown(f"""
                <div style="
                    background:#fff7ed;
                    border:2px solid #f97316;
                    border-radius:12px;
                    padding:16px 18px;
                    margin-top:14px;
                    color:#7c2d12;
                    line-height:1.6;
                ">
                    <div style="
                        font-weight:800;
                        font-size:1rem;
                        margin-bottom:6px;
                    ">
                        ⚠️ Catatan Risiko Operasional
                    </div>
                    <div style="font-size:0.9rem;">
                        Model memprediksi gangguan sebagai <b>Ringan</b>. Namun, input mengandung
                        indikator risiko berupa <b>{indikator_teks}</b>. Oleh karena itu, hasil prediksi
                        sebaiknya tetap diverifikasi oleh petugas atau Command Center sebelum dijadikan
                        dasar penugasan akhir.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            background:#fefce8;
            border:1px dashed #ca8a04;
            border-radius:10px;
            padding:2rem;
            text-align:center;
            color:#92400e;
        ">
            <div style="font-size:2rem;">⚠️</div>
            <div style="
                font-weight:700;
                font-size:1.05rem;
                margin:8px 0;
            ">
                Silakan selesaikan tahapan Modeling terlebih dahulu sebelum melakukan klasifikasi.
            </div>
            <div style="font-size:0.85rem;">
                Navigasi → 📚 Tahapan → 4. Modeling
            </div>
        </div>
        """, unsafe_allow_html=True)