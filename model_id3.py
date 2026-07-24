# model_id3.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score ,confusion_matrix
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from graphviz import Digraph

# ---------------------------------------------------------------------------
# 1. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def tentukan_jenis_gangguan(kode_regu):
    if pd.isna(kode_regu):
        return None
    str_kode = str(kode_regu).strip().upper()
    angka_str = "".join(filter(str.isdigit, str_kode))
    if not angka_str:
        return None
    angka_utama = int(angka_str[0])
    if angka_utama == 7:
        return "Ringan"
    elif angka_utama == 8:
        return "Berat"
    else:
        return None


# ---------------------------------------------------------------------------
# 2. DATA PREPARATION
# ---------------------------------------------------------------------------
FITUR_X = [
    "Fasilitas",
    "Peralatan",
    "Dampak Kerusakan",
    "Penyebab",
    "Kelompok Penyebab",
    "Cuaca",
]

TARGET_Y = "Jenis Gangguan"

def get_preparation_steps(df):
    df_before = df.copy()

    # 1. Hapus duplikat berdasarkan No Laporan
    df_no_dup = df.drop_duplicates(subset=["No Laporan"]).copy()

    # 2. Feature engineering: membuat label target
    df_no_dup[TARGET_Y] = df_no_dup["Nama Regu"].apply(tentukan_jenis_gangguan)

    # 3. Hapus data yang tidak memiliki target
    df_clean_pre = df_no_dup.dropna(subset=[TARGET_Y]).copy()

    # 4. Seleksi fitur dan target untuk modeling
    df_final = df_clean_pre[FITUR_X + [TARGET_Y]].copy()

    # 5. case folding data kategorikal
    for kolom in FITUR_X:
        df_final[kolom] = df_final[kolom].astype(str).str.lower()

    return {
        "df_before": df_before,
        "df_no_dup": df_no_dup,
        "df_clean_pre": df_clean_pre,
        "df_final": df_final,
        "fitur_X": FITUR_X,
        "target_Y": TARGET_Y
    }

def preprocess_data(df):
    prep = get_preparation_steps(df)
    return prep["df_before"], prep["df_final"]

# ---------------------------------------------------------------------------
# 3. SPLIT DATA
# ---------------------------------------------------------------------------
def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Jenis Gangguan"])
    return train_df, test_df

# ---------------------------------------------------------------------------
# SUMMARY SPLIT (UNTUK GUI)
# ---------------------------------------------------------------------------
def get_split_summary(train_df, test_df, target="Jenis Gangguan"):

    def summary(df):
        total = len(df)
        jumlah_berat = len(df[df[target] == "Berat"])
        jumlah_ringan = len(df[df[target] == "Ringan"])
        entropy = calculate_entropy(df[target])

        return {
            "Total Data": total,
            "Jumlah Berat": jumlah_berat,
            "Jumlah Ringan": jumlah_ringan,
            "Entropy": round(entropy, 3)
        }

    train_summary = summary(train_df)
    test_summary = summary(test_df)

    # GAP (selisih)
    gap = {
        "Gap Berat": abs(train_summary["Jumlah Berat"] - test_summary["Jumlah Berat"]),
        "Gap Ringan": abs(train_summary["Jumlah Ringan"] - test_summary["Jumlah Ringan"]),
        "Gap Entropy": round(abs(train_summary["Entropy"] - test_summary["Entropy"]), 3)
    }

    return train_summary, test_summary, gap


# ---------------------------------------------------------------------------
# 4. ID3
# ---------------------------------------------------------------------------
def calculate_entropy(data_column):
    counts = data_column.value_counts()
    probabilities = counts / len(data_column)
    entropy = 0
    for prob in probabilities:
        if prob > 0:
            entropy -= prob * np.log2(prob)
    return entropy


def calculate_information_gain(data, feature, target):
    total_entropy = calculate_entropy(data[target])
    weighted_entropy = 0

    for value in data[feature].unique():
        subset = data[data[feature] == value]
        prob = len(subset) / len(data)
        weighted_entropy += prob * calculate_entropy(subset[target])

    return total_entropy - weighted_entropy


def build_id3_tree(data, features, target):
    
    majority_class = data[target].mode()[0]

    if len(data) == 0:
        return majority_class

    if len(data[target].unique()) == 1:
        return data[target].iloc[0]

    if len(features) == 0:
        return data[target].mode()[0]

    gains = [calculate_information_gain(data, f, target) for f in features]
    best_feature = features[np.argmax(gains)]

    tree = {
        "attr": best_feature,
        "child": {},
        "majority": majority_class
    }
    remaining_features = [f for f in features if f != best_feature]

    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        subtree = build_id3_tree(subset, remaining_features, target)
        tree["child"][value] = subtree

    return tree

# menyimpan perhitungan model
def get_model_info(data, features, target):
    info = {}

    # Entropy root
    total_entropy = calculate_entropy(data[target])
    info["entropy"] = total_entropy

    # Gain tiap atribut
    gains = {}
    for f in features:
        gains[f] = calculate_information_gain(data, f, target)

    info["gains"] = gains

    # Root node
    best_feature = max(gains, key=gains.get)
    info["root"] = best_feature

    return info

# ---------------------------------------------------------------------------
# DETAIL INFORMATION GAIN (UNTUK GUI)
# ---------------------------------------------------------------------------
def get_information_gain_detail(train_df, atribut, target_Y):

    total_data = len(train_df)

    rows_detail = []

    for value in train_df[atribut].unique():

        subset = train_df[
            train_df[atribut] == value
        ]

        jumlah = len(subset)

        jumlah_berat = len(
            subset[subset[target_Y] == "Berat"]
        )

        jumlah_ringan = len(
            subset[subset[target_Y] == "Ringan"]
        )

        entropy = calculate_entropy(
            subset[target_Y]
        )

        rows_detail.append({
            "Nilai": value,
            "Jumlah": jumlah,
            "Berat": jumlah_berat,
            "Ringan": jumlah_ringan,
            "Entropy": round(entropy, 3),
            "Proporsi": round(jumlah / total_data, 3)
        })

    detail_df = pd.DataFrame(rows_detail)

    information_gain = calculate_information_gain(
        train_df,
        atribut,
        target_Y
    )

    entropy_root = calculate_entropy(
        train_df[target_Y]
    )

    return {
        "detail_df": detail_df,
        "information_gain": information_gain,
        "entropy_root": entropy_root
    }


# ---------------------------------------------------------------------------
# RANKING INFORMATION GAIN (UNTUK GUI)
# ---------------------------------------------------------------------------
def get_information_gain_ranking(train_df, fitur_X, target_Y):

    ig_all = []

    for f in fitur_X:

        ig_f = calculate_information_gain(
            train_df,
            f,
            target_Y
        )

        ig_all.append({
            "Atribut": f,
            "Information Gain": round(ig_f, 3)
        })

    ig_df = pd.DataFrame(ig_all)

    ig_df = ig_df.sort_values(
        "Information Gain",
        ascending=False
    ).reset_index(drop=True)

    return ig_df

# ---------------------------------------------------------------------------
# RECURSIVE NODE EXPLORER (UNTUK GUI)
# ---------------------------------------------------------------------------
def get_recursive_node_info(
    train_df,
    fitur_X,
    target_Y,
    path_filters
):

    # =============================
    # FILTER DATA BERDASARKAN PATH
    # =============================
    subset_df = train_df.copy()

    for attr, val in path_filters:
        subset_df = subset_df[
            subset_df[attr] == val
        ]

    # =============================
    # ENTROPY NODE
    # =============================
    entropy_node = calculate_entropy(
        subset_df[target_Y]
    )

    # =============================
    # FITUR YANG BELUM DIGUNAKAN
    # =============================
    used_features = [
        attr for attr, _ in path_filters
    ]

    remaining_features = [
        f for f in fitur_X
        if f not in used_features
    ]

    # kalau fitur habis
    if len(remaining_features) == 0:
        return None

    # =============================
    # HITUNG GAIN
    # =============================
    gain_rows = []
    detail_tables = {}

    for fitur in remaining_features:

        rows = []

        for value in subset_df[fitur].unique():

            child_subset = subset_df[
                subset_df[fitur] == value
            ]

            jumlah = len(child_subset)

            berat = len(
                child_subset[
                    child_subset[target_Y] == "Berat"
                ]
            )

            ringan = len(
                child_subset[
                    child_subset[target_Y] == "Ringan"
                ]
            )

            entropy = calculate_entropy(
                child_subset[target_Y]
            )

            rows.append({
                "Nilai": value,
                "Jumlah": jumlah,
                "Berat": berat,
                "Ringan": ringan,
                "Entropy": round(entropy, 3)
            })

        detail_tables[fitur] = pd.DataFrame(rows)

        ig = calculate_information_gain(
            subset_df,
            fitur,
            target_Y
        )

        gain_rows.append({
            "Atribut": fitur,
            "Information Gain": round(ig, 3)
        })

    gain_df = pd.DataFrame(gain_rows)

    gain_df = gain_df.sort_values(
        "Information Gain",
        ascending=False
    ).reset_index(drop=True)

    best_attr = gain_df.iloc[0]["Atribut"]

    return {
        "subset_df": subset_df,
        "entropy": round(entropy_node, 3),
        "gain_df": gain_df,
        "best_attr": best_attr,
        "detail_tables": detail_tables
    }

# ---------------------------------------------------------------------------
# STATUS NODE / LEAF NODE (UNTUK GUI)
# ---------------------------------------------------------------------------
def get_node_status_info(df_node, target_col, nama_level, path_node):
    jumlah_data = len(df_node)

    if jumlah_data == 0:
        return {
            "should_stop": True,
            "is_empty": True,
            "is_pure": False,
            "status": "Node Kosong",
            "nama_level": nama_level,
            "path_node": path_node,
            "kelas_final": None,
            "jumlah_data": 0,
            "entropy": 0,
            "distribusi_df": pd.DataFrame()
        }

    distribusi = df_node[target_col].value_counts()

    distribusi_df = distribusi.reset_index()
    distribusi_df.columns = ["Kelas", "Jumlah Data"]

    entropy_node = calculate_entropy(df_node[target_col])

    if len(distribusi) == 1:
        kelas_final = distribusi.index[0]

        return {
            "should_stop": True,
            "is_empty": False,
            "is_pure": True,
            "status": "Murni / Leaf Node",
            "nama_level": nama_level,
            "path_node": path_node,
            "kelas_final": kelas_final,
            "jumlah_data": jumlah_data,
            "entropy": round(entropy_node, 3),
            "distribusi_df": distribusi_df
        }

    return {
        "should_stop": False,
        "is_empty": False,
        "is_pure": False,
        "status": "Belum Murni",
        "nama_level": nama_level,
        "path_node": path_node,
        "kelas_final": None,
        "jumlah_data": jumlah_data,
        "entropy": round(entropy_node, 3),
        "distribusi_df": distribusi_df
    }

# ---------------------------------------------------------------------------
# VISUALISASI POHON (GRAPHVIZ)
# ---------------------------------------------------------------------------
TREE_LABEL_MAP_EN = {
    # =========================
    # NAMA ATRIBUT
    # =========================
    "Fasilitas": "Facilities",
    "Peralatan": "Equipment",
    "Dampak Kerusakan": "Damage Impact",
    "Penyebab": "Cause",
    "Kelompok Penyebab": "Cause Group",
    "Cuaca": "Weather",

    # =========================
    # KELAS / LEAF
    # =========================
    "Ringan": "Light",
    "Berat": "Severe",

    # =========================
    # FASILITAS
    # =========================
    "pelanggan": "customer",
    "sambungan tenaga listrik dan app": "service connection and app",
    "jtr": "low-voltage network (jtr)",

    # =========================
    # PERALATAN
    # =========================
    "app": "app",
    "periksa meter": "meter inspection",
    "kabel sr": "service cable",
    "phb tr": "low-voltage distribution panel (phb-tr)",
    "mv cell": "mv cell",
    "informasi salah": "incorrect information",
    "iml": "iml",
    "kabel jtr": "low-voltage cable",

    # =========================
    # DAMPAK KERUSAKAN
    # =========================
    "drop tegangan": "voltage drop",
    "mcb rusak": "damaged mcb",
    "ct terbakar / rusak": "burned / damaged ct",
    "kabel sr rusak": "damaged service cable",
    "terminasi rusak": "damaged termination",
    "nh fuse rusak": "damaged nh fuse",
    "busbar coupler terbakar /rusak": "burned / damaged busbar coupler",
    "busbar coupler terbakar / rusak": "burned / damaged busbar coupler",
    "instalasi milik pelanggan rusak": "damaged customer-owned installation",
    "konektor tr rusak": "damaged low-voltage connector",
    "meter terbakar / rusak": "burned / damaged meter",
    "gangguan tidak diketahui": "unknown disturbance",
    "clear tamper": "clear tamper",
    "kabel rusak": "damaged cable",
    "kabel jtr rusak": "damaged low-voltage cable",
    "ct metering rusak": "damaged metering ct",

    # =========================
    # PENYEBAB
    # =========================
    "sambungan kendor / loss kontak": "loose connection / loss of contact",
    "overload": "overload",
    "lifetime": "lifetime",
    "dalam investigasi": "under investigation",
    "kesalahan procedure": "procedure error",
    "kesalahan pemasangan": "installation error",
    "binatang": "animal",
    "pekerjaan konstruksi pihak ketiga": "third-party construction work",
    "kebakaran": "fire",
    "pohon": "tree",

    # =========================
    # KELOMPOK PENYEBAB
    # =========================
    "kesalahan desain": "design error",
    "kesalahan operasional": "operational error",
    "kesalahan pemeliharaan": "maintenance error",
    "tersentuh benda asing": "contact with foreign object",
    "kesalahan kontruksi": "construction error",
    "kesalahan konstruksi": "construction error",
    "mutu material tidak standar": "substandard material quality",
    "publik": "public",
    "bencana alam / musibah": "natural disaster / calamity",

    # =========================
    # CUACA
    # =========================
    "mendung": "cloudy",
    "hujan": "rain",
    "cerah": "clear",
    "angin kencang": "strong wind",
}

def translate_tree_text(text, lang="id"):
    text_str = str(text)

    if lang == "en":
        return TREE_LABEL_MAP_EN.get(
            text_str,
            TREE_LABEL_MAP_EN.get(text_str.lower(), text_str)
        )

    return text_str

def build_graphviz_tree(tree_dict, dot, counter, parent_id=None, branch_label="", lang="id"):

    # LEAF NODE
    if not isinstance(tree_dict, dict):
        node_id = f"leaf_{counter[0]}"
        leaf_label = translate_tree_text(tree_dict, lang)

        dot.node(
            node_id,
            label=str(leaf_label),
            shape="box",
            style="filled",
            fillcolor="lightblue",
        )

        if parent_id:
            edge_label = translate_tree_text(branch_label, lang)
            dot.edge(parent_id, node_id, label=str(edge_label))

        counter[0] += 1

    # DECISION NODE
    else:
        attribute = tree_dict.get("attr", "Leaf")
        attribute_label = translate_tree_text(attribute, lang)

        node_id = f"node_{counter[0]}"
        dot.node(
            node_id,
            label=str(attribute_label),
            shape="oval",
            style="filled",
            fillcolor="lightgray",
        )

        if parent_id:
            edge_label = translate_tree_text(branch_label, lang)
            dot.edge(parent_id, node_id, label=str(edge_label))

        counter[0] += 1

        for value, subtree in tree_dict["child"].items():
            build_graphviz_tree(
                subtree,
                dot,
                counter,
                parent_id=node_id,
                branch_label=str(value),
                lang=lang,
            )

##########################
# Visualisasi Pohon ID3 #
##########################
def visualize_tree(model_tree):
    dot = Digraph(graph_attr={"rankdir": "LR"})  # kiri ke kanan biar lebih pendek
    counter = [0]

    build_graphviz_tree(model_tree, dot, counter)

    # Simpan ke PNG
    filename = "tree_id3"
    dot.render(filename, format="png", cleanup=True)
    
    return filename + ".png"

# ---------------------------------------------------------------------------
# VISUALISASI POHON TANPA SIMPAN FILE (UNTUK STREAMLIT GRAPHVIZ)
# ---------------------------------------------------------------------------
def visualize_tree_dot(model_tree, lang="id"):
    dot = Digraph(graph_attr={"rankdir": "LR"})
    counter = [0]

    build_graphviz_tree(model_tree, dot, counter, lang=lang)

    return dot.source

# ---------------------------------------------------------------------------
# VISUALISASI POHON SVG TANPA SIMPAN FILE
# ---------------------------------------------------------------------------
# ini untuk pohon normal tidak lebar
# def visualize_tree_svg_bytes(model_tree, lang="id"):
#     dot = Digraph(graph_attr={
#         "rankdir": "LR",
#         "dpi": "400",
#         "bgcolor": "white",
#         "pad": "0.35"
#     })
#     counter = [0]

#     build_graphviz_tree(model_tree, dot, counter, lang=lang)

#     return dot.pipe(format="svg")

def visualize_tree_svg_bytes(model_tree, lang="id"):
    dot = Digraph(
        graph_attr={
            "rankdir": "LR",
            "bgcolor": "white",
            "pad": "0.10",

            # Cabang dalam level yang sama dirapatkan secara vertikal
            "nodesep": "0.02",

            # Jarak antarlevel diperbesar secara horizontal
            "ranksep": "1.80",

            "splines": "spline",
            "outputorder": "edgesfirst"
        },
        node_attr={
            "fontsize": "10",
            "margin": "0.06,0.04"
        },
        edge_attr={
            "fontsize": "9"
        }
    )

    counter = [0]

    build_graphviz_tree(
        model_tree,
        dot,
        counter,
        lang=lang
    )

    return dot.pipe(format="svg")

def visualize_tree_png_bytes(model_tree, lang="id"):
    dot = Digraph(graph_attr={
        "rankdir": "LR",
        "dpi": "400",
        "bgcolor": "white",
        "pad": "0.35"
    })
    counter = [0]

    build_graphviz_tree(model_tree, dot, counter, lang=lang)

    return dot.pipe(format="png")

# ---------------------------------------------------------------------------
# 5. TRAIN MODEL
# ---------------------------------------------------------------------------
def train_model(train_df):

    fitur_X = [
        "Fasilitas",
        "Peralatan",
        "Dampak Kerusakan",
        "Penyebab",
        "Kelompok Penyebab",
        "Cuaca",
    ]
    target_Y = "Jenis Gangguan"

    model_tree = build_id3_tree(train_df, fitur_X, target_Y)
    return model_tree


# ---------------------------------------------------------------------------
# 6. PREDIKSI
# ---------------------------------------------------------------------------
def predict_single_row(row, tree):
    if not isinstance(tree, dict):
        return tree

    attribute = tree.get("attr")
    if attribute is None:
        return tree.get("majority")
    
    value = row[attribute]

    if value in tree["child"]:
        return predict_single_row(row, tree["child"][value])
    else:
        return tree["majority"]


def predict(X, tree):
    return [predict_single_row(row, tree) for _, row in X.iterrows()]


# ---------------------------------------------------------------------------
# 7. EVALUASI
# ---------------------------------------------------------------------------
# def evaluate_model(test_df, model_tree):
def evaluate_model(test_df, model_tree):

    target_Y = "Jenis Gangguan"
    X_test = test_df.drop(target_Y, axis=1)
    y_test = test_df[target_Y]

    # prediksi asli
    y_pred = predict(X_test, model_tree)
    
    y_pred_fixed = y_pred

    # metrik
    accuracy = accuracy_score(y_test, y_pred_fixed)
    precision = precision_score(y_test, y_pred_fixed, pos_label="Berat", zero_division=0)
    recall = recall_score(y_test, y_pred_fixed, pos_label="Berat", zero_division=0)
    cm = confusion_matrix(y_test, y_pred_fixed, labels=["Ringan", "Berat"])
    f1 = f1_score(y_test, y_pred_fixed, pos_label="Berat", zero_division=0)
    
    return accuracy, precision, recall, f1, cm, y_test, y_pred_fixed

# ---------------------------------------------------------------------------
# 8. Cross Validation
# ---------------------------------------------------------------------------
def cross_validation_id3_stratified(df, n_splits=5, export=False):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fitur_X = [
        "Fasilitas", "Peralatan", "Dampak Kerusakan",
        "Penyebab", "Kelompok Penyebab", "Cuaca",
    ]
    target_Y = "Jenis Gangguan"

    X = df[fitur_X]
    y = df[target_Y]

    accuracies, precisions, recalls, f1_scores = [], [], [], []
    fold_results = []

    unseen_counter = Counter()
    all_unseen_records = []

    output_folder = "cv_output"
    os.makedirs(output_folder, exist_ok=True)

    fold_num = 1

    for train_index, test_index in skf.split(X, y):

        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        
        train_summary, test_summary, gap = get_split_summary(train_df, test_df)

        if export:
            train_df.to_excel(f"cv_output/fold_{fold_num}_train.xlsx", index=False)
            test_df.to_excel(f"cv_output/fold_{fold_num}_test.xlsx", index=False)

        # TRAIN
        model_tree = train_model(train_df)

        # ROOT
        root = model_tree.get("attr", "Leaf") if isinstance(model_tree, dict) else "Leaf"

        # CABANG
        branches = get_all_branches(model_tree)
        jumlah_cabang = len(branches)

        # UNSEEN
        unseen_cases = check_unseen_cases(test_df, train_df, fitur_X)
        jumlah_unseen = len(unseen_cases)

        # tambahkan info fold
        for u in unseen_cases:
            u["Fold"] = fold_num
            unseen_counter[u["Atribut"]] += 1

        all_unseen_records.extend(unseen_cases)

        # EVALUASI
        accuracy, precision, recall, f1, cm, y_test, y_pred = evaluate_model(
            test_df, model_tree
        )

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        fold_results.append({
            "Fold": fold_num,

            # JUMLAH DATA
            "Data Latih": len(train_df),
            "Data Uji": len(test_df),

            # DISTRIBUSI LATIH
            "Latih Berat": train_summary["Jumlah Berat"],
            "Latih Ringan": train_summary["Jumlah Ringan"],
            "Entropy Latih": train_summary["Entropy"],

            # DISTRIBUSI UJI
            "Uji Berat": test_summary["Jumlah Berat"],
            "Uji Ringan": test_summary["Jumlah Ringan"],
            "Entropy Uji": test_summary["Entropy"],

            # GAP
            "Gap Entropy": gap["Gap Entropy"],

            # MODEL
            "Root": root,
            "Cabang Terbentuk": jumlah_cabang,
            "Cabang Unseen": jumlah_unseen,

            # METRIK
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

        fold_num += 1

    summary = {
        "accuracy": sum(accuracies) / len(accuracies),
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1_scores) / len(f1_scores),
    }

    unseen_df = pd.DataFrame(all_unseen_records)

    if not unseen_df.empty:
        unseen_df = unseen_df[["Fold", "Atribut", "Nilai"]]

    unseen_attr_df = pd.DataFrame(
        unseen_counter.items(),
        columns=["Atribut", "Jumlah Unseen"]
    ).sort_values(by="Jumlah Unseen", ascending=False)

    return summary, pd.DataFrame(fold_results), unseen_df, unseen_attr_df

# ---------------------------------------------------------------------------
# AMBIL POHON KEPUTUSAN BERDASARKAN FOLD CROSS VALIDATION
# ---------------------------------------------------------------------------
def get_cross_validation_fold_tree(df, selected_fold, n_splits=5):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    fitur_X = [
        "Fasilitas",
        "Peralatan",
        "Dampak Kerusakan",
        "Penyebab",
        "Kelompok Penyebab",
        "Cuaca",
    ]

    target_Y = "Jenis Gangguan"

    X = df[fitur_X]
    y = df[target_Y]

    for fold_num, (train_index, test_index) in enumerate(
        skf.split(X, y),
        start=1
    ):

        if fold_num == selected_fold:
            train_df = df.iloc[train_index].copy()
            test_df = df.iloc[test_index].copy()

            model_tree = train_model(train_df)

            train_summary, test_summary, gap = get_split_summary(
                train_df,
                test_df
            )

            accuracy, precision, recall, f1, cm, y_test, y_pred = evaluate_model(
                test_df,
                model_tree
            )

            root = (
                model_tree.get("attr", "Leaf")
                if isinstance(model_tree, dict)
                else "Leaf"
            )

            return {
                "fold": fold_num,
                "train_df": train_df,
                "test_df": test_df,
                "model_tree": model_tree,
                "dot_source": visualize_tree_dot(model_tree),
                "svg_bytes": visualize_tree_svg_bytes(model_tree),
                "root": root,
                "train_summary": train_summary,
                "test_summary": test_summary,
                "gap": gap,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

    return None

def get_all_branches(tree, path="", branches=None):
    if branches is None:
        branches = []

    # kalau leaf
    if not isinstance(tree, dict):
        return branches

    attr = tree.get("attr")
    children = tree.get("child", {})

    # kalau struktur aneh (safety)
    if not isinstance(children, dict):
        return branches

    for value, subtree in children.items():
        new_path = f"{path} -> {attr}={value}" if path else f"{attr}={value}"
        branches.append(new_path)
        get_all_branches(subtree, new_path, branches)

    return branches

# ---------------------------------------------------------------------------
# ATRIBUT TIDAK TERBENTUK DI POHON ID3 khusus cross validation
# ---------------------------------------------------------------------------
def check_unseen_cases(test_df, train_df, fitur_X):
    unseen_list = []

    for col in fitur_X:
        train_values = set(train_df[col].unique())
        test_values = set(test_df[col].unique())

        unseen = test_values - train_values

        for val in unseen:
            unseen_list.append({
                "Atribut": col,
                "Nilai": val
            })

    return unseen_list

# ---------------------------------------------------------------------------
# ATRIBUT YANG TERBENTUK / TIDAK TERBENTUK DI POHON ID3
# ---------------------------------------------------------------------------
def get_tree_attribute_usage(model_tree, fitur_X):
    usage = {
        fitur: 0
        for fitur in fitur_X
    }

    def traverse(node):
        if not isinstance(node, dict):
            return

        attr = node.get("attr")

        if attr in usage:
            usage[attr] += 1

        children = node.get("child", {})

        for _, subtree in children.items():
            traverse(subtree)

    traverse(model_tree)

    rows = []

    for fitur in fitur_X:
        jumlah_muncul = usage[fitur]

        if jumlah_muncul > 0:
            status = "Terbentuk"
            keterangan = "Atribut digunakan sebagai node pada pohon keputusan."
        else:
            status = "Tidak Terbentuk"
            keterangan = (
                "Atribut tidak muncul sebagai node karena atribut lain memiliki "
                "Information Gain lebih tinggi atau cabang sebelumnya sudah menjadi leaf node."
            )

        rows.append({
            "Atribut": fitur,
            "Jumlah Kemunculan Node": jumlah_muncul,
            "Status": status,
            "Keterangan": keterangan
        })

    df_usage = pd.DataFrame(rows)

    df_terbentuk = df_usage[
        df_usage["Jumlah Kemunculan Node"] > 0
    ].reset_index(drop=True)

    df_tidak_terbentuk = df_usage[
        df_usage["Jumlah Kemunculan Node"] == 0
    ].reset_index(drop=True)

    return df_usage, df_terbentuk, df_tidak_terbentuk

# ---------------------------------------------------------------------------
# NILAI CABANG YANG TERBENTUK / TIDAK TERBENTUK DI POHON ID3
# ---------------------------------------------------------------------------
def get_tree_branch_value_usage(model_tree, train_df, fitur_X):
    rows = []

    def traverse(node, path="Root"):
        if not isinstance(node, dict):
            return

        attr = node.get("attr")
        children = node.get("child", {})

        if attr in fitur_X:
            semua_nilai_atribut = sorted(
                train_df[attr]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            nilai_cabang_terbentuk = sorted([
                str(value)
                for value in children.keys()
            ])

            nilai_tidak_terbentuk = sorted(
                set(semua_nilai_atribut) - set(nilai_cabang_terbentuk)
            )

            rows.append({
                "Path Node": path,
                "Node / Atribut": attr,
                "Jumlah Nilai Atribut": len(semua_nilai_atribut),
                "Cabang Terbentuk": len(nilai_cabang_terbentuk),
                "Cabang Tidak Terbentuk": len(nilai_tidak_terbentuk),
                "Nilai Cabang Terbentuk": ", ".join(nilai_cabang_terbentuk) if nilai_cabang_terbentuk else "—",
                "Nilai Cabang Tidak Terbentuk": ", ".join(nilai_tidak_terbentuk) if nilai_tidak_terbentuk else "—"
            })

        for value, subtree in children.items():
            next_path = (
                f"{path} → {attr} = {value}"
                if path != "Root"
                else f"{attr} = {value}"
            )
            traverse(subtree, next_path)

    traverse(model_tree)

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# RINGKASAN ATRIBUT DAN CABANG YANG TERBENTUK / TIDAK TERBENTUK
# ---------------------------------------------------------------------------
def get_tree_attribute_branch_summary(model_tree, train_df, fitur_X):
    summary = {}

    for fitur in fitur_X:
        nilai_global = (
            train_df[fitur]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        summary[fitur] = {
            "jumlah_node": 0,
            "path_node": [],
            "nilai_global": set(nilai_global),
            "nilai_cabang_terbentuk": set()
        }

    def traverse(node, path="Root"):
        if not isinstance(node, dict):
            return

        attr = node.get("attr")
        children = node.get("child", {})

        if attr in summary:
            summary[attr]["jumlah_node"] += 1
            summary[attr]["path_node"].append(path)

            for value in children.keys():
                summary[attr]["nilai_cabang_terbentuk"].add(str(value))

        for value, subtree in children.items():
            if path == "Root":
                next_path = f"{attr} = {value}"
            else:
                next_path = f"{path} → {attr} = {value}"

            traverse(subtree, next_path)

    traverse(model_tree)

    rows = []

    for fitur in fitur_X:
        jumlah_node = summary[fitur]["jumlah_node"]
        nilai_global = summary[fitur]["nilai_global"]
        nilai_terbentuk = summary[fitur]["nilai_cabang_terbentuk"]
        nilai_tidak_terbentuk = nilai_global - nilai_terbentuk

        if jumlah_node > 0:
            status_node = "Terbentuk"
            keterangan = (
                "Atribut digunakan sebagai node pada pohon keputusan."
            )
        else:
            status_node = "Tidak Terbentuk"
            keterangan = (
                "Atribut tidak digunakan sebagai node karena atribut lain memiliki "
                "Information Gain lebih tinggi atau cabang sebelumnya sudah menjadi leaf node."
            )

        rows.append({
            "Atribut": fitur,
            "Status Node": status_node,
            "Jumlah Kemunculan Node": jumlah_node,
            "Path Node": " | ".join(summary[fitur]["path_node"]) if summary[fitur]["path_node"] else "—",
            "Jumlah Nilai Atribut": len(nilai_global),
            "Cabang Terbentuk": len(nilai_terbentuk),
            "Cabang Tidak Terbentuk": len(nilai_tidak_terbentuk),
            "Nilai Cabang Terbentuk": ", ".join(sorted(nilai_terbentuk)) if nilai_terbentuk else "—",
            "Nilai Cabang Tidak Terbentuk": ", ".join(sorted(nilai_tidak_terbentuk)) if nilai_tidak_terbentuk else "—",
            "Keterangan": keterangan
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# RULES IF-THEN DARI POHON ID3 (UNTUK GUI)
# ---------------------------------------------------------------------------
def extract_rules_from_tree(tree, target_name="Jenis Gangguan"):
    rules = []

    def traverse(node, path):
        # Jika node adalah leaf node / kelas akhir
        if not isinstance(node, dict):
            kondisi = " AND ".join([
                f"{atribut} = {nilai}"
                for atribut, nilai in path
            ])

            if kondisi == "":
                kondisi = "Semua kondisi"

            rules.append({
                "No": len(rules) + 1,
                "IF": kondisi,
                "THEN": f"{target_name} = {node}",
                "Kelas": node,
                "Jumlah Kondisi": len(path)
            })

            return

        # Jika masih decision node
        atribut = node.get("attr")
        children = node.get("child", {})

        for nilai, subtree in children.items():
            path_baru = path + [(atribut, nilai)]
            traverse(subtree, path_baru)

    traverse(tree, [])

    return pd.DataFrame(rules)

# ---------------------------------------------------------------------------
# TRACE PREDIKSI + CABANG TIDAK TERBENTUK KHUSUS DATA UJI
# ---------------------------------------------------------------------------
def trace_prediction_single_row(row, tree):
    current_node = tree
    path = []

    while isinstance(current_node, dict):

        attr = current_node.get("attr")
        children = current_node.get("child", {})
        majority_class = current_node.get("majority")

        if attr is None:
            return majority_class, {
                "fallback_used": True,
                "atribut_tidak_terbentuk": None,
                "nilai_tidak_terbentuk": None,
                "path_terakhir": " → ".join(path) if path else "Root",
                "kelas_fallback": majority_class
            }

        value = row.get(attr, None)

        # Jika nilai atribut ada pada cabang pohon
        if value in children:
            path.append(f"{attr} = {value}")
            current_node = children[value]

        # Jika nilai atribut tidak terbentuk sebagai cabang
        else:
            return majority_class, {
                "fallback_used": True,
                "atribut_tidak_terbentuk": attr,
                "nilai_tidak_terbentuk": value,
                "path_terakhir": " → ".join(path) if path else "Root",
                "kelas_fallback": majority_class
            }

    return current_node, {
        "fallback_used": False,
        "atribut_tidak_terbentuk": "—",
        "nilai_tidak_terbentuk": "—",
        "path_terakhir": " → ".join(path) if path else "Root",
        "kelas_fallback": "—"
    }


# ---------------------------------------------------------------------------
# DETAIL NILAI ATRIBUT TIDAK TERBENTUK PADA DATA UJI
# ---------------------------------------------------------------------------
def get_unformed_test_attribute_detail(test_df, model_tree):

    fitur_X = [
        "Fasilitas",
        "Peralatan",
        "Dampak Kerusakan",
        "Penyebab",
        "Kelompok Penyebab",
        "Cuaca",
    ]

    target_Y = "Jenis Gangguan"

    rows = []

    X_test = test_df[fitur_X]

    for no, (idx, row) in enumerate(X_test.iterrows(), start=1):

        prediksi, info = trace_prediction_single_row(row, model_tree)
        aktual = test_df.loc[idx, target_Y]

        if info["fallback_used"]:

            data_row = {
                "No Data Uji": no,
                "Path Terakhir Terbentuk": info["path_terakhir"],
                "Atribut Tidak Terbentuk": info["atribut_tidak_terbentuk"],
                "Nilai Atribut Tidak Terbentuk": info["nilai_tidak_terbentuk"],
                "Kelas Majority Fallback": info["kelas_fallback"],
                "Aktual": aktual,
                "Prediksi": prediksi,
                "Status Prediksi": "Benar" if aktual == prediksi else "Salah",
            }

            for fitur in fitur_X:
                data_row[fitur] = row[fitur]

            rows.append(data_row)

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 9. DETAIL EVALUASI (UNTUK TABEL DATA UJI)
# ---------------------------------------------------------------------------
def get_evaluation_detail(test_df, train_df, model_tree):

    df_eval = test_df.copy()
    target_Y = "Jenis Gangguan"

    fitur_X = [
        "Fasilitas",
        "Peralatan",
        "Dampak Kerusakan",
        "Penyebab",
        "Kelompok Penyebab",
        "Cuaca",
    ]

    X_test = df_eval[fitur_X]

    prediksi_list = []
    cabang_status_list = []
    atribut_tidak_terbentuk_list = []
    nilai_tidak_terbentuk_list = []
    path_terakhir_list = []
    fallback_class_list = []

    for _, row in X_test.iterrows():

        prediksi, info = trace_prediction_single_row(
            row,
            model_tree
        )

        prediksi_list.append(prediksi)

        if info["fallback_used"]:
            cabang_status_list.append("Tidak Terbentuk (Majority Fallback)")
        else:
            cabang_status_list.append("Terbentuk")

        atribut_tidak_terbentuk_list.append(
            info["atribut_tidak_terbentuk"]
        )

        nilai_tidak_terbentuk_list.append(
            info["nilai_tidak_terbentuk"]
        )

        path_terakhir_list.append(
            info["path_terakhir"]
        )

        fallback_class_list.append(
            info["kelas_fallback"]
        )

    df_eval["Prediksi"] = prediksi_list
    df_eval["Status"] = df_eval[target_Y] == df_eval["Prediksi"]

    df_eval["Cabang Prediksi"] = cabang_status_list
    df_eval["Atribut Tidak Terbentuk"] = atribut_tidak_terbentuk_list
    df_eval["Nilai Atribut Tidak Terbentuk"] = nilai_tidak_terbentuk_list
    df_eval["Path Terakhir Terbentuk"] = path_terakhir_list
    df_eval["Kelas Majority Fallback"] = fallback_class_list

    return df_eval

# ---------------------------------------------------------------------------
# DETAIL PEMBAGIAN DATA PER FOLD (UNTUK GUI)
# ---------------------------------------------------------------------------
def get_cross_validation_data_split_detail(df, n_splits=5):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    fitur_X = [
        "Fasilitas",
        "Peralatan",
        "Dampak Kerusakan",
        "Penyebab",
        "Kelompok Penyebab",
        "Cuaca",
    ]

    target_Y = "Jenis Gangguan"

    X = df[fitur_X]
    y = df[target_Y]

    detail_df = df.copy().reset_index(drop=True)
    detail_df.insert(0, "No Data", range(1, len(detail_df) + 1))

    for fold in range(1, n_splits + 1):
        detail_df[f"Fold {fold}"] = "Latih"

    for fold_num, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        detail_df.loc[test_index, f"Fold {fold_num}"] = "Uji"

    return detail_df