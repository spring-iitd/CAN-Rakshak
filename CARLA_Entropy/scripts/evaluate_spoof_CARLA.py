#!/usr/bin/env python3
"""
CARLA Spoof Entropy Evaluation Script
Description: Detects attacks based on Shannon Entropy of CAN payloads.
Replaces the Inception-ResNet target model with entropy-based detection.
Follows the exact same pipeline structure as evaluate_spoof_CARLA.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import csv
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ---------------------------------------------------------
# CARLA Spoof ENTROPY CONSTANTS
# (Computed from benign-only training data:
#  CARLA_dataset/Spoof/Target/target_train.csv)
# ---------------------------------------------------------
TRAIN_MEAN = 5.6106
TRAIN_STD  = 0.1734
K          = 3.5
WINDOW     = 0.0376
LOWER      = TRAIN_MEAN - K * TRAIN_STD   # 5.0027
UPPER      = TRAIN_MEAN + K * TRAIN_STD   # 6.2185


# ---------------------------------------------------------
# Helper: Safe Hex/Int Parser
# ---------------------------------------------------------
def parse_hex(x):
    try:
        if pd.isna(x) or str(x).strip() == "":
            return 0
        s = str(x).strip()
        if '.' in s:
            return int(float(s))
        return int(s, 16)
    except:
        return 0


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
def preprocess_dataframe(df):
    print(f"   -> Raw data shape: {df.shape}")

    df.columns = df.columns.str.strip()

    # 1. Standardize Timestamp
    ts_col = next((c for c in ["Timestamp", "timestamp", "Time", "TimeStamp", "time"]
                   if c in df.columns), None)
    if ts_col is None:
        ts_col = df.columns[0]

    df = df.rename(columns={ts_col: "Timestamp"})
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df.dropna(subset=["Timestamp"], inplace=True)

    # 2. Standardize ID
    if "ID" in df.columns and "can_id" not in df.columns:
        df = df.rename(columns={"ID": "can_id"})
    if "can_id" not in df.columns:
        df.rename(columns={df.columns[1]: "can_id"}, inplace=True)
    df["can_id"] = df["can_id"].apply(parse_hex)

    # 3. Standardize DLC
    if "DLC" in df.columns and "dlc" not in df.columns:
        df = df.rename(columns={"DLC": "dlc"})
    df["dlc"] = pd.to_numeric(df["dlc"], errors="coerce").fillna(0).astype(int)

    # 4. Standardize Payload
    payload_cols = ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]
    for c in payload_cols:
        if c not in df.columns:
            df[c] = 0
    df[payload_cols] = df[payload_cols].fillna(0)
    for c in payload_cols:
        df[c] = df[c].apply(parse_hex)

    df["payload"] = df[payload_cols].values.tolist()

    # 5. Standardize Label
    if "label" not in df.columns:
        df["label"] = 0

    df["label"] = df["label"].astype(str).str.upper().map({
        "B": 0, "R": 0, "0": 0, "BENIGN": 0, "NAN": 0, "NONE": 0,
        "T": 1, "A": 1, "1": 1, "ATTACK": 1, "SPOOF": 1
    }).fillna(0).astype(int)

    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------
# Windowing
# ---------------------------------------------------------
def split_into_windows(df, window_size):
    if df.empty:
        return [], np.array([]), []
    start, end = df["Timestamp"].min(), df["Timestamp"].max()
    windows, labels, indices = [], [], []
    t = start
    while t <= end:
        w = df[(df["Timestamp"] >= t) & (df["Timestamp"] < t + window_size)]
        if not w.empty:
            windows.append(w)
            labels.append(int((w["label"] == 1).any()))
            indices.append(w.index)
        t += window_size
    return windows, np.array(labels), indices


# ---------------------------------------------------------
# Entropy Calculation
# ---------------------------------------------------------
def calculate_entropy(windows):
    ent = []
    for w in windows:
        symbols = []
        for _, r in w.iterrows():
            for i, v in enumerate(r["payload"]):
                symbols.append((r["can_id"], r["dlc"], i, v))
        if not symbols:
            ent.append(0.0)
            continue
        _, c = np.unique(symbols, axis=0, return_counts=True)
        p = c / c.sum()
        ent.append(-np.sum(p * np.log2(p)))
    return np.array(ent)


# ---------------------------------------------------------
# Save Predictions & Update Tracksheet
# (Same logic as evaluate_spoof_CARLA.py save_preds)
# ---------------------------------------------------------
def save_preds(pass_num, tracksheet, pred_labels_list, output_path, preds):

    print(f"-> Updating tracksheet: {tracksheet}")
    try:
        df = pd.read_csv(tracksheet, dtype=str, low_memory=False)
    except FileNotFoundError:
        print(f"[ERROR] Tracksheet {tracksheet} not found.")
        return

    df.columns = df.columns.str.strip()
    df = df.fillna("None")

    df["row_no"]     = df["row_no"].astype(int)
    df["timestamp"]  = df["timestamp"].astype(float)
    df["image_no"]   = df["image_no"].astype(int)
    df["valid_flag"] = df["valid_flag"].astype(int)

    pred_labels = pred_labels_list

    n_df   = len(df)
    n_pred = len(pred_labels)

    if n_pred < n_df:
        print(f"[WARN] pred_labels shorter than packet CSV: "
              f"{n_pred} vs {n_df}. Filling remaining using operation_label.")
        for i in range(n_pred, n_df):
            op = str(df.iloc[i]["operation_label"]).strip().upper()
            if op == "NONE":
                pred_labels.append("B")
            else:
                pred_labels.append("A")
    elif n_pred > n_df:
        print(f"[WARN] pred_labels longer than packet CSV: "
              f"{n_pred} vs {n_df}. Truncating extra predictions.")
        pred_labels = pred_labels[:n_df]

    assert len(pred_labels) == n_df

    df["pred_label"] = pred_labels

    df["timestamp"] = df["timestamp"].map(lambda x: f"{x:.6f}")

    int_cols = ["row_no", "image_no", "valid_flag"]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype(int)

    tracksheet_dir = "tracksheets_CARLA"
    os.makedirs(tracksheet_dir, exist_ok=True)

    new_tracksheet = os.path.join(tracksheet_dir, f"spoof_test_track_{pass_num}.csv")
    df.to_csv(new_tracksheet, index=False)

    print(f"Saved updated packet-level CSV -> {new_tracksheet} "
          f"(rows={n_df}, preds={len(pred_labels)})")


# ---------------------------------------------------------
# Confusion Matrix Plot
# ---------------------------------------------------------
def plot_confusion(cm, pass_num, y_test, preds):
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix - Spoof (Entropy)")
    plt.colorbar()
    ticks = ["Benign", "Attack"]
    plt.xticks(range(2), ticks)
    plt.yticks(range(2), ticks)

    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, f"{cm[i,j]}",
                 ha="center",
                 color="white" if cm[i,j] > np.max(cm)/2 else "black")

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()

    os.makedirs("./CF_target", exist_ok=True)
    plt.savefig("./CF_target/CARLA_spoof_entropy_pass_{}.png".format(pass_num))
    plt.close()

    TN, FP, FN, TP = cm.ravel()

    accuracy     = accuracy_score(y_test, preds)
    precision    = precision_score(y_test, preds, pos_label=1, zero_division=0)
    rec          = recall_score(y_test, preds, pos_label=1, zero_division=0)
    f1           = f1_score(y_test, preds, pos_label=1, zero_division=0)
    tpr          = TP / (TP + FN) if (TP + FN) > 0 else 0
    tnr          = TN / (TN + FP) if (TN + FP) > 0 else 0
    fpr          = FP / (FP + TN) if (FP + TN) > 0 else 0
    fnr          = FN / (TP + FN) if (TP + FN) > 0 else 0
    balanced_acc = balanced_accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, preds)
    except:
        auc = 0.0

    print("\n--------------- PERFORMANCE METRICS ----------------")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall / TPR:", rec)
    print("True Negative Rate (TNR):", tnr)
    print("False Positive Rate (FPR):", fpr)
    print("False Negative Rate (FNR):", fnr)
    print("F1 Score:", f1)
    print("Balanced Accuracy:", balanced_acc)
    print("ROC AUC:", auc)
    print("---------------------------------------------------\n")

    print("Confusion Matrix (Raw Values):")
    print(cm)
    print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def run(params):

    rounds       = params["rounds"]
    traffic_path = params["traffic_path"]
    tracksheet   = params["tracksheet"]
    output_path  = params["output_path"]

    print(f"\nDataset: CARLA Spoof (Entropy) | Mean: {TRAIN_MEAN} | Std: {TRAIN_STD} "
          f"| K: {K} | Window: {WINDOW}")
    print(f"Thresholds: Lower={LOWER:.4f}, Upper={UPPER:.4f}")

    # 1. Load & Preprocess
    print(f"\n--- Loading Data: {traffic_path} ---")
    if not os.path.exists(traffic_path):
        print(f"CRITICAL ERROR: File {traffic_path} not found.")
        return

    try:
        df = pd.read_csv(traffic_path, on_bad_lines='skip', low_memory=False)
        df.columns = df.columns.str.strip()

        if df.columns[0] not in ["Timestamp", "timestamp", "Time", "time"]:
            col_names = ["Timestamp", "can_id", "dlc",
                         "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "label"]
            df = pd.read_csv(traffic_path, delimiter=',', header=None,
                             names=col_names, on_bad_lines='skip', low_memory=False)
            df.columns = df.columns.str.strip()

            if str(df.iloc[0]["Timestamp"]).lower() in ["timestamp", "time"]:
                df = df.iloc[1:].reset_index(drop=True)

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    try:
        df = preprocess_dataframe(df)
    except KeyError as e:
        print(f"Preprocessing Error: {e}")
        return

    if df.empty:
        print("Error: DataFrame is empty.")
        return

    # 2. Windowing
    print("\n--- Splitting into Time Windows ---")
    windows, y_test, window_indices = split_into_windows(df, WINDOW)

    print("\nWINDOW DISTRIBUTION")
    print("-----------------------------------")
    print(f"Total Windows: {len(y_test)}")
    print(f"Benign: {(y_test == 0).sum()}")
    print(f"Attack: {(y_test == 1).sum()}")
    print("-----------------------------------\n")

    if not windows:
        print("Error: No windows created.")
        return

    # 3. Calculate Entropy
    print("--- Calculating Entropy ---")
    ent = calculate_entropy(windows)

    # 4. Prediction
    print(f"Applying Thresholds: Lower={LOWER:.4f}, Upper={UPPER:.4f}")
    preds = ((ent < LOWER) | (ent > UPPER)).astype(int)

    # 5. Evaluate & Plot
    cm = confusion_matrix(y_test, preds)
    plot_confusion(cm, rounds, y_test, preds)

    print(f"\nSaved confusion matrix: CARLA_spoof_entropy_pass_{rounds}.png\n")

    # 6. Map window predictions back to packet-level
    df["pred_label"] = "B"
    for i, idxs in enumerate(window_indices):
        if preds[i] == 0:
            df.loc[idxs, "pred_label"] = "B"
        else:
            df.loc[idxs, "pred_label"] = df.loc[idxs, "label"].map({1: "A", 0: "B"})

    # 7. Save detailed prediction output
    df_out = df.drop(columns=["payload"], errors='ignore')
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print("Saved detailed prediction results ->", output_path)

    # 8. Update tracksheet
    pred_labels = df["pred_label"].tolist()
    save_preds(rounds, tracksheet, pred_labels, output_path, preds)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_spoof_CARLA.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    if "evaluate" not in cfg:
        raise ValueError("Config file must contain 'evaluate' section.")

    run(cfg["evaluate"])
