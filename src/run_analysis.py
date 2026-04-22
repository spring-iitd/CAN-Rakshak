"""Run dataset analysis for Car Hacking Dataset files."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_analysis import (
    convert_payload_to_int,
    basic_statistics,
    plot_can_id_distribution,
    plot_message_rate,
    plot_interarrival,
    plot_class_distribution,
    plot_timeline_distribution,
    plot_canid_vs_time,
    plot_payload_entropy,
    plot_byte_correlation,
    plot_canid_periodicity,
    plot_distinct_ids,
    plot_attack_distribution,
)

DATASET_PATH = os.path.join("datasets", "CarHackingDataset")
MODIFIED_DIR = os.path.join(DATASET_PATH, "modified_dataset")

# All CSV files in the modified_dataset folder
csv_files = [f for f in os.listdir(MODIFIED_DIR) if f.endswith(".csv")]
csv_files.sort()

print(f"Found {len(csv_files)} CSV files:")
for i, f in enumerate(csv_files):
    print(f"  [{i}] {f}")

cols = ["timestamp", "can_id", "dlc",
        "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "flag"]

for csv_file in csv_files:
    print(f"\n{'='*60}")
    print(f"Analyzing: {csv_file}")
    print(f"{'='*60}")

    file_path = os.path.join(MODIFIED_DIR, csv_file)
    name = csv_file.rsplit(".", 1)[0]

    output_dir = os.path.join(DATASET_PATH, "analysis", name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(file_path, header=None, names=cols)
    except Exception:
        # Try with header
        df = pd.read_csv(file_path)
        # Rename columns if needed
        if list(df.columns) != cols:
            print(f"  Skipping {csv_file}: unexpected columns {list(df.columns)}")
            continue

    df["timestamp"] = df["timestamp"].astype(float)

    convert_payload_to_int(df)
    basic_statistics(df)
    plot_can_id_distribution(df, output_dir)
    plot_message_rate(df, output_dir)
    plot_interarrival(df, output_dir)
    plot_class_distribution(df, output_dir)
    plot_canid_vs_time(df, output_dir)
    plot_payload_entropy(df, output_dir)
    plot_byte_correlation(df, output_dir)
    plot_canid_periodicity(df, output_dir)
    plot_distinct_ids(df, output_dir)
    plot_attack_distribution(df, output_dir)
    plot_timeline_distribution(df, output_dir)

    print(f"  Analysis saved to: {output_dir}")

print("\nDone! All analyses complete.")
