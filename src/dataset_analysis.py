import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dataset_analysis(dataset_path, cfg):

    print("Running Dataset Analysis...")
    file_name = cfg['file_name']
    file_path = os.path.join(dataset_path, "modified_dataset", file_name[:-3] + "csv")

    cols = ["timestamp","can_id","dlc",
            "b0","b1","b2","b3","b4","b5","b6","b7","flag"]

    df = pd.read_csv(file_path, header=None, names=cols)

    df["timestamp"] = df["timestamp"].astype(float)

    output_dir = os.path.join(dataset_path,"analysis", file_name[:-4])
    os.makedirs(output_dir, exist_ok=True)

    convert_payload_to_int(df)

    basic_statistics(df)

    plot_can_id_distribution(df, output_dir)

    plot_message_rate(df, output_dir)

    plot_interarrival(df, output_dir)

    plot_payload_histograms(df, output_dir)

    plot_canid_vs_time(df, output_dir)

    plot_payload_entropy(df, output_dir)

    plot_byte_correlation(df, output_dir)

    plot_canid_periodicity(df, output_dir)

    plot_distinct_ids(df, output_dir)

    plot_attack_distribution(df, output_dir)

    print("Analysis saved to:", output_dir)



def convert_payload_to_int(df):

    payload_cols = ["b0","b1","b2","b3","b4","b5","b6","b7"]

    for col in payload_cols:
        df[col] = df[col].apply(lambda x: int(str(x),16))


def basic_statistics(df):

    print("\n===== Dataset Statistics =====")

    print("Total Frames:", len(df))
    print("Unique CAN IDs:", df["can_id"].nunique())
    print("DLC Distribution:\n", df["dlc"].value_counts())


def plot_can_id_distribution(df, output_dir):

    counts = df["can_id"].value_counts().head(20)

    plt.figure()
    counts.plot(kind="bar")
    plt.title("Top 20 CAN ID Frequency")
    plt.xlabel("CAN ID")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"canid_distribution.png"))
    plt.close()



def plot_message_rate(df, output_dir):

    df["time_bin"] = (df["timestamp"] // 0.1)

    rate = df.groupby("time_bin").size()

    plt.figure()
    rate.plot()

    plt.title("Message Rate Over Time")
    plt.xlabel("Time Bin (0.1s)")
    plt.ylabel("Messages")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"message_rate.png"))
    plt.close()



def plot_interarrival(df, output_dir):

    df["iat"] = df["timestamp"].diff()

    plt.figure()
    plt.hist(df["iat"].dropna(), bins=100)

    plt.title("Inter Arrival Time Distribution")
    plt.xlabel("Time Difference")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"interarrival.png"))
    plt.close()



def plot_payload_histograms(df, output_dir):

    payload_cols = ["b0","b1","b2","b3","b4","b5","b6","b7"]

    for col in payload_cols:

        plt.figure()
        df[col].hist(bins=50)

        plt.title(f"{col} Value Distribution")
        plt.xlabel("Value")
        plt.ylabel("Count")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,f"{col}_hist.png"))
        plt.close()



def plot_canid_vs_time(df, output_dir):

    plt.figure()

    ids = df["can_id"].apply(lambda x: int(x,16))

    plt.scatter(df["timestamp"], ids, s=1)

    plt.title("CAN ID vs Time")
    plt.xlabel("Timestamp")
    plt.ylabel("CAN ID")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"canid_vs_time.png"))
    plt.close()



def plot_payload_entropy(df, output_dir):

    payload_cols = ["b0","b1","b2","b3","b4","b5","b6","b7"]

    payload = df[payload_cols].values


    def entropy(row):

        counts = np.bincount(row, minlength=256)

        probs = counts / counts.sum()

        probs = probs[probs > 0]

        return -np.sum(probs * np.log2(probs))


    ent = [entropy(row) for row in payload]

    plt.figure()
    plt.hist(ent, bins=50)

    plt.title("Payload Entropy Distribution")
    plt.xlabel("Entropy")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"payload_entropy.png"))
    plt.close()



def plot_byte_correlation(df, output_dir):

    payload_cols = ["b0","b1","b2","b3","b4","b5","b6","b7"]

    corr = df[payload_cols].corr()

    plt.figure()
    plt.imshow(corr)
    plt.colorbar()

    plt.xticks(range(8), payload_cols)
    plt.yticks(range(8), payload_cols)

    plt.title("Payload Byte Correlation")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"byte_correlation.png"))
    plt.close()

def plot_canid_periodicity(df, output_dir):

    df_sorted = df.sort_values("timestamp")

    df_sorted["iat"] = df_sorted.groupby("can_id")["timestamp"].diff()

    plt.figure()

    df_sorted["iat"].dropna().hist(bins=100)

    plt.title("CAN ID Periodicity (Inter-arrival time per ID)")
    plt.xlabel("Time Difference")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "canid_periodicity.png"))
    plt.close()

def plot_distinct_ids(df, output_dir):

    df_sorted = df.sort_values("timestamp")

    unique_ids = []

    seen = set()

    for cid in df_sorted["can_id"]:
        seen.add(cid)
        unique_ids.append(len(seen))

    plt.figure()

    plt.plot(unique_ids)

    plt.title("Distinct CAN IDs Over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("Number of Unique IDs")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distinct_can_ids.png"))
    plt.close()

def plot_attack_distribution(df, output_dir):

    if "label" not in df.columns:
        print("No label column found. Skipping attack distribution.")
        return

    counts = df["label"].value_counts()

    labels = ["Benign", "Attack"]

    plt.figure()

    plt.bar(labels, counts)

    plt.title("Attack vs Benign Distribution")
    plt.ylabel("Number of Frames")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attack_distribution.png"))
    plt.close()


def plot_class_distribution(df, output_dir):
    print("Plotting class distribution...")

    plt.figure(figsize=(6,4))

    df["flag"].value_counts().plot(kind="bar")

    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Messages")
    plt.xticks([0,1],["Normal","Attack"], rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()

def plot_timeline_distribution(df, output_dir):
    print("Plotting timeline distribution...")

    plt.figure(figsize=(10,5))

    plt.hist(df["timestamp"], bins=100)

    plt.title("Message Distribution Over Time")
    plt.xlabel("Time")
    plt.ylabel("Messages")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "timeline_distribution.png"))
    plt.close()