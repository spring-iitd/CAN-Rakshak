# PixNet splitter — work in progress, will be released upon publication
from .base import BaseSplitter
import os
import re
import shutil
import pandas as pd


class PixSplitter(BaseSplitter):
    def __init__(self, input_dir, feature_extractor, cfg):
        super().__init__(input_dir)
        self.split_ratio       = cfg['split_ratio']
        self.feature_extractor = feature_extractor
        self.cfg               = cfg

    def split(self):
        # PixNet split logic — work in progress, will be released upon publication
        split_and_store_data(self.cfg)


def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else float('inf')


valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def extract_files(src_folder):
    return [
        f for f in os.listdir(src_folder)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]


def sequential_split_images(src_folder, train_folder, test_folder, split_ratio=0.2):
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder,  exist_ok=True)

    images      = extract_files(src_folder)
    total       = len(images)
    split_index = total - int(total * split_ratio)

    sorted_images = sorted(images, key=extract_number)
    train_images  = sorted_images[:split_index]
    test_images   = sorted_images[split_index:]

    for img in train_images:
        shutil.copy(os.path.join(src_folder, img), os.path.join(train_folder, img))
    for img in test_images:
        shutil.copy(os.path.join(src_folder, img), os.path.join(test_folder, img))


def split_labels(label_file, train_images, test_images, train_label_file, test_label_file):
    labels = {}
    with open(label_file, "r") as f:
        for line in f:
            if ":" in line:
                img, lab = line.strip().split(":", 1)
                labels[img.strip()] = lab.strip()

    with open(train_label_file, "w") as f:
        for img in train_images:
            if img in labels:
                f.write(f"{img}: {labels[img]}\n")

    with open(test_label_file, "w") as f:
        for img in test_images:
            if img in labels:
                f.write(f"{img}: {labels[img]}\n")


def split_track_csv(track_csv, train_images, test_images, train_csv, test_csv):
    df = pd.read_csv(track_csv)
    df.columns = df.columns.str.strip()

    train_img_nums = {int(''.join(filter(str.isdigit, img))) for img in train_images}
    test_img_nums  = {int(''.join(filter(str.isdigit, img))) for img in test_images}

    train_df = df[df["image_no"].isin(train_img_nums)]
    test_df  = df[df["image_no"].isin(test_img_nums)]

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Track split → Train rows: {len(train_df)}, Test rows: {len(test_df)}")


def split_and_store_data(cfg):
    if not cfg['split']:
        return

    dir_path     = cfg['dir_path']
    dataset_name = cfg['dataset_name']
    file_name    = cfg['file_name']

    input_dir       = os.path.join(dir_path, "..", "datasets", dataset_name)
    train_dir       = os.path.join(input_dir, "train", cfg['train_dataset_dir'])
    test_dir        = os.path.join(input_dir, "test",  cfg['test_dataset_dir'])
    input_directory = os.path.join(input_dir, "features", "Images", file_name[:-4] + "_images")

    print("Splitting dataset into Train and Test")
    sequential_split_images(input_directory, train_dir, test_dir, cfg['split_ratio'])

    # PixNet label/track splitting — work in progress, will be released upon publication
    if cfg['feature_extractor'] == "PixNet":
        label_file       = os.path.join(input_directory, "labels.txt")
        train_images     = sorted(extract_files(train_dir), key=extract_number)
        test_images      = sorted(extract_files(test_dir),  key=extract_number)
        train_label_file = os.path.join(train_dir, "labels.txt")
        test_label_file  = os.path.join(test_dir,  "labels.txt")
        split_labels(label_file, train_images, test_images, train_label_file, test_label_file)

        csv_file        = os.path.join(input_dir, "csv_files", file_name[:-4] + "_track.csv")
        train_track_csv = os.path.join(train_dir, "track.csv")
        test_track_csv  = os.path.join(test_dir,  "track.csv")
        split_track_csv(csv_file, train_images, test_images, train_track_csv, test_track_csv)
