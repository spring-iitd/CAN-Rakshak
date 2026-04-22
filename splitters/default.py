from .base import BaseSplitter
import os
import csv
import numpy as np


class Default(BaseSplitter):
    def __init__(self, input_dir, feature_extractor, cfg):
        super().__init__(input_dir)
        self.split_ratio       = cfg['split_ratio']
        self.feature_extractor = feature_extractor
        self.cfg               = cfg

    def split(self):
        cfg       = self.cfg
        file_name = cfg['file_name']
        prefix    = file_name[:-4]

        features_path = self.feature_extractor.features_path
        features_csv  = os.path.join(features_path, "Stat", prefix + "_features.csv")
        labels_csv    = os.path.join(features_path, "Stat", prefix + "_labels.csv")

        if not (os.path.exists(features_csv) and os.path.exists(labels_csv)):
            print(f"  No Stat features found at {os.path.dirname(features_csv)}, skipping split.")
            return

        X_all, y_all = self.load_stat_features(features_csv, labels_csv)
        split_index  = int((1 - self.split_ratio) * X_all.shape[0])
        x_train, y_train = X_all[:split_index], y_all[:split_index]
        x_test,  y_test  = X_all[split_index:],  y_all[split_index:]

        train_dir = os.path.join(self.input_dir, "train", cfg['train_dataset_dir'])
        test_dir  = os.path.join(self.input_dir, "test",  cfg['test_dataset_dir'])
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir,  exist_ok=True)

        self.save_stat_features(x_train, y_train,
                                os.path.join(train_dir, prefix + "_train_features.csv"),
                                os.path.join(train_dir, prefix + "_train_labels.csv"))
        self.save_stat_features(x_test,  y_test,
                                os.path.join(test_dir,  prefix + "_test_features.csv"),
                                os.path.join(test_dir,  prefix + "_test_labels.csv"))
        np.savez(os.path.join(train_dir, prefix + "_train_data.npz"), x_train=x_train, y_train=y_train)
        np.savez(os.path.join(test_dir,  prefix + "_test_data.npz"),  x_test=x_test,  y_test=y_test)
        print(f"  Split          : Train={len(y_train)}, Test={len(y_test)}")

    def load_stat_features(self, features_csv, labels_csv):
        X = np.loadtxt(features_csv, delimiter=",")

        labels = []
        with open(labels_csv, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                labels.append(int(row[1]))

        return X, np.array(labels)

    def save_stat_features(self, X, Y, features_csv, labels_csv):
        np.savetxt(features_csv, X, delimiter=",")

        with open(labels_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "label"])
            for i, label in enumerate(Y):
                writer.writerow([i, label])
