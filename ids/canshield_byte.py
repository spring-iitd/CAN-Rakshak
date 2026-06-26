"""
Byte-level CANShield IDS module for the CAN-Rakshak framework.

Extends CANShield (IEEE IoT Journal 2023) to operate directly on raw CAN
payload bytes, removing the dependency on proprietary DBC signal-decoding
files. Adds per-CAN-ID anomaly scoring to overcome the feature-dilution
problem on single-byte attacks.

Author: Shubham Thakur (IIT Delhi)

Conforms to the CAN-Rakshak IDS interface (ids.base.IDS):
  train(), test(), predict(), save(), load()

Reads the framework-standard CSV (no header):
  timestamp, can_id, dlc, b1..b8, label
from datasets/<DATASET_NAME>/modified_dataset/<FILE_NAME>.csv
"""

import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd

from config import *
from ids.base import IDS

# TensorFlow is imported lazily inside _build_ae so the module can be
# imported even in environments where only inference/load is needed.


class CANShieldByte(IDS):
    """Unsupervised byte-level CAN IDS using a convolutional autoencoder."""

    def __init__(self, top_n_ids: int = 20, window: int = 50,
                 periods=(1, 5, 10), epochs: int = None,
                 rl_percentile: float = 95.0, rt_percentile: float = 99.0):
        """
        Args:
            top_n_ids:     number of most-frequent CAN IDs to monitor.
            window:        sliding-window length W (messages per view).
            periods:       multi-scale sampling periods for the views.
            epochs:        AE training epochs (falls back to config EPOCHS).
            rl_percentile: per-feature reconstruction-error threshold percentile.
            rt_percentile: per-feature exceedance-count threshold percentile.
        """
        self.top_n_ids = top_n_ids
        self.window = window
        self.periods = tuple(periods)
        self.epochs = epochs if epochs is not None else int(globals().get("EPOCHS", 50))
        self.rl_percentile = rl_percentile
        self.rt_percentile = rt_percentile

        # learned state
        self.selected_ids = None      # list of monitored CAN IDs
        self.id_to_col = None         # id -> starting column in byte matrix
        self.n_features = None        # n_ids * 8
        self.feat_min = None          # per-feature normalisation min
        self.feat_max = None          # per-feature normalisation max
        self.order = None             # correlation-clustering column order
        self.models = {}              # period -> keras autoencoder
        self.thresholds = {}          # period -> (RL, RT)
        super().__init__()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    _COLUMNS = ["timestamp", "can_id", "dlc",
                "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "label"]

    @staticmethod
    def _is_hex(s):
        try:
            int(str(s), 16)
            return True
        except (ValueError, TypeError):
            return False

    def _load_csv(self, path):
        """Load framework CSV into (timestamps, ids, bytes[N,8], labels)."""
        df = pd.read_csv(path, header=None, names=self._COLUMNS)
        ts = df["timestamp"].astype(float).values
        ids = np.array([int(x, 16) if self._is_hex(x) else int(x)
                        for x in df["can_id"].values], dtype=np.int64)
        byte_cols = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"]
        raw = df[byte_cols].values
        by = np.zeros((len(df), 8), dtype=np.float32)
        for i in range(len(df)):
            for j in range(8):
                v = raw[i, j]
                by[i, j] = int(v, 16) if self._is_hex(v) else 0
        labels = df["label"].astype(int).values if "label" in df else np.zeros(len(df), int)
        return ts, ids, by, labels

    @staticmethod
    def _dataset_csv():
        """Path to the framework-standard CSV for the configured dataset."""
        return os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME,
                            "modified_dataset", FILE_NAME[:-4] + ".csv")

    # ------------------------------------------------------------------
    # Byte-matrix construction
    # ------------------------------------------------------------------
    def _build_matrix(self, ids, by):
        """Forward-filled byte matrix [N, n_features] over monitored IDs."""
        n = len(ids)
        mat = np.zeros((n, self.n_features), dtype=np.float32)
        for i in range(n):
            if i > 0:
                mat[i] = mat[i - 1]
            col = self.id_to_col.get(int(ids[i]))
            if col is not None:
                mat[i, col:col + 8] = by[i]
        return mat

    def _normalise(self, mat):
        rng = self.feat_max - self.feat_min
        rng[rng == 0] = 1.0
        return np.clip((mat - self.feat_min) / rng, 0.0, 1.0).astype(np.float32)

    def _make_views(self, mat, labels, period):
        """Multi-scale views [num, n_features, W] at the given period."""
        max_T = max(self.periods)
        start = max_T * (self.window - 1)
        if start >= len(mat):
            return None, None
        num = (len(mat) - start)
        idxs = []
        view_labels = []
        for pos in range(start, len(mat)):
            rows = np.arange(pos - (self.window - 1) * period, pos + 1, period)
            idxs.append(rows)
            lo = max(0, pos - max_T * self.window)
            view_labels.append(1 if labels[lo:pos + 1].max() > 0 else 0)
        views = np.stack([mat[r][:, self.order].T for r in idxs]).astype(np.float32)
        return views, np.array(view_labels, dtype=np.int64)

    # ------------------------------------------------------------------
    # Autoencoder
    # ------------------------------------------------------------------
    def _build_ae(self):
        import tensorflow as tf
        from tensorflow.keras import layers, Model

        h, w = self.n_features, self.window
        inp = layers.Input(shape=(h, w, 1))
        x = layers.Conv2D(32, 3, padding="same")(inp)
        x = layers.BatchNormalization()(x); x = layers.LeakyReLU()(x)
        x = layers.MaxPooling2D(2, padding="same")(x)
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x); x = layers.LeakyReLU()(x)
        x = layers.MaxPooling2D(2, padding="same")(x)
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x); x = layers.LeakyReLU()(x)
        x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x); x = layers.LeakyReLU()(x)
        x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x); x = layers.LeakyReLU()(x)
        out = layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
        out = layers.Resizing(h, w)(out)
        model = Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        return model

    # ------------------------------------------------------------------
    # IDS interface
    # ------------------------------------------------------------------
    def train(self, X_train=None, Y_train=None, **kwargs):
        """Fit the byte-level autoencoders on normal traffic."""
        ts, ids, by, labels = self._load_csv(self._dataset_csv())
        normal = labels == 0
        ids_n, by_n, lab_n = ids[normal], by[normal], labels[normal]

        # select top-N CAN IDs by frequency
        counts = Counter(int(x) for x in ids_n)
        self.selected_ids = [i for i, _ in counts.most_common(self.top_n_ids)]
        self.id_to_col = {sid: k * 8 for k, sid in enumerate(self.selected_ids)}
        self.n_features = len(self.selected_ids) * 8

        mat = self._build_matrix(ids_n, by_n)
        self.feat_min = mat.min(axis=0)
        self.feat_max = mat.max(axis=0)
        mat = self._normalise(mat)

        # correlation clustering for column ordering
        self.order = self._cluster_order(mat)

        # train one AE per period
        for T in self.periods:
            views, _ = self._make_views(mat, lab_n, T)
            if views is None:
                continue
            model = self._build_ae()
            x = views[..., np.newaxis]
            model.fit(x, x, epochs=self.epochs, batch_size=64,
                      validation_split=0.3, verbose=2)
            self.models[T] = model

            recon = model.predict(x, verbose=0)
            err = np.abs(x - recon)[..., 0]                       # [num, feat, W]
            RL = np.percentile(err, self.rl_percentile, axis=(0, 2))
            mask = (err > RL[np.newaxis, :, np.newaxis]).astype(int)
            RT = np.percentile(mask.sum(axis=2), self.rt_percentile, axis=0)
            self.thresholds[T] = (RL, RT)
            print(f"[CANShieldByte] period T={T}: trained, thresholds set.")

    def _cluster_order(self, mat):
        from scipy.cluster.hierarchy import linkage, leaves_list
        c = np.corrcoef(mat.T)
        c = np.nan_to_num(c)
        d = 1.0 - np.abs(c)
        np.fill_diagonal(d, 0.0)
        try:
            from scipy.spatial.distance import squareform
            Z = linkage(squareform(d, checks=False), method="average")
            return list(leaves_list(Z))
        except Exception:
            return list(range(self.n_features))

    def _score(self, mat, labels):
        """Return (scores, view_labels) averaged over periods, with per-CAN-ID."""
        agg = None
        view_labels = None
        for T in self.periods:
            if T not in self.models:
                continue
            views, vlab = self._make_views(mat, labels, T)
            if views is None:
                continue
            x = views[..., np.newaxis]
            recon = self.models[T].predict(x, verbose=0)
            err = np.abs(x - recon)[..., 0]
            RL, RT = self.thresholds[T]
            mask = (err > RL[np.newaxis, :, np.newaxis]).astype(int)
            V = mask.sum(axis=2)
            S = (V > RT[np.newaxis, :]).astype(int)            # [num, feat]

            # per-CAN-ID score: max over IDs of fraction-flagged within that ID
            n_ids = self.n_features // 8
            per_id = np.zeros((S.shape[0], n_ids))
            for k in range(n_ids):
                per_id[:, k] = S[:, k * 8:(k + 1) * 8].mean(axis=1)
            score = per_id.max(axis=1)

            agg = score if agg is None else agg + score
            view_labels = vlab
        if agg is None:
            return None, None
        return agg / len(self.models), view_labels

    def test(self, X_test=None, Y_test=None, **kwargs):
        """Score attack traffic and report AUROC / F1 if labels available."""
        ts, ids, by, labels = self._load_csv(self._dataset_csv())
        mat = self._normalise(self._build_matrix(ids, by))
        scores, vlab = self._score(mat, labels)
        if scores is None:
            print("[CANShieldByte] not enough data to score.")
            return
        if vlab is not None and len(np.unique(vlab)) > 1:
            from sklearn.metrics import roc_auc_score, f1_score
            auroc = roc_auc_score(vlab, scores)
            th = np.percentile(scores, 90)
            f1 = f1_score(vlab, (scores > th).astype(int), zero_division=0)
            print(f"[CANShieldByte] AUROC={auroc:.4f}  F1@p90={f1:.4f}")
        return scores

    def predict(self, X_test=None, **kwargs):
        """Return binary anomaly predictions for the configured dataset."""
        ts, ids, by, labels = self._load_csv(self._dataset_csv())
        mat = self._normalise(self._build_matrix(ids, by))
        scores, _ = self._score(mat, labels)
        if scores is None:
            return None
        th = np.percentile(scores, 90)
        return (scores > th).astype(int)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path):
        """Save normalisation, ordering, thresholds, and AE weights."""
        os.makedirs(path, exist_ok=True)
        meta = {
            "top_n_ids": self.top_n_ids, "window": self.window,
            "periods": self.periods, "selected_ids": self.selected_ids,
            "id_to_col": self.id_to_col, "n_features": self.n_features,
            "feat_min": self.feat_min, "feat_max": self.feat_max,
            "order": self.order, "thresholds": self.thresholds,
            "rl_percentile": self.rl_percentile, "rt_percentile": self.rt_percentile,
        }
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        for T, model in self.models.items():
            model.save(os.path.join(path, f"ae_T{T}.keras"))

    def load(self, path):
        """Load a previously saved model directory."""
        import tensorflow as tf
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        self.__dict__.update(meta)
        self.models = {}
        for T in self.periods:
            p = os.path.join(path, f"ae_T{T}.keras")
            if os.path.exists(p):
                self.models[T] = tf.keras.models.load_model(p)
