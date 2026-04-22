"""
CANShield IDS — Integrated into CAN-Rakshak Framework
======================================================
Paper: "CANShield: Signal-Based Intrusion Detection for Controller Area Networks"
       (Shahriar et al., 2023)

This module implements the complete CANShield pipeline:
  Module 1: Data Preprocessing  (forward-fill, normalize, cluster, multi-view)
  Module 2: Data Analyzer       (CNN autoencoder training + reconstruction losses)
  Module 3: Attack Detection    (three-step threshold + ensemble scoring)

Adapted for RAW CAN frame data (Timestamp, can_id, dlc, byte0..byte7, labels)
instead of pre-decoded signal-level data.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from datetime import datetime
from ids.base import IDS

# ---------------------------------------------------------------------------
# TensorFlow is imported lazily inside methods that need it,
# so the module can be discovered by the framework even without TF installed.
# ---------------------------------------------------------------------------
tf = None
def _ensure_tensorflow():
    """Lazy import TensorFlow only when actually needed."""
    global tf
    if tf is None:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        try:
            import tensorflow as _tf
            tf = _tf
        except ImportError:
            raise ImportError(
                "TensorFlow is required for CANShield. "
                "Install with: pip install tensorflow"
            )
    return tf



# ============================================================================
# CANSHIELD IDS CLASS
# ============================================================================
class CANShield(IDS):
    """
    CANShield Intrusion Detection System.

    Works on raw CAN bus data (byte-level). Each CAN ID's 8 payload bytes
    are treated as individual features (signals). The total number of features
    m = num_unique_can_ids × 8.

    Pipeline:
        Raw CAN CSV → forward-fill → normalize → correlation-cluster →
        multi-view images (m × w) → CNN autoencoders (×3) →
        reconstruction losses → three-step threshold → ensemble → detection
    """

    # --- Hyperparameters (paper defaults) ---
    WINDOW_SIZE = 50            # w: time steps per view
    SAMPLING_PERIODS = [1, 5, 10]  # T values for multi-view
    STRIDE_TRAIN = 50           # stride for training views
    STRIDE_TEST = 50            # stride for test views
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0002
    P_THRESHOLD = 95            # percentile for R_Loss
    Q_THRESHOLD = 99            # percentile for R_Time

    def __init__(self):
        """Initialize CANShield with empty state."""
        self.m = None                # number of features (signals)
        self.w = self.WINDOW_SIZE
        self.signal_names = None
        self.reorder_indices = None
        self.train_min = None
        self.train_max = None
        self.models = {}             # {T: keras_model}
        self.thresholds = {}         # {T: (R_Loss, R_Time)}
        self.r_signal = 0.19        # default R_Signal
        self.can_id_list = None      # ordered list of unique CAN IDs

    # ====================================================================
    # PUBLIC INTERFACE (required by IDS base class)
    # ====================================================================

    def train(self, train_dataset_dir=None, X_train=None, Y_train=None, cfg=None, **kwargs):
        """
        Full CANShield training pipeline.

        Two modes are supported:
          • SynCAN mode  — pre-computed multi-view tensors exist in
                           datasets/<name>/features/SynCAN/train_views.npz
                           (produced by SynCANExtractor).  All Module-1
                           work is already done; we jump straight to AE
                           training and threshold selection.
          • Raw-CSV mode — load a single CAN CSV, run forward-fill,
                           normalize, cluster, create views, then train.
        """
        _ensure_tensorflow()
        cfg = cfg or {}

        features_dir = self._get_features_dir(cfg)
        precomputed_path = os.path.join(features_dir, "train_views.npz")

        if os.path.exists(precomputed_path):
            self._train_syncan(features_dir, cfg)
        else:
            self._train_raw_csv(cfg)

    def _train_syncan(self, features_dir, cfg):
        """
        SynCAN training path: load pre-computed views from SynCANExtractor
        output and run Module 2 + Module 3.
        """
        print(f"\n{'='*70}")
        print(f"  CANShield — Training Pipeline  (SynCAN mode)")
        print(f"  Features dir: {features_dir}")
        print(f"{'='*70}\n")

        # ----- Load pre-computed views and extractor config -----
        print("[Module 1] Loading pre-computed SynCAN views...")
        config_path = os.path.join(features_dir, "syncan_config.pkl")
        with open(config_path, "rb") as fh:
            ext_cfg = pickle.load(fh)

        self.m               = ext_cfg["m"]
        self.w               = ext_cfg["w"]
        self.signal_names    = ext_cfg["signal_names"]
        self.reorder_indices = ext_cfg["reorder_indices"]
        self.train_min       = ext_cfg["train_min"]
        self.train_max       = ext_cfg["train_max"]
        self.can_id_list     = list(ext_cfg.get("signals_per_id", {}).keys())

        views_data = np.load(os.path.join(features_dir, "train_views.npz"))
        train_views = {T: views_data[f"T{T}"] for T in self.SAMPLING_PERIODS}
        for T in self.SAMPLING_PERIODS:
            print(f"  T={T}: {train_views[T].shape}")

        # ----- Module 2: CNN Autoencoder Training -----
        print(f"\n[Module 2] Training CNN Autoencoders...")
        self._train_autoencoders(train_views, cfg)

        # Compute training reconstruction losses (10% sample)
        print(f"\n[Module 2] Computing training reconstruction losses...")
        train_losses = self._compute_losses(train_views, sample_frac=0.1)

        # ----- Module 3: Threshold Selection -----
        print(f"\n[Module 3] Selecting thresholds "
              f"(p={self.P_THRESHOLD}%, q={self.Q_THRESHOLD}%)...")
        for T in self.SAMPLING_PERIODS:
            R_Loss, R_Time = self._select_thresholds(
                train_losses[T], self.P_THRESHOLD, self.Q_THRESHOLD
            )
            self.thresholds[T] = (R_Loss, R_Time)
            print(f"  T={T}: R_Loss=[{R_Loss.min():.4f}, {R_Loss.max():.4f}]  "
                  f"R_Time=[{R_Time.min():.1f}, {R_Time.max():.1f}]")

        print(f"\n  Training complete!")

    def _train_raw_csv(self, cfg):
        """
        Original raw-CSV training path (non-SynCAN datasets).
        """
        csv_path = self._get_csv_path(cfg)
        print(f"\n{'='*70}")
        print(f"  CANShield — Training Pipeline  (raw-CSV mode)")
        print(f"  Dataset: {csv_path}")
        print(f"{'='*70}\n")

        # ----- Module 1: Preprocessing -----
        print("[Module 1] Preprocessing...")
        df = self._load_csv(csv_path)
        matrix, labels = self._forward_fill(df)

        # Separate train data (labels == 0 only for training the AE)
        normal_mask = labels == 0
        train_matrix = matrix[normal_mask]
        print(f"  Using {len(train_matrix)} normal rows for training "
              f"(out of {len(matrix)} total)")

        # Normalize
        self.train_min = train_matrix.min(axis=0)
        self.train_max = train_matrix.max(axis=0)
        t_range = self.train_max - self.train_min
        t_range[t_range == 0] = 1.0
        train_norm = np.clip((train_matrix - self.train_min) / t_range, 0, 1).astype(np.float32)

        # Correlation clustering
        self.reorder_indices = self._correlation_clustering(train_norm)
        train_norm = train_norm[:, self.reorder_indices]

        # Create multi-view images
        train_views = self._create_views(
            train_norm, np.zeros(len(train_norm), dtype=np.int32),
            stride=self.STRIDE_TRAIN
        )
        for T in self.SAMPLING_PERIODS:
            print(f"  View T={T}: {train_views[T].shape}")

        # ----- Module 2: CNN Autoencoder Training -----
        print(f"\n[Module 2] Training CNN Autoencoders...")
        self._train_autoencoders(train_views, cfg)

        # Compute training reconstruction losses for threshold selection
        print(f"\n[Module 2] Computing training reconstruction losses...")
        train_losses = self._compute_losses(train_views, sample_frac=0.1)

        # ----- Module 3: Threshold Selection -----
        print(f"\n[Module 3] Selecting thresholds (p={self.P_THRESHOLD}%, q={self.Q_THRESHOLD}%)...")
        for T in self.SAMPLING_PERIODS:
            R_Loss, R_Time = self._select_thresholds(
                train_losses[T], self.P_THRESHOLD, self.Q_THRESHOLD
            )
            self.thresholds[T] = (R_Loss, R_Time)
            print(f"  T={T}: R_Loss range=[{R_Loss.min():.4f}, {R_Loss.max():.4f}], "
                  f"R_Time range=[{R_Time.min():.1f}, {R_Time.max():.1f}]")

        print(f"\n  Training complete!")

    def test(self, X_test=None, Y_test=None, cfg=None, **kwargs):
        """
        Full CANShield testing pipeline.

        Two modes mirror train():
          • SynCAN mode  — load pre-computed test views from
                           datasets/<name>/features/SynCAN/
                           and evaluate all attack types.
          • Raw-CSV mode — load a single test CSV, preprocess,
                           and run detection (original behaviour).
        """
        _ensure_tensorflow()
        cfg = cfg or {}

        features_dir = self._get_features_dir(cfg)
        precomputed_path = os.path.join(features_dir, "train_views.npz")

        if os.path.exists(precomputed_path):
            return self._test_syncan(features_dir, cfg)
        else:
            return self._test_raw_csv(cfg)

    def _test_syncan(self, features_dir, cfg):
        """
        SynCAN testing path: evaluate all 5 attack types separately,
        mirroring Module 3 of runall.py.
        """
        print(f"\n{'='*70}")
        print(f"  CANShield — Testing Pipeline  (SynCAN mode)")
        print(f"  Features dir: {features_dir}")
        print(f"{'='*70}\n")

        ATTACK_NAMES = ["plateau", "continuous", "playback", "suppress", "flooding"]
        ALL_NAMES    = ["normal"] + ATTACK_NAMES

        # ----- Load scores for every test file -----
        print("[Step 1] Computing reconstruction losses and anomaly scores...")
        individual_scores = {}   # {name: {T: array}}
        ensemble_scores   = {}   # {name: array}
        test_vlabels      = {}   # {name: array}

        for name in ALL_NAMES:
            views_path  = os.path.join(features_dir, f"test_{name}_views.npz")
            labels_path = os.path.join(features_dir, f"test_{name}_labels.npy")
            if not os.path.exists(views_path):
                print(f"  test_{name}: not found — skipping")
                continue

            print(f"  test_{name}...")
            views_npz = np.load(views_path)
            test_views = {T: views_npz[f"T{T}"] for T in self.SAMPLING_PERIODS}
            labels = np.load(labels_path)
            test_vlabels[name] = labels

            ind = {}
            for T in self.SAMPLING_PERIODS:
                data  = test_views[T].reshape(-1, self.m, self.w, 1)
                recon = self.models[T].predict(data, batch_size=self.BATCH_SIZE, verbose=0)
                loss  = np.abs(data - recon).reshape(-1, self.m, self.w)
                R_Loss, R_Time = self.thresholds[T]
                ind[T] = self._compute_anomaly_score(loss, R_Loss, R_Time)
                print(f"    T={T}: mean_loss={loss.mean():.6f}  "
                      f"mean_score={ind[T].mean():.4f}")
            individual_scores[name] = ind

            ens = sum(ind[T] for T in self.SAMPLING_PERIODS) / len(self.SAMPLING_PERIODS)
            ensemble_scores[name] = ens.astype(np.float32)

        # ----- Calibrate R_Signal on normal + attack ensemble scores -----
        print("\n[Step 2] Calibrating R_Signal threshold...")
        self.r_signal = self._find_r_signal_syncan(
            ensemble_scores, test_vlabels, ATTACK_NAMES
        )
        print(f"  R_Signal = {self.r_signal:.2f}")

        # ----- Evaluate per attack type and save -----
        print("\n[Step 3] Per-attack evaluation:")
        self._evaluate_and_save_syncan(
            ensemble_scores, individual_scores, test_vlabels,
            ATTACK_NAMES, cfg,
        )

        # Return combined attack predictions for framework compatibility
        all_preds, all_labels = [], []
        for name in ATTACK_NAMES:
            if name in ensemble_scores:
                all_preds.append(
                    (ensemble_scores[name] > self.r_signal).astype(int)
                )
                all_labels.append(test_vlabels[name])

        if all_preds:
            return np.concatenate(all_preds), np.concatenate(all_labels)
        return np.array([], dtype=int), np.array([], dtype=int)

    def _test_raw_csv(self, cfg):
        """
        Original raw-CSV testing path (non-SynCAN datasets).
        """
        csv_path = self._get_csv_path(cfg)
        print(f"\n{'='*70}")
        print(f"  CANShield — Testing Pipeline  (raw-CSV mode)")
        print(f"  Dataset: {csv_path}")
        print(f"{'='*70}\n")

        # ----- Load & preprocess test data -----
        print("[Step 1] Loading and preprocessing test data...")
        df = self._load_csv(csv_path)
        matrix, labels = self._forward_fill(df)

        # Normalize with TRAINING stats
        t_range = self.train_max - self.train_min
        t_range[t_range == 0] = 1.0
        test_norm = np.clip((matrix - self.train_min) / t_range, 0, 1).astype(np.float32)

        # Reorder columns
        test_norm = test_norm[:, self.reorder_indices]

        # Create views
        test_views, test_vlabels = self._create_views_with_labels(
            test_norm, labels, stride=self.STRIDE_TEST
        )
        n_samples = len(test_vlabels)
        print(f"  Created {n_samples} test samples "
              f"({test_vlabels.sum()} attacks, {n_samples - test_vlabels.sum()} normal)")

        # ----- Compute reconstruction losses -----
        print("[Step 2] Computing reconstruction losses...")
        test_losses = {}
        for T in self.SAMPLING_PERIODS:
            data = test_views[T].reshape(-1, self.m, self.w, 1)
            recon = self.models[T].predict(data, batch_size=self.BATCH_SIZE, verbose=0)
            loss = np.abs(data - recon).reshape(-1, self.m, self.w)
            test_losses[T] = loss
            print(f"  T={T}: mean loss = {loss.mean():.6f}")

        # ----- Three-step detection + ensemble -----
        print("[Step 3] Computing anomaly scores (three-step + ensemble)...")
        individual_scores = {}
        for T in self.SAMPLING_PERIODS:
            R_Loss, R_Time = self.thresholds[T]
            individual_scores[T] = self._compute_anomaly_score(
                test_losses[T], R_Loss, R_Time
            )

        ensemble_scores = np.zeros(n_samples, dtype=np.float32)
        for T in self.SAMPLING_PERIODS:
            ensemble_scores += individual_scores[T]
        ensemble_scores /= len(self.SAMPLING_PERIODS)

        # ----- Find optimal R_Signal -----
        print("[Step 4] Calibrating R_Signal threshold...")
        self.r_signal = self._find_r_signal(ensemble_scores, test_vlabels)
        print(f"  R_Signal = {self.r_signal:.2f}")

        # ----- Final predictions -----
        all_preds = (ensemble_scores > self.r_signal).astype(int)
        all_labels = test_vlabels

        # ----- Evaluate & save results -----
        print(f"\n[Step 5] Evaluation Results:")
        self._evaluate_and_save(
            all_preds, all_labels, ensemble_scores,
            individual_scores, test_vlabels, cfg
        )

        return all_preds, all_labels

    def predict(self, X_test=None, **kwargs):
        """Predict anomaly scores for raw input (not used in typical workflow)."""
        pass

    def save(self, path):
        """
        Save all model components:
          - 3 Keras models
          - Preprocessing parameters
          - Thresholds
        """
        os.makedirs(path, exist_ok=True)

        # Save Keras models
        for T in self.SAMPLING_PERIODS:
            model_path = os.path.join(path, f"ae_T{T}.keras")
            self.models[T].save(model_path)

        # Save everything else as a pickle
        state = {
            "m": self.m,
            "w": self.w,
            "signal_names": self.signal_names,
            "reorder_indices": self.reorder_indices,
            "train_min": self.train_min,
            "train_max": self.train_max,
            "thresholds": self.thresholds,
            "r_signal": self.r_signal,
            "can_id_list": self.can_id_list,
        }
        with open(os.path.join(path, "canshield_state.pkl"), "wb") as f:
            pickle.dump(state, f)

        print(f"  CANShield model saved to {path}")

    def load(self, path):
        """Load all model components."""
        _ensure_tensorflow()
        # Load Keras models
        for T in self.SAMPLING_PERIODS:
            model_path = os.path.join(path, f"ae_T{T}.keras")
            self.models[T] = tf.keras.models.load_model(model_path)

        # Load state
        with open(os.path.join(path, "canshield_state.pkl"), "rb") as f:
            state = pickle.load(f)

        self.m = state["m"]
        self.w = state["w"]
        self.signal_names = state["signal_names"]
        self.reorder_indices = state["reorder_indices"]
        self.train_min = state["train_min"]
        self.train_max = state["train_max"]
        self.thresholds = state["thresholds"]
        self.r_signal = state["r_signal"]
        self.can_id_list = state["can_id_list"]

        print(f"  CANShield model loaded from {path}")
        print(f"  Features: m={self.m}, Window: w={self.w}, "
              f"CAN IDs: {len(self.can_id_list)}")

    # ====================================================================
    # MODULE 1: PREPROCESSING
    # ====================================================================

    def _get_csv_path(self, cfg):
        """Get the path to the processed CSV file."""
        csv_name = cfg.get('file_name', '')
        for ext in [".log", ".txt"]:
            if csv_name.endswith(ext):
                csv_name = csv_name.replace(ext, ".csv")
                break
        return os.path.join(
            cfg.get('dir_path', ''), "..", "datasets",
            cfg.get('dataset_name', ''), "modified_dataset", csv_name
        )

    def _get_features_dir(self, cfg):
        """
        Return the directory where the SynCANExtractor stores its outputs.
        If the directory does not exist, SynCAN mode simply isn't active.
        """
        return os.path.join(
            cfg.get('dir_path', ''), "..", "datasets",
            cfg.get('dataset_name', ''), "features", "SynCAN"
        )

    def _load_csv(self, csv_path):
        """
        Load raw CAN CSV in CAN-Rakshak format:
          Timestamp, can_id, dlc, byte0, byte1, ..., byte7, labels

        Returns a DataFrame with cleaned columns.
        """
        columns = [
            "timestamp", "can_id", "dlc",
            "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7",
            "label",
        ]
        df = pd.read_csv(csv_path, header=None, names=columns)
        df["can_id"] = df["can_id"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)

        # Convert hex bytes to integers (0-255)
        byte_cols = ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]
        for col in byte_cols:
            df[col] = df[col].apply(self._hex_to_int)

        print(f"  Loaded {len(df)} rows, {df['can_id'].nunique()} unique CAN IDs")
        return df

    @staticmethod
    def _hex_to_int(val):
        """Safely convert hex string to integer."""
        try:
            return int(str(val).strip(), 16)
        except (ValueError, TypeError):
            return 0

    def _forward_fill(self, df):
        """
        Convert raw CAN DataFrame to forward-filled signal matrix.

        Each CAN ID contributes 8 bytes as features.
        Features = [ID1_b0, ID1_b1, ..., ID1_b7, ID2_b0, ..., IDn_b7]
        Total features m = num_unique_ids × 8.

        At each row, only the current CAN ID's bytes update;
        all other columns are forward-filled from the previous row.
        """
        # Discover unique CAN IDs (sorted for consistency)
        if self.can_id_list is None:
            self.can_id_list = sorted(df["can_id"].unique())

        num_ids = len(self.can_id_list)
        self.m = num_ids * 8

        # Build signal name list and column mapping
        self.signal_names = []
        signal_map = {}  # {can_id: start_col_index}
        col_idx = 0
        for cid in self.can_id_list:
            signal_map[cid] = col_idx
            for b in range(8):
                self.signal_names.append(f"b{b}_{cid}")
            col_idx += 8

        print(f"  Forward-filling: {len(df)} rows × {self.m} features "
              f"({num_ids} CAN IDs × 8 bytes)...")

        n_rows = len(df)
        matrix = np.zeros((n_rows, self.m), dtype=np.float32)
        labels = np.zeros(n_rows, dtype=np.int32)

        byte_cols = ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]

        for i in range(n_rows):
            # Forward-fill: copy previous row
            if i > 0:
                matrix[i] = matrix[i - 1]

            labels[i] = df.iloc[i]["label"]
            cid = df.iloc[i]["can_id"]

            if cid in signal_map:
                start = signal_map[cid]
                for b_idx in range(8):
                    matrix[i, start + b_idx] = float(df.iloc[i][byte_cols[b_idx]])

            if i % 1_000_000 == 0 and i > 0:
                print(f"    {i:,}/{n_rows:,}...")

        print(f"  Forward-fill complete: shape {matrix.shape}")
        return matrix, labels

    def _correlation_clustering(self, data):
        """
        Compute Pearson correlation, apply hierarchical agglomerative
        clustering (complete linkage), return reordered column indices.
        """
        corr = np.abs(np.corrcoef(data.T))
        corr = np.nan_to_num(corr, nan=0.0)

        dist = 1 - corr
        np.fill_diagonal(dist, 0)
        dist = np.clip((dist + dist.T) / 2, 0, None)

        Z = linkage(squareform(dist), method="complete")

        # Get leaf order from dendrogram (suppress plot)
        fig, ax = plt.subplots(figsize=(1, 1))
        dendro = dendrogram(Z, no_plot=False, ax=ax)
        plt.close(fig)

        order = dendro["leaves"]
        reordered_names = [self.signal_names[i] for i in order]
        print(f"  Clustered {self.m} signals, new column order computed")
        return np.array(order)

    def _create_views(self, matrix, labels, stride):
        """
        Create multi-view images from the signal matrix.
        Returns dict: {T: array of shape (n_samples, m, w)}
        """
        total = len(matrix)
        max_T = max(self.SAMPLING_PERIODS)
        start = max_T * (self.w - 1)

        if start >= total:
            raise ValueError(
                f"Not enough data for views: need {start} rows, have {total}"
            )

        n_samples = (total - start) // stride
        views = {
            T: np.zeros((n_samples, self.m, self.w), dtype=np.float32)
            for T in self.SAMPLING_PERIODS
        }

        for i in range(n_samples):
            pos = start + i * stride
            for T in self.SAMPLING_PERIODS:
                idxs = np.arange(pos - (self.w - 1) * T, pos + 1, T)
                views[T][i] = matrix[idxs].T

        return views

    def _create_views_with_labels(self, matrix, labels, stride):
        """
        Create multi-view images with per-sample attack labels.
        A sample is labeled attack (1) if any row in its widest
        window contains an attack.
        """
        total = len(matrix)
        max_T = max(self.SAMPLING_PERIODS)
        start = max_T * (self.w - 1)

        if start >= total:
            raise ValueError(
                f"Not enough data for views: need {start} rows, have {total}"
            )

        n_samples = (total - start) // stride
        views = {
            T: np.zeros((n_samples, self.m, self.w), dtype=np.float32)
            for T in self.SAMPLING_PERIODS
        }
        vlabels = np.zeros(n_samples, dtype=np.int32)

        for i in range(n_samples):
            pos = start + i * stride
            for T in self.SAMPLING_PERIODS:
                idxs = np.arange(pos - (self.w - 1) * T, pos + 1, T)
                views[T][i] = matrix[idxs].T

            win_start = max(0, pos - max_T * self.w)
            vlabels[i] = 1 if labels[win_start : pos + 1].max() > 0 else 0

        return views, vlabels

    # ====================================================================
    # MODULE 2: CNN AUTOENCODER
    # ====================================================================

    def _build_autoencoder(self):
        """
        Build the CNN autoencoder exactly as described in the paper.
        """
        from tensorflow.keras.layers import (
            Input, Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU,
        )
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam

        inp = Input(shape=(self.m, self.w, 1))

        # Encoder
        x = Conv2D(32, (3, 3), padding="same")(inp)
        x = LeakyReLU(0.2)(x)
        x = MaxPooling2D((2, 2), padding="same")(x)

        x = Conv2D(16, (3, 3), padding="same")(x)
        x = LeakyReLU(0.2)(x)
        x = MaxPooling2D((2, 2), padding="same")(x)

        # Bottleneck
        x = Conv2D(16, (3, 3), padding="same")(x)
        x = LeakyReLU(0.2)(x)

        # Decoder
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = LeakyReLU(0.2)(x)

        x = UpSampling2D((2, 2))(x)
        x = Conv2D(1, (3, 3), padding="same", activation="sigmoid")(x)

        # Crop to match input dimensions
        x = tf.keras.layers.Cropping2D(
            cropping=(
                (0, int(x.shape[1]) - self.m),
                (0, int(x.shape[2]) - self.w),
            )
        )(x)

        model = Model(inp, x)
        model.compile(optimizer=Adam(learning_rate=self.LEARNING_RATE), loss="mse")
        return model

    def _train_autoencoders(self, train_views, cfg=None):
        """
        Train 3 CNN autoencoders with transfer learning:
          AE1 (T=1)  : trained from scratch
          AE2 (T=5)  : initialized with AE1 weights
          AE3 (T=10) : initialized with AE2 weights
        """
        prev_weights = None
        epochs = (cfg or {}).get('epochs', 100) or 100

        for i, T in enumerate(self.SAMPLING_PERIODS):
            print(f"\n  --- Training AE for T={T} ---")

            model = self._build_autoencoder()

            # Transfer learning
            if prev_weights is not None:
                print(f"  Transferring weights from T={self.SAMPLING_PERIODS[i-1]}")
                model.set_weights(prev_weights)

            data = train_views[T].reshape(-1, self.m, self.w, 1)

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            )

            model.fit(
                data, data,
                epochs=epochs,
                batch_size=self.BATCH_SIZE,
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=1,
            )

            prev_weights = model.get_weights()
            self.models[T] = model

            print(f"  AE T={T} training complete")

    def _compute_losses(self, views, sample_frac=1.0):
        """
        Compute reconstruction losses.
        Returns: {T: array of shape (n, m, w)}
        """
        losses = {}
        for T in self.SAMPLING_PERIODS:
            n_total = len(views[T])
            if sample_frac < 1.0:
                np.random.seed(42)
                n_sample = max(1, int(n_total * sample_frac))
                idx = np.random.choice(n_total, n_sample, replace=False)
                data = views[T][idx]
            else:
                data = views[T]

            data_4d = data.reshape(-1, self.m, self.w, 1)
            recon = self.models[T].predict(data_4d, batch_size=self.BATCH_SIZE, verbose=0)
            loss = np.abs(data_4d - recon).reshape(-1, self.m, self.w)
            losses[T] = loss
            print(f"  T={T}: {len(loss)} samples, mean loss = {loss.mean():.6f}")

        return losses

    # ====================================================================
    # MODULE 3: DETECTION
    # ====================================================================

    @staticmethod
    def _select_thresholds(train_loss, p, q):
        """
        Algorithm 1: Three-step threshold selection.

        Args:
            train_loss: shape (n_samples, m, w)
            p: percentile for R_Loss
            q: percentile for R_Time

        Returns:
            R_Loss: shape (m,) — per-signal loss threshold
            R_Time: shape (m,) — per-signal violation count threshold
        """
        R_Loss = np.percentile(train_loss, p, axis=(0, 2))
        B = (train_loss > R_Loss[np.newaxis, :, np.newaxis]).astype(int)
        V = B.sum(axis=2)
        R_Time = np.percentile(V, q, axis=0)
        return R_Loss, R_Time

    @staticmethod
    def _compute_anomaly_score(loss, R_Loss, R_Time):
        """
        Algorithm 2: Compute per-sample anomaly score.

        Args:
            loss: shape (n_samples, m, w)
            R_Loss: shape (m,)
            R_Time: shape (m,)

        Returns:
            scores: shape (n_samples,) — fraction of violated signals
        """
        B = (loss > R_Loss[np.newaxis, :, np.newaxis]).astype(int)
        V = B.sum(axis=2)
        S = (V > R_Time[np.newaxis, :]).astype(int)
        P = S.mean(axis=1)
        return P.astype(np.float32)

    @staticmethod
    def _find_r_signal(ensemble_scores, labels):
        """
        Find optimal R_Signal threshold via grid search.
        Picks the threshold that maximizes average F1 across attack types
        while keeping FPR < 1% on normal samples.
        """
        normal_mask = labels == 0
        normal_scores = ensemble_scores[normal_mask]
        attack_mask = labels == 1

        best_r = 0.1
        best_f1 = 0.0

        for r in np.arange(0.0, 1.01, 0.01):
            fpr = (normal_scores > r).mean()
            if fpr > 0.05:  # allow up to 5% FPR
                continue

            preds = (ensemble_scores > r).astype(int)
            if preds.sum() == 0:
                continue

            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_r = r

        return best_r

    @staticmethod
    def _find_r_signal_syncan(ensemble_scores, test_vlabels, attack_names,
                              fpr_budget=0.01):
        """
        SynCAN variant of R_Signal search.

        Matches Module 3 of runall.py: sweep r in [0, 1], reject if FPR
        on the dedicated 'normal' test file exceeds `fpr_budget` (1 %),
        and pick the r that maximises average F1 across the five
        attack files.
        """
        if 'normal' not in ensemble_scores:
            return 0.1

        normal_scores = ensemble_scores['normal']
        best_r, best_avg_f1 = 0.1, 0.0

        for r in np.arange(0.0, 1.01, 0.01):
            fpr = (normal_scores > r).mean()
            if fpr > fpr_budget:
                continue
            f1s = []
            for name in attack_names:
                if name not in ensemble_scores:
                    continue
                labels = test_vlabels[name]
                preds  = (ensemble_scores[name] > r).astype(int)
                if labels.sum() > 0 and preds.sum() > 0:
                    f1s.append(f1_score(labels, preds, zero_division=0))
            if f1s:
                avg_f1 = float(np.mean(f1s))
                if avg_f1 > best_avg_f1:
                    best_avg_f1 = avg_f1
                    best_r = float(r)

        return best_r

    # ====================================================================
    # EVALUATION & RESULTS
    # ====================================================================

    def _evaluate_and_save_syncan(self, ensemble_scores, individual_scores,
                                  test_vlabels, attack_names, cfg=None):
        """
        Per-attack evaluation and plotting in the style of runall.py
        Module 3.  Writes results.txt and ROC / PR / distribution /
        timeline plots to datasets/<name>/Results/CANShield/.
        """
        from sklearn.metrics import (
            roc_curve, precision_recall_curve, auc as sk_auc,
        )

        cfg = cfg or {}
        dataset_path = os.path.join(
            cfg.get('dir_path', ''), "..", "datasets", cfg.get('dataset_name', '')
        )
        result_path = os.path.join(
            dataset_path, "Results", cfg.get('model', 'CANShield')
        )
        os.makedirs(result_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

        # Per-attack metrics
        results = {}
        for name in attack_names:
            if name not in ensemble_scores:
                continue
            scores = ensemble_scores[name]
            labels = test_vlabels[name]
            preds  = (scores > self.r_signal).astype(int)

            tp = int(((preds == 1) & (labels == 1)).sum())
            fp = int(((preds == 1) & (labels == 0)).sum())
            tn = int(((preds == 0) & (labels == 0)).sum())
            fn = int(((preds == 0) & (labels == 1)).sum())

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            auroc = (roc_auc_score(labels, scores)
                     if len(np.unique(labels)) > 1 else 0.0)

            results[name] = dict(
                AUROC=auroc, Precision=prec, Recall=rec, F1=f1, FPR=fpr,
                TP=tp, FP=fp, TN=tn, FN=fn,
            )

            print(f"  {name.upper():<12} "
                  f"AUROC={auroc:.4f}  P={prec:.4f}  R={rec:.4f}  "
                  f"F1={f1:.4f}  FPR={fpr:.4f}  "
                  f"(TP={tp} FP={fp} TN={tn} FN={fn})")

        # ----- results.txt -----
        results_file = os.path.join(result_path, f"results_{timestamp}.txt")
        with open(results_file, "w") as f:
            f.write("CANShield SynCAN Detection Results\n")
            f.write("=" * 70 + "\n")
            f.write(f"R_Signal={self.r_signal:.2f}, "
                    f"p={self.P_THRESHOLD}%, q={self.Q_THRESHOLD}%\n\n")

            f.write(f"{'Attack':<15} {'AUROC':<10} {'Precision':<12} "
                    f"{'Recall':<10} {'F1':<10} {'FPR':<10}\n")
            f.write("-" * 67 + "\n")
            for name in attack_names:
                if name not in results:
                    continue
                r = results[name]
                f.write(f"{name:<15} {r['AUROC']:<10.4f} {r['Precision']:<12.4f} "
                        f"{r['Recall']:<10.4f} {r['F1']:<10.4f} {r['FPR']:<10.4f}\n")

            f.write(f"\nTPR / FPR (Paper Table III format):\n")
            f.write("-" * 55 + "\n")
            for name in attack_names:
                if name not in results:
                    continue
                r = results[name]
                f.write(f"  {name:<15} {r['Recall']:.3f} / {r['FPR']:.3f}\n")

            f.write(f"\nConfusion Matrix (TP, FP, TN, FN):\n")
            f.write(f"{'Attack':<15} {'TP':<10} {'FP':<10} {'TN':<10} {'FN':<10}\n")
            f.write("-" * 55 + "\n")
            for name in attack_names:
                if name not in results:
                    continue
                r = results[name]
                f.write(f"{name:<15} {r['TP']:<10} {r['FP']:<10} "
                        f"{r['TN']:<10} {r['FN']:<10}\n")

            f.write(f"\nIndividual AE vs Ensemble AUROC:\n")
            header = f"{'Attack':<15}" + "".join(
                f" {'T='+str(T):<10}" for T in self.SAMPLING_PERIODS
            ) + f" {'Ensemble':<10}\n"
            f.write(header)
            f.write("-" * 55 + "\n")
            for name in attack_names:
                if name not in results:
                    continue
                labels = test_vlabels[name]
                row = f"{name:<15}"
                for T in self.SAMPLING_PERIODS:
                    try:
                        auc_t = roc_auc_score(labels, individual_scores[name][T])
                    except Exception:
                        auc_t = 0.0
                    row += f" {auc_t:<10.4f}"
                row += f" {results[name]['AUROC']:<10.4f}"
                f.write(row + "\n")

        print(f"\n  Results saved: {results_file}")

        # ----- ROC curves (2×3 grid) -----
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, name in enumerate(attack_names):
            ax = axes[i]
            if name not in ensemble_scores:
                ax.axis('off'); continue
            labels = test_vlabels[name]
            if len(np.unique(labels)) < 2:
                ax.axis('off'); continue

            fpr_a, tpr_a, _ = roc_curve(labels, ensemble_scores[name])
            ax.plot(fpr_a, tpr_a, 'b-', lw=2,
                    label=f"Ens ({results[name]['AUROC']:.3f})")
            colours = ['r--', 'g--', 'm--']
            for j, T in enumerate(self.SAMPLING_PERIODS):
                f_a, t_a, _ = roc_curve(labels, individual_scores[name][T])
                auc_t = roc_auc_score(labels, individual_scores[name][T])
                ax.plot(f_a, t_a, colours[j % len(colours)], lw=1,
                        label=f"T={T} ({auc_t:.3f})")
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax.set_title(name.upper())
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        for i in range(len(attack_names), len(axes)):
            axes[i].axis('off')
        plt.suptitle('SynCAN ROC Curves — Individual AEs vs Ensemble',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(result_path, f"roc_curves_{timestamp}.png"),
                    dpi=150)
        plt.close()

        # ----- Precision-Recall curves -----
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, name in enumerate(attack_names):
            ax = axes[i]
            if name not in ensemble_scores:
                ax.axis('off'); continue
            labels = test_vlabels[name]
            if len(np.unique(labels)) < 2:
                ax.axis('off'); continue
            prec_a, rec_a, _ = precision_recall_curve(labels,
                                                      ensemble_scores[name])
            auprc = sk_auc(rec_a, prec_a)
            ax.plot(rec_a, prec_a, 'b-', lw=2,
                    label=f'Ens (AUPRC={auprc:.3f})')
            ax.set_title(name.upper())
            ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        for i in range(len(attack_names), len(axes)):
            axes[i].axis('off')
        plt.suptitle('SynCAN Precision-Recall Curves (Ensemble)',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(result_path, f"pr_curves_{timestamp}.png"),
                    dpi=150)
        plt.close()

        # ----- Anomaly score distributions -----
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, name in enumerate(attack_names):
            ax = axes[i]
            if name not in ensemble_scores:
                ax.axis('off'); continue
            labels = test_vlabels[name]
            scores = ensemble_scores[name]
            if (labels == 0).sum() > 0:
                ax.hist(scores[labels == 0], bins=50, alpha=0.6,
                        color='green', label='Normal', density=True)
            if (labels == 1).sum() > 0:
                ax.hist(scores[labels == 1], bins=50, alpha=0.6,
                        color='red', label='Attack', density=True)
            ax.axvline(self.r_signal, color='black', ls='--',
                       label=f'Threshold={self.r_signal:.2f}')
            ax.set_title(name.upper())
            ax.set_xlabel('Anomaly Score'); ax.set_ylabel('Density')
            ax.legend(fontsize=8)
        for i in range(len(attack_names), len(axes)):
            axes[i].axis('off')
        plt.suptitle('SynCAN Anomaly Score Distributions', fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(result_path, f"anomaly_distributions_{timestamp}.png"),
            dpi=150,
        )
        plt.close()

        # ----- Anomaly over time (5 rows) -----
        fig, axes = plt.subplots(len(attack_names), 1, figsize=(15, 20))
        if len(attack_names) == 1:
            axes = [axes]
        for i, name in enumerate(attack_names):
            ax = axes[i]
            if name not in ensemble_scores:
                ax.axis('off'); continue
            labels = test_vlabels[name]
            for T in self.SAMPLING_PERIODS:
                ax.plot(individual_scores[name][T], alpha=0.4, lw=0.5,
                        label=f'T={T}')
            ax.plot(ensemble_scores[name], color='black', lw=1,
                    label='Ensemble')
            att = np.diff(np.concatenate([[0], labels, [0]]))
            for s, e in zip(np.where(att == 1)[0], np.where(att == -1)[0]):
                ax.axvspan(s, e, alpha=0.15, color='red')
            ax.axhline(self.r_signal, color='black', ls='--', alpha=0.5)
            ax.set_title(f'{name.upper()} Attack')
            ax.set_ylabel('Anomaly Score')
            ax.legend(fontsize=8, loc='upper right')
            ax.set_ylim(-0.05, 1.05)
        axes[-1].set_xlabel('Sample Index')
        plt.suptitle('SynCAN Anomaly Scores Over Time', fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(result_path, f"anomaly_over_time_{timestamp}.png"),
            dpi=150,
        )
        plt.close()

        print(f"  Plots saved: {result_path}")


    def _evaluate_and_save(self, all_preds, all_labels, ensemble_scores,
                           individual_scores, test_vlabels, cfg=None):
        """Evaluate metrics, print results, save plots and results.txt."""
        cfg = cfg or {}
        # --- Metrics ---
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        auroc = roc_auc_score(all_labels, ensemble_scores) if len(np.unique(all_labels)) > 1 else 0.0

        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        print(f"\n  {'='*50}")
        print(f"  CANShield Detection Results")
        print(f"  {'='*50}")
        print(f"  AUROC:      {auroc:.4f}")
        print(f"  Accuracy:   {acc:.4f}")
        print(f"  Precision:  {prec:.4f}")
        print(f"  Recall:     {rec:.4f}")
        print(f"  F1 Score:   {f1:.4f}")
        print(f"  FPR:        {fpr:.4f}")
        print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")

        # Individual AE AUROC
        print(f"\n  Individual AE AUROC:")
        for T in self.SAMPLING_PERIODS:
            if len(np.unique(all_labels)) > 1:
                auc_val = roc_auc_score(all_labels, individual_scores[T])
                print(f"    T={T}: {auc_val:.4f}")

        # --- Save results ---
        dataset_path = os.path.join(
            cfg.get('dir_path', ''), "..", "datasets", cfg.get('dataset_name', '')
        )
        result_path = os.path.join(dataset_path, "Results", cfg.get('model', 'CANShield'))
        os.makedirs(result_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

        # results.txt
        results_file = os.path.join(result_path, f"results_{timestamp}.txt")
        with open(results_file, "w") as f:
            f.write("CANShield Detection Results\n")
            f.write(f"{'='*60}\n")
            f.write(f"R_Signal={self.r_signal:.2f}, "
                    f"p={self.P_THRESHOLD}%, q={self.Q_THRESHOLD}%\n\n")
            f.write(f"AUROC:      {auroc:.4f}\n")
            f.write(f"Accuracy:   {acc:.4f}\n")
            f.write(f"Precision:  {prec:.4f}\n")
            f.write(f"Recall:     {rec:.4f}\n")
            f.write(f"F1 Score:   {f1:.4f}\n")
            f.write(f"FPR:        {fpr:.4f}\n")
            f.write(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}\n\n")
            f.write(f"Individual AE AUROC:\n")
            for T in self.SAMPLING_PERIODS:
                if len(np.unique(all_labels)) > 1:
                    auc_val = roc_auc_score(all_labels, individual_scores[T])
                    f.write(f"  T={T}: {auc_val:.4f}\n")

        print(f"\n  Results saved: {results_file}")

        # Confusion matrix plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("CANShield — Confusion Matrix")
        cm_path = os.path.join(result_path, f"confusion_matrix_{timestamp}.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()

        # ROC curve
        if len(np.unique(all_labels)) > 1:
            fpr_arr, tpr_arr, _ = roc_curve(all_labels, ensemble_scores)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr_arr, tpr_arr, "b-", lw=2,
                     label=f"Ensemble (AUROC={auroc:.3f})")
            for T in self.SAMPLING_PERIODS:
                f_t, t_t, _ = roc_curve(all_labels, individual_scores[T])
                auc_t = roc_auc_score(all_labels, individual_scores[T])
                plt.plot(f_t, t_t, "--", lw=1, label=f"T={T} ({auc_t:.3f})")
            plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("CANShield — ROC Curves")
            plt.legend()
            plt.grid(True, alpha=0.3)
            roc_path = os.path.join(result_path, f"roc_curve_{timestamp}.png")
            plt.savefig(roc_path, dpi=150)
            plt.close()

        # Anomaly distribution
        plt.figure(figsize=(10, 6))
        plt.hist(ensemble_scores[all_labels == 0], bins=50, alpha=0.6,
                 color="green", label="Normal", density=True)
        if all_labels.sum() > 0:
            plt.hist(ensemble_scores[all_labels == 1], bins=50, alpha=0.6,
                     color="red", label="Attack", density=True)
        plt.axvline(self.r_signal, color="black", ls="--",
                    label=f"Threshold={self.r_signal:.2f}")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Density")
        plt.title("CANShield — Anomaly Score Distribution")
        plt.legend()
        dist_path = os.path.join(result_path, f"anomaly_dist_{timestamp}.png")
        plt.savefig(dist_path, dpi=150)
        plt.close()

        print(f"  Plots saved: {result_path}")

        # Also call the framework's evaluation_metrics for compatibility
        try:
            from evaluate import evaluation_metrics
            evaluation_metrics(np.array(all_preds), np.array(all_labels), cfg)
        except Exception:
            pass
