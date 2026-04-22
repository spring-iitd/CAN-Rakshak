# CAN Rakshak — Developer Guide

> Complete documentation for pipeline execution, module architecture, data formats, and extension patterns.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Configuration System](#3-configuration-system)
4. [Pipeline Execution Flow](#4-pipeline-execution-flow)
5. [Stage 1 — Dataset Processing](#5-stage-1--dataset-processing)
6. [Stage 2 — Training](#6-stage-2--training)
7. [Stage 3 — Testing & Evaluation](#7-stage-3--testing--evaluation)
8. [Stage 4 — Adversarial Attacks](#8-stage-4--adversarial-attacks)
9. [Stage 5 — Adversarial Retraining](#9-stage-5--adversarial-retraining)
10. [IDS Model Implementations](#10-ids-model-implementations)
11. [Utility Modules](#11-utility-modules)
12. [Dynamic Class Loading Pattern](#12-dynamic-class-loading-pattern)
13. [Data Formats](#13-data-formats)
14. [Dependencies](#14-dependencies)
15. [Quick Reference Table](#15-quick-reference-table)

> **Extension guidance is embedded within each stage section.** Look for the _Extending Stage N_ subsection at the end of Stages 1–5.

---

## 1. Project Overview

**CAN Rakshak** is a modular adversarial machine learning research pipeline for CAN (Controller Area Network) bus Intrusion Detection Systems (IDS). It supports:

- Raw CAN log preprocessing and feature extraction
- Deep learning and classical ML IDS model training/testing
- Adversarial example generation via genetic algorithms
- Adversarial robustness retraining (adversarial training defense)
- Evaluation metrics, confusion matrices, and attack success rate (ASR) computation

All components (datasets, feature extractors, models, splitters, attacks, defenses) are pluggable and are auto-discovered at runtime. Adding a new component requires only creating a file that inherits the correct abstract base class.

---

## 2. Directory Structure

```
CAN_Rakshak/
├── driver.py                           # Main entry point — orchestrates all pipeline stages
├── run_analysis.py                     # Standalone dataset statistical analysis tool
├── requirements.txt
│
├── src/
│   ├── config.yaml                     # Primary configuration file
│   ├── common_imports.py               # Centralized import hub (stdlib, numpy, TF, PyTorch, sklearn)
│   ├── utilities.py                    # Hex/binary conversion, data balancing, sequencing
│   ├── base_preprocessor.py            # Abstract base class for dataset-specific preprocessors
│   ├── preprocessing.py                # Preprocessor loader and runner
│   ├── get_extractor.py                # Feature extractor factory
│   ├── get_splitter.py                 # Data splitter factory
│   ├── get_ids.py                      # IDS model factory
│   ├── get_attack.py                   # Attack factory
│   ├── train.py                        # Training wrapper (train_model, retrain_model)
│   ├── test.py                         # Test wrapper (test_model)
│   ├── evaluate.py                     # Metrics computation (accuracy, F1, ASR, confusion matrix)
│   ├── retraining.py                   # Adversarial retraining orchestrator
│   ├── dataset_analysis.py             # Statistical analysis & plotting functions
│   └── postprocessing.py               # Stub — reserved for future post-processing
│
├── ids/                                # IDS model implementations
│   ├── __init__.py                     # Auto-discovery loader
│   ├── base.py                         # Abstract IDS base class
│   ├── FrameInceptionResNet.py         # Primary model: Inception-ResNet-V1
│   ├── Resnet.py / Resnet50.py
│   ├── densenet121.py / densenet161.py
│   ├── convnxtBase.py
│   ├── MLP.py                          # PyTorch MLP
│   ├── RandomForest.py                 # scikit-learn RF
│   ├── DecisionTree.py
│   ├── shannon.py                      # Entropy-based anomaly detector
│   └── corrected_shannon.py
│
├── features/
│   ├── feature_extractors/
│   │   ├── __init__.py                 # Auto-discovery loader
│   │   ├── base.py                     # Abstract FeatureExtractor base class
│   │   ├── frame_builder.py            # FrameBuilder: 29×29 binary CAN frames
│   │   ├── stat_features.py            # Stat: statistical feature vector
│   │   └── pixnet.py                   # PixNet: under publication (stub)
│   └── image/
│       ├── traffic_encoder.py
│       ├── traffic_decoder.py
│       ├── data_frame.py
│       └── extract_feature_images.py
│
├── splitters/
│   ├── __init__.py                     # Auto-discovery loader
│   ├── base.py                         # Abstract BaseSplitter
│   ├── default.py                      # 2-way train/test split
│   └── threeway.py                     # 3-way surrogate/target/test split
│
├── attacks/
│   ├── attack_handler/
│   │   ├── __init__.py                 # Auto-discovery loader
│   │   ├── base.py                     # Abstract Attack / EvasionAttack / GeneticAttack
│   │   ├── genetic_attack.py           # GeneticAdvAttack: routes to DoS/Fuzzy/Spoof
│   │   └── FGSM.py                     # FGSM evasion attack (partial)
│   └── Genetic_algorithm/
│       ├── Adversarial_DoS.py
│       ├── Adversarial_Fuzzy.py
│       └── Adversarial_Spoof.py
│
├── defense/
│   ├── base.py                         # Abstract BaseDefense
│   └── retrainers/
│       └── FrameBuilderRetrainer.py    # Adversarial retraining for FrameBuilder models
│
├── datasets/
│   ├── CarHackingDataset/
│   │   ├── preprocess_dataset.py       # Dataset-specific preprocessor
│   │   ├── original_dataset/           # Raw input files (created at runtime)
│   │   ├── modified_dataset/           # Cleaned CSVs (created at runtime)
│   │   ├── features/Frames/            # Extracted frame/label CSVs
│   │   ├── train/                      # Train splits (.csv + .npz)
│   │   ├── test/                       # Test splits (.csv + .npz)
│   │   └── Results/                    # Metrics, confusion matrices, attack results
│   ├── CANIntrusionDataset/
│   ├── CARLA/
│   ├── MIRGU/
│   └── Demo_dataset/
│
└── models/                             # Saved model checkpoints (.h5)
```

---

## 3. Configuration System

All pipeline behavior is controlled by `src/config.yaml`. The `driver.py` reads the YAML and builds a flat `cfg` dictionary that is passed to every pipeline function.

### `src/config.yaml` — Annotated Reference

```yaml
# ─── Pipeline stage toggles ───────────────────────────────────────────────────
run_steps:
  dataset_processing: true      # Stage 1: preprocess → extract features → split
  training: true                # Stage 2: train IDS model
  testing: true                 # Stage 3: evaluate IDS model
  adversarial_perturbation: true  # Stage 4: generate adversarial examples
  robust_training: true         # Stage 5: retrain on adversarial examples

# ─── Dataset selection ────────────────────────────────────────────────────────
dataset_name: CarHackingDataset   # Must match a subdirectory in datasets/
file_name: DoS_target.csv         # Input CSV inside the dataset directory

# ─── Stage 1: Dataset processing ─────────────────────────────────────────────
dataset_processing:
  preprocess: true              # Run custom DataPreprocessor subclass
  split: true                   # Split extracted features into train/test
  split_mode: default           # "default" (2-way) | "threeway" (3-way)
  split_ratio: 0.2              # Fraction of data for test set
  feature_extraction: true      # Run feature extractor
  feature_extractor: FrameBuilder  # "FrameBuilder" | "Stat" | "PixNet"

# ─── Stage 2: Training ────────────────────────────────────────────────────────
training:
  model: FrameInceptionResNet   # Class name in ids/ (auto-discovered)
  model_name: CH_DoS_target     # Suffix for checkpoint filename
  epochs: 10
  train_dataset_dir: CH_DoS_train_frames  # Subdirectory under datasets/<name>/train/

# ─── Stage 3: Testing ─────────────────────────────────────────────────────────
testing:
  model: FrameInceptionResNet
  model_name: CH_DoS_target
  test_dataset_dir: CH_DoS_test_frames    # Subdirectory under datasets/<name>/test/

# ─── Stage 4: Adversarial perturbation ───────────────────────────────────────
adversarial_perturbation:
  adv_attack: GeneticAdvAttack  # Class name in attacks/attack_handler/ (auto-discovered)
  attack_type: DoS              # "DoS" | "Fuzzy" | "Spoof"

# ─── Stage 5: Robust training ─────────────────────────────────────────────────
adversarial defence:
  defense_method: AdversarialRetraining
  adv_samples: 350              # Max adversarial samples merged into retrain dataset
  adv_examples_path: ~          # Set automatically after Stage 4; or provide manually
```

### How `driver.py` Builds the Config Dict

```python
# driver.py (pseudocode)
import yaml

with open("src/config.yaml") as f:
    raw = yaml.safe_load(f)

cfg = {}
# Flatten nested keys — e.g., training.epochs → cfg["epochs"]
# Top-level keys → cfg["dataset_name"], cfg["file_name"]
# run_steps → cfg["dataset_processing"], cfg["training"], ...
# After Stage 4 completes, driver adds: cfg["adv_examples_path"] = <path to .npz>
```

Every pipeline function receives this same `cfg` dict. Functions pull the keys they need; unknown keys are ignored.

---

## 4. Pipeline Execution Flow

```
python driver.py
        │
        ▼
  load src/config.yaml
  build_config() → cfg dict
        │
        ▼
  run_pipeline(cfg)
        │
        ├─── [dataset_processing == true]
        │         │
        │         ├── preprocessing.preprocess(dataset_path)
        │         │      └─ Loads DataPreprocessor subclass from datasets/<name>/preprocess_dataset.py
        │         │         Calls instance.run(dataset_path)
        │         │
        │         ├── get_extractor(cfg["feature_extractor"], cfg)
        │         │      └─ Returns initialized extractor (runs extract() internally)
        │         │
        │         └── get_splitter(cfg["split_mode"], extractor, cfg)
        │                └─ Calls splitter.split() → saves train/test to disk
        │
        ├─── [training == true]
        │         └── train_model(modelName, modelPath, cfg)
        │                └─ get_ids(modelName) → model.train(train_dir, cfg) → model.save(path)
        │
        ├─── [testing == true]
        │         └── test_model(modelName, modelPath, cfg)
        │                └─ model.load(path) → model.test(cfg) → (preds, labels)
        │                   evaluation_metrics(preds, labels, cfg)
        │
        ├─── [adversarial_perturbation == true]
        │         └── get_attack(attackName, cfg)
        │                └─ attack.apply(cfg) → generates .npz → cfg["adv_examples_path"] updated
        │
        └─── [robust_training == true]
                  └── adversarial_retraining(modelPath, adv_path, cfg, limit)
                         ├─ get_retrainer(feature_extractor) → retrainer.make_dataset(...)
                         └─ retrain_model(modelPath, retrain_dir, cfg)
```

---

## 5. Stage 1 — Dataset Processing

### 5.1 Preprocessing (`src/preprocessing.py`)

**Entry function:** `preprocess(dataset_path)`

```
preprocess(dataset_path)
  │
  ├── Scan datasets/<name>/preprocess_dataset.py via importlib
  ├── Find class inheriting from DataPreprocessor using inspect
  └── Call instance.run(dataset_path)
```

**Abstract base** (`src/base_preprocessor.py`):

```python
class DataPreprocessor(ABC):
    def run(self, dataset_path):
        # 1. Moves all non-.py files → original_dataset/
        # 2. Calls self.preprocess_dataset(orig_path, modified_path)
        # 3. Output lands in modified_dataset/

    @abstractmethod
    def preprocess_dataset(self, orig_file_path, modified_file_path, **kwargs):
        # Implement in each dataset-specific file
        # Read from orig_file_path (directory)
        # Write cleaned CSV(s) to modified_file_path (directory)
```

**To add a new dataset:**

1. Create `datasets/MyDataset/preprocess_dataset.py`
2. Subclass `DataPreprocessor` and implement `preprocess_dataset()`
3. Set `dataset_name: MyDataset` in `config.yaml`

---

### 5.2 Feature Extraction (`features/feature_extractors/`)

**Entry function** (`src/get_extractor.py`): `get_extractor(name, cfg) -> FeatureExtractor`

Uses `features.feature_extractors.__all_classes__` (auto-discovered) with case-insensitive name matching.

**Abstract base** (`features/feature_extractors/base.py`):

```python
class FeatureExtractor:
    def __init__(self, cfg):
        self.dataset_name = cfg["dataset_name"]
        self.file_name    = cfg["file_name"]
        self.dataset_path = "datasets/{dataset_name}/modified_dataset/"
        self.csv_file_path = <full path to input CSV>
        self.features_path = "datasets/{dataset_name}/features/"
        self.extract()    # Called automatically on construction

    @abstractmethod
    def extract(self):
        # Read self.csv_file_path
        # Write outputs to self.features_path
```

#### FrameBuilder (`frame_builder.py`)

Converts raw CAN packets to **29×29 binary matrices**.

**Input CSV columns:** `timestamp, can_id, dlc, b0, b1, ..., b7, flag`

**Process:**
1. Group rows into windows of 29 consecutive packets (frames)
2. For each packet: extract 4-char hex CAN ID → convert to 16-bit binary string
3. Assemble 29 rows × 16 bits = 29×16 sub-frame (padded to 29×29)
4. Label frame as `1` (attack) if **any** packet in the window has `flag == 'T'`

**Outputs:**
```
datasets/<name>/features/Frames/<file_name>_frames.csv   # (num_frames × 841) pixel values
datasets/<name>/features/Frames/<file_name>_labels.csv   # (num_frames) frame_id, label
```

#### Stat (`stat_features.py`)

Extracts a **1-D statistical feature vector** per packet.

**Features:**
- `can_id` → decimal integer
- `dlc` → data length code integer
- `b0–b7` → combined as a single integer
- `iat` → inter-arrival time (timestamp[i] − timestamp[i−1])

**Preprocessing:** StandardScaler normalization; scaler saved as pickle for test-time re-use.

**Returns:** `(X, Y)` tuple (numpy arrays) rather than writing CSVs.

---

### 5.3 Data Splitting (`splitters/`)

**Entry function** (`src/get_splitter.py`): `get_splitter(mode, extractor, cfg)`

Uses `splitters.__all_classes__` (auto-discovered).

**Abstract base** (`splitters/base.py`):

```python
class BaseSplitter(ABC):
    @abstractmethod
    def split(self):
        # Partition data; save train/test to disk
```

#### Default (`default.py`) — 2-way split

For **FrameBuilder** data:
1. Load `features/Frames/<name>_frames.csv` and `<name>_labels.csv`
2. Reshape flat pixels → `(num_frames, 29, 29, 1)` numpy array
3. Split at index `int((1 - split_ratio) * num_frames)`
4. Save train and test portions as both `.csv` and `.npz`

```
datasets/<name>/train/<train_dataset_dir>/
    <file_name>_train_frames.csv
    <file_name>_train_labels.csv
    <file_name>_train_data.npz

datasets/<name>/test/<test_dataset_dir>/
    <file_name>_test_frames.csv
    <file_name>_test_labels.csv
    <file_name>_test_data.npz
```

#### ThreeWay (`threeway.py`) — 3-way split

Produces **Surrogate (1/3) | Target (1/3) | Test (1/3)** partitions. Used for black-box attack scenarios where the attacker trains on the surrogate model to attack the unseen target model.

---

### Extending Stage 1

#### Adding a new dataset preprocessor

1. Create `datasets/<MyDataset>/preprocess_dataset.py` — the file must live inside the dataset directory so `preprocessing.py` can find it via reflection.
2. Subclass `DataPreprocessor` from `src/base_preprocessor.py` and implement `preprocess_dataset()`:

```python
from src.base_preprocessor import DataPreprocessor

class MyDatasetPreprocessor(DataPreprocessor):
    def preprocess_dataset(self, orig_file_path, modified_file_path, **kwargs):
        # orig_file_path  → directory containing original raw files
        # modified_file_path → directory to write cleaned CSV(s)
        # Write one CSV per attack type with columns:
        #   timestamp, can_id, dlc, b0, b1, ..., b7, flag
        # flag: 'T' = attack, 'R' or '0' = normal
```

3. Set `dataset_name: MyDataset` in `config.yaml`. No other registration needed — `preprocessing.py` uses `importlib` + `inspect` to locate the subclass automatically.

---

#### Adding a new feature extractor

**Step 1 — Understand what the base class provides**

`features/feature_extractors/base.py` sets up every path you need. Calling `super().__init__(cfg)` makes these attributes available:

| Attribute | Value | Description |
|-----------|-------|-------------|
| `self.dir_path` | `cfg['dir_path']` | Absolute path to `src/` |
| `self.dataset_name` | `cfg['dataset_name']` | e.g. `"CarHackingDataset"` |
| `self.file_name` | `cfg['file_name']` | e.g. `"DoS_target.csv"` |
| `self.feature_extractor_name` | `cfg['feature_extractor']` | Class name string from config |
| `self.feature_extraction` | `cfg['feature_extraction']` | Boolean flag |
| `self.dataset_path` | `…/datasets/<name>/modified_dataset/` | Directory of cleaned CSV input (created) |
| `self.file_path` | `…/modified_dataset/<file>.csv` | Full path to the input CSV |
| `self.csv_file_name` | `.csv` normalised filename | Extension coerced from `.log`/`.txt`/`.csv` |
| `self.json_folder` | `…/datasets/<name>/json_files/` | Optional JSON output dir (created) |
| `self.json_file_path` | `…/json_files/<file>.json` | Full path for optional JSON |
| `self.features_path` | `…/datasets/<name>/features/` | Root output directory (created) |

All three output directories are created in the base constructor; you do not need to call `os.makedirs` on them.

**Step 2 — Write the extractor class**

Create `features/feature_extractors/my_extractor.py`. The existing extractors (`FrameBuilder`, `Stat`) do **not** override the abstract `extract()` method — they call their own processing method directly from `__init__`. All processing must finish before `__init__` returns, because `get_extractor()` passes the object straight to the splitter.

```python
# features/feature_extractors/my_extractor.py
import os
import csv
import numpy as np
import pandas as pd
from features.feature_extractors.base import FeatureExtractor


class MyExtractor(FeatureExtractor):
    def __init__(self, cfg):
        super().__init__(cfg)     # populates all self.* path attributes
        self._run()               # extract immediately

    def _run(self):
        output_dir = os.path.join(self.features_path, "MyFeatures")
        os.makedirs(output_dir, exist_ok=True)

        # ── Read the cleaned input CSV ─────────────────────────────────────
        # Columns: timestamp, can_id, dlc, b0-b7, flag
        df = pd.read_csv(self.file_path)

        # ── Compute features ───────────────────────────────────────────────
        X = ...  # numpy array, shape (N, num_features)
        Y = df['flag'].str.upper().isin(['T', '1', 'ATTACK', 'A']).astype(int).values

        # ── Write outputs under self.features_path ─────────────────────────
        prefix       = self.file_name[:-4]   # strip ".csv"
        features_csv = os.path.join(output_dir, prefix + "_features.csv")
        labels_csv   = os.path.join(output_dir, prefix + "_labels.csv")

        np.savetxt(features_csv, X, delimiter=",")

        with open(labels_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "label"])
            for i, lbl in enumerate(Y):
                writer.writerow([i, lbl])

        print(f"  Extractor      : MyExtractor")
        print(f"  Samples        : {len(Y)}  "
              f"(Benign: {(Y==0).sum()}, Attack: {(Y==1).sum()})")

    def extract(self):
        pass   # abstract method satisfied; work is done in __init__ via _run()
```

**Output layout the splitter will expect:**

```
datasets/<name>/features/MyFeatures/
    <file_name>_features.csv   # (N × F) — one sample per row, no header
    <file_name>_labels.csv     # header: sample_id, label
```

No registration is needed — `features/feature_extractors/__init__.py` auto-discovers every class defined in that directory.

**Step 3 — Add a split branch in the splitter**

The splitter reads `cfg['feature_extractor']` (the string name) to decide which files to load. Open `splitters/default.py` and add a branch inside `Default.split()`:

```python
# splitters/default.py — inside Default.split()
if cfg['feature_extractor'] == "MyExtractor":
    prefix       = file_name[:-4]
    features_csv = os.path.join(self.input_dir, "features", "MyFeatures",
                                prefix + "_features.csv")
    labels_csv   = os.path.join(self.input_dir, "features", "MyFeatures",
                                prefix + "_labels.csv")

    X = np.loadtxt(features_csv, delimiter=",")
    with open(labels_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)                               # skip header
        Y = np.array([int(row[1]) for row in reader])

    split_index = int((1 - self.split_ratio) * len(Y))
    x_train, y_train = X[:split_index], Y[:split_index]
    x_test,  y_test  = X[split_index:],  Y[split_index:]

    train_dir = os.path.join(self.input_dir, "train", cfg['train_dataset_dir'])
    test_dir  = os.path.join(self.input_dir, "test",  cfg['test_dataset_dir'])
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    # Save CSVs
    np.savetxt(os.path.join(train_dir, prefix + "_train_features.csv"), x_train, delimiter=",")
    np.savetxt(os.path.join(test_dir,  prefix + "_test_features.csv"),  x_test,  delimiter=",")

    for split_Y, split_dir, tag in [(y_train, train_dir, "train"), (y_test, test_dir, "test")]:
        lbl_path = os.path.join(split_dir, f"{prefix}_{tag}_labels.csv")
        with open(lbl_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "label"])
            for i, lbl in enumerate(split_Y):
                writer.writerow([i, lbl])

    # Save .npz (required for genetic attack compatibility)
    np.savez(os.path.join(train_dir, prefix + "_train_data.npz"),
             x_train=x_train, y_train=y_train)
    np.savez(os.path.join(test_dir,  prefix + "_test_data.npz"),
             x_test=x_test,  y_test=y_test)

    print(f"  Split          : Train={len(y_train)}, Test={len(y_test)}")
```

**Step 4 — Register a retrainer (Stage 5 only)**

If you plan to use adversarial retraining with this extractor, implement a `BaseDefense` subclass (see [Stage 5](#9-stage-5--adversarial-retraining)) and register it in `src/retraining.py`:

```python
_RETRAINER_REGISTRY = {
    "FrameBuilder": FrameBuilderRetrainer,
    "MyExtractor":  MyRetrainer,          # ← add this line
}
```

**Step 5 — Config**

```yaml
dataset_processing:
  feature_extraction: true
  feature_extractor: MyExtractor   # must match class name (case-insensitive prefix match)
  split: true
  split_mode: default
```

Name matching rule in `get_extractor.py`:
```python
if extractor_class.__name__.lower() in feature_extractor.lower():
```
Use the exact class name to avoid accidental prefix collisions.

**Checklist**

| Task | File | Required |
|------|------|----------|
| Extractor class | `features/feature_extractors/my_extractor.py` (new) | Yes |
| Split branch | `splitters/default.py` inside `Default.split()` | Yes |
| 3-way split branch | `splitters/threeway.py` inside `ThreeWay.split()` | Only if using ThreeWay |
| Retrainer class | `defense/retrainers/MyRetrainer.py` (new) | Only for Stage 5 |
| Retrainer registry | `src/retraining.py` — `_RETRAINER_REGISTRY` | Only for Stage 5 |
| Config | `src/config.yaml` | Yes |

---

#### Adding a new data splitter

1. Create `splitters/my_splitter.py`. The class name must contain the string that will be set in `split_mode` (case-insensitive match).
2. Inherit from `BaseSplitter` and implement `split()`:

```python
from splitters.base import BaseSplitter

class MySplitter(BaseSplitter):
    def __init__(self, input_dir, feature_extractor, cfg):
        super().__init__(input_dir)
        self.feature_extractor = feature_extractor  # string name from cfg
        self.cfg = cfg

    def split(self):
        # Partition data from self.input_dir
        # Write train/test to datasets/<name>/train/ and datasets/<name>/test/
        # Save .npz files for attack compatibility
        ...
```

3. Set `split_mode: MySplitter` in `config.yaml`. Auto-discovered — no registration needed.

---

## 6. Stage 2 — Training

**Entry function** (`src/train.py`): `train_model(modelName, modelPath, cfg)`

```python
def train_model(modelName, modelPath, cfg):
    model = get_ids(modelName)          # Factory lookup
    train_dir = f"datasets/{cfg['dataset_name']}/train/{cfg['train_dataset_dir']}"
    model.train(train_dir, cfg=cfg)
    model.save(modelPath)               # Saves as modelPath.h5
```

**Model path convention:** `models/<ModelClass>_<model_name>.h5`

### IDS Abstract Interface (`ids/base.py`)

Every IDS class must implement:

```python
class IDS(ABC):
    @abstractmethod
    def train(self, X_train, Y_train, cfg, **kwargs):
        """Train the model. X_train may be a directory path or array."""

    @abstractmethod
    def test(self, X_test, Y_test, cfg, **kwargs) -> tuple:
        """Evaluate and return (predictions, true_labels)."""

    @abstractmethod
    def predict(self, X_test, **kwargs):
        """Return predictions for X_test."""

    @abstractmethod
    def save(self, path):
        """Persist model to disk."""

    @abstractmethod
    def load(self, path):
        """Restore model from disk."""
```

---

### Extending Stage 2

#### Adding a new IDS model

1. Create `ids/MyModel.py`. The filename becomes the discoverable module; the class name is what you set in `config.yaml`.
2. Inherit from `IDS` (`ids/base.py`) and implement all five abstract methods:

```python
# ids/MyModel.py
from ids.base import IDS
import numpy as np


class MyModel(IDS):

    def train(self, train_dataset_dir, cfg, **kwargs):
        # train_dataset_dir: absolute path to the train split folder
        # cfg keys available: epochs, model_name, dataset_name, file_name, etc.
        # Load your feature files from train_dataset_dir, fit the model.
        # For FrameBuilder data, load:
        #   <prefix>_train_frames.csv  and  <prefix>_train_labels.csv
        # For other extractors, load whatever format that extractor wrote.
        ...

    def test(self, cfg, **kwargs) -> tuple:
        # Load test data from datasets/<name>/test/<test_dataset_dir>/
        # Return (predictions, true_labels) as 1-D numpy arrays.
        # predictions: 0 = normal, 1 = attack (or multi-class integers)
        preds  = ...
        labels = ...
        return preds, labels

    def predict(self, X, **kwargs):
        # Return raw predictions for a batch X.
        # Used by attack and retraining modules — must accept a numpy array.
        return self.model.predict(X)

    def save(self, path):
        # Persist model to disk.
        # Use path as-is; driver.py appends .h5 for Keras or .pkl for sklearn.
        self.model.save(path)

    def load(self, path):
        # Restore model from disk and prepare it for inference.
        # Recompile / reload scaler if needed.
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)
```

3. No registration needed — `ids/__init__.py` auto-discovers every class in `ids/`.
4. Set in `config.yaml`:
```yaml
training:
  model: MyModel
testing:
  model: MyModel
```

**Conventions to follow:**

- `train()` receives the directory path, not pre-loaded arrays. Load data inside the method using the same CSV/npz layout that your feature extractor and splitter produced.
- `test()` must return `(preds, labels)` — `evaluate.py` depends on this exact signature.
- `predict()` must accept a numpy array and return class indices (not probabilities), because the genetic attack uses it as the fitness oracle.
- `save()` / `load()` paths are constructed by `driver.py` as `models/<ClassName>_<model_name>.h5`.

---

## 7. Stage 3 — Testing & Evaluation

**Entry function** (`src/test.py`): `test_model(modelName, modelPath, cfg, adv_attack=None)`

```python
def test_model(modelName, modelPath, cfg, adv_attack=None):
    model = get_ids(modelName)
    model.load(modelPath)
    preds, labels = model.test(cfg=cfg)
    evaluation_metrics(preds, labels, cfg)
```

### Evaluation (`src/evaluate.py`)

**Function:** `evaluation_metrics(all_preds, all_labels, cfg)`

Computes and saves:
- Confusion matrix (PNG image)
- Accuracy, Precision, Recall, F1-score
- For adversarial context: TNR, MDR (Missed Detection Rate), ASR (Attack Success Rate)

**Output directory:**
- Normal test: `datasets/<name>/Results/<model_name>/`
- Attack evaluation: `datasets/<name>/Results/<attack_type>/`

---

### Extending Stage 3

Stage 3 does not have pluggable components — it calls `model.test()` and passes the results to `evaluation_metrics()`. Extension here means modifying one of those two functions:

**To change what metrics are computed or reported**, edit `src/evaluate.py: evaluation_metrics()`. The function receives `(all_preds, all_labels, cfg)` and has full access to the config dict. Add any new metric computation there and append to the saved results file.

**To change how a specific model runs inference**, override `test()` in your IDS class (see Extending Stage 2). The only contract is that it returns `(predictions, true_labels)` as 1-D numpy arrays.

---

## 8. Stage 4 — Adversarial Attacks

**Entry function** (`src/get_attack.py`): `get_attack(attack_name, cfg) -> str (path to .npz)`

Uses `attacks.attack_handler.__all_classes__` (auto-discovered). Calls `attack.apply(cfg)`.

### Attack Class Hierarchy (`attacks/attack_handler/base.py`)

```python
class Attack(ABC):
    @abstractmethod
    def apply(self, **kwargs):
        """Core attack logic. Returns path to saved adversarial examples."""

class EvasionAttack(Attack):
    @abstractmethod
    def apply(self, frames, labels) -> np.ndarray:
        """Return perturbed frames."""

class GeneticAttack(Attack):
    def __init__(self, model_path, file_path, pop_size, max_gens, mutation_rate):
        ...

    @abstractmethod
    def mutate(self, frame) -> np.ndarray:
        """Apply a mutation operator to one frame."""

    @abstractmethod
    def crossover(self, parent1, parent2) -> np.ndarray:
        """Produce offspring from two parent frames."""

    @abstractmethod
    def generate_adversarial_attack(self) -> tuple:
        """Run the full genetic algorithm. Return (adv_frames, labels, orig_frames, gens)."""
```

### GeneticAdvAttack (`attacks/attack_handler/genetic_attack.py`)

Routes to concrete attack based on `cfg["attack_mode"]` (lowercased):

```python
ATTACK_REGISTRY = {
    "dos":   AdversarialDosAttack,
    "fuzzy": AdversarialFuzzyAttack,
    "spoof": AdversarialSpoofAttack,
}
```

Loads:
- Model from `models/<ModelClass>_<model_name>.h5`
- Test data from `datasets/<name>/test/<test_dataset_dir>/<file_name>_test_data.npz`

### AdversarialDosAttack (`attacks/Genetic_algorithm/Adversarial_DoS.py`)

**Strategy:** Only modifies **dummy (all-zero) rows** — rows that represent absent CAN IDs in the frame — preserving the semantic validity of attack traffic.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 100 | Number of candidate frames per generation |
| `max_generations` | 20 | GA iteration limit |
| `mutation_rate` | 0.4 | Probability of mutating a bit |
| `dummy_row_threshold` | 10 | Min zero-rows required for a frame to be attackable |
| `max_frames` | 7000 | Total frames to process (30% benign, 70% attack) |

**GA loop per frame:**
```
for each attack frame with > threshold dummy rows:
    population = [mutate(frame) × pop_size]
    for gen in range(max_generations):
        scores = model.predict(population)         # Confidence for attack class
        if min(scores) < 0.5: break               # Evasion achieved
        parents = top-k by lowest score
        offspring = [crossover(p1, p2) for p1, p2 in pairs]
        population = mutate(offspring)
    best = argmin(scores)
    return population[best]
```

**`apply(cfg)` flow:**
1. Run / load from cache `adversarial_DoS_attack.npz`
2. Save to `datasets/<name>/adversarial_DoS_attack.npz`
3. Update `cfg["adv_examples_path"]`
4. Evaluate model on adversarial examples → compute ASR, FNR, ER
5. Save confusion matrix and sample comparison plots to `Results/attack_results/DoS_<timestamp>/`

### AdversarialFuzzyAttack

**Strategy:** Modifies **any bits** in the frame (not restricted to dummy rows).

| Difference vs DoS | Detail |
|-------------------|--------|
| `mutate()` | Flips 3 random bits at arbitrary (row, col) positions |
| `crossover()` | Pixel-by-pixel 50% inheritance from either parent |
| No dummy row filtering | All attack frames are candidates |

### AdversarialSpoofAttack

Similar structure; implements spoofing-specific mutation and crossover strategies.

---

### Extending Stage 4

#### Adding a new standalone attack

1. Create `attacks/attack_handler/my_attack.py`. The class name is what you set in `adv_attack` in `config.yaml`.
2. Inherit from `Attack` (for a generic attack) or `EvasionAttack` (for gradient/perturbation-based) from `attacks/attack_handler/base.py`:

```python
# attacks/attack_handler/my_attack.py
from attacks.attack_handler.base import Attack


class MyAttack(Attack):
    def apply(self, cfg) -> str:
        # cfg contains: model_name, dataset_name, file_name,
        #               test_dataset_dir, attack_type, etc.
        #
        # 1. Load the target model from models/<ModelClass>_<model_name>.h5
        # 2. Load test data from datasets/<name>/test/<test_dataset_dir>/
        # 3. Generate adversarial examples
        # 4. Save to datasets/<name>/my_attack_examples.npz with keys:
        #       final_test = adversarial frames
        #       y_test     = labels
        #       x_test     = original frames (for comparison)
        # 5. Run evaluation_metrics() on the adversarial examples
        # 6. Return the path to the saved .npz file (driver passes it to Stage 5)
        adv_path = ...
        return adv_path
```

3. No registration needed — `attacks/attack_handler/__init__.py` auto-discovers the class.
4. Set in `config.yaml`:
```yaml
adversarial_perturbation:
  adv_attack: MyAttack
```

**`apply()` must return the `.npz` path** — `driver.py` stores it in `cfg["adv_examples_path"]` and passes it to Stage 5.

---

#### Adding a new genetic attack type

If you want to add a new attack type routed through the existing `GeneticAdvAttack` dispatcher:

1. Create `attacks/Genetic_algorithm/Adversarial_MyType.py`:

```python
from attacks.attack_handler.base import GeneticAttack


class AdversarialMyTypeAttack(GeneticAttack):

    def mutate(self, frame) -> np.ndarray:
        # Apply domain-valid mutation to one 29×29×1 frame.
        # Only flip bits that are semantically meaningful for your attack type.
        ...

    def crossover(self, parent1, parent2) -> np.ndarray:
        # Combine two candidate frames to produce offspring.
        ...

    def generate_adversarial_attack(self) -> tuple:
        # Run the full GA loop.
        # Return (adversarial_frames, labels, original_frames, generations_per_frame)
        ...

    def apply(self, cfg) -> str:
        # Orchestrate: generate (or load cached), save .npz, evaluate, return path.
        ...
```

2. Register the type string in `attacks/attack_handler/genetic_attack.py`:

```python
ATTACK_REGISTRY = {
    "dos":    AdversarialDosAttack,
    "fuzzy":  AdversarialFuzzyAttack,
    "spoof":  AdversarialSpoofAttack,
    "mytype": AdversarialMyTypeAttack,   # ← add this
}
```

3. Set in `config.yaml`:
```yaml
adversarial_perturbation:
  adv_attack: GeneticAdvAttack
  attack_type: mytype
```

---

## 9. Stage 5 — Adversarial Retraining

**Entry function** (`src/retraining.py`): `adversarial_retraining(model_path, adv_path, cfg, limit=800)`

```python
def adversarial_retraining(model_path, adv_path, cfg, limit=800):
    retrainer = get_retrainer(cfg["feature_extractor"])
    retrain_dir = retrainer.make_dataset(model_path, adv_path, cfg, limit)
    retrain_model(model_path, retrain_dir, cfg)
```


### BaseDefense (`defense/base.py`)

```python
class BaseDefense(ABC):
    @abstractmethod
    def make_dataset(self, model_path, adv_npz_path, cfg, limit) -> str:
        """Build and save retrain dataset. Return path to directory."""
```

### FrameBuilderRetrainer (`defense/retrainers/FrameBuilderRetrainer.py`)

**`make_dataset(model_path, adv_npz_path, cfg, limit=5000)`:**

1. Load original training frames & labels from `train/<train_dataset_dir>/`
2. Load adversarial examples from `.npz` (`final_test`, `y_test` keys)
3. Evaluate current model on adversarial examples
4. Filter adversarial set: keep **only attack frames that were misclassified as normal**
   ```python
   successful_idx = np.where((labels == 1) & (preds == 0))[0]
   ```
5. If `len(successful_idx) > limit`, randomly subsample to `limit`
6. Concatenate: `X = [original_frames, adversarial_frames]`, `Y = [0s, 1s]`
7. Save merged dataset to `datasets/<name>/train/retrain_dataset/`
8. Return `retrain_dataset/` path

**`retrain_model(modelPath, retrain_dir, cfg)`** (in `src/train.py`):
- Loads model, calls `model.train(retrain_dir, cfg=cfg)`
- Saves retrained model to `<base>_retrained.h5`

---

### Extending Stage 5

#### Adding a new retrainer

A retrainer is needed whenever you introduce a new feature extractor, because the merged dataset must be built in the format that extractor produces (FrameBuilder uses 29×29 frame CSVs; a statistical extractor would use a flat feature CSV, etc.).

1. Create `defense/retrainers/MyRetrainer.py`:

```python
# defense/retrainers/MyRetrainer.py
from defense.base import BaseDefense
import numpy as np
import os


class MyRetrainer(BaseDefense):

    def make_dataset(self, model_path, adv_npz_path, cfg, limit=5000) -> str:
        """
        Build a merged training dataset from the original training data and
        the adversarial examples generated in Stage 4.

        Parameters
        ----------
        model_path    : str  — path to the current .h5 model checkpoint
        adv_npz_path  : str  — path to the .npz produced by the attack (Stage 4)
                               Keys: final_test (adv frames), y_test (labels),
                                     x_test (originals)
        cfg           : dict — full pipeline config
        limit         : int  — max adversarial samples to include

        Returns
        -------
        str — path to the retrain dataset directory (passed to retrain_model())
        """

        # ── 1. Load original training data ────────────────────────────────
        train_dir  = os.path.join(cfg['dir_path'], '..', 'datasets',
                                  cfg['dataset_name'], 'train',
                                  cfg['train_dataset_dir'])
        prefix     = cfg['file_name'][:-4]
        x_orig     = np.loadtxt(os.path.join(train_dir, prefix + "_train_features.csv"),
                                delimiter=",")
        y_orig     = ...   # load from labels CSV

        # ── 2. Load adversarial examples ───────────────────────────────────
        data       = np.load(adv_npz_path)
        x_adv      = data['final_test']
        y_adv      = data['y_test']

        # ── 3. Load model and filter to successful adversarial examples ────
        # (examples where an attack frame was misclassified as normal)
        import tensorflow as tf
        model      = tf.keras.models.load_model(model_path)
        preds      = np.argmax(model.predict(x_adv), axis=1)
        success    = np.where((y_adv == 1) & (preds == 0))[0]

        if len(success) > limit:
            success = np.random.choice(success, limit, replace=False)

        x_adv_filtered = x_adv[success]
        y_adv_filtered = y_adv[success]

        # ── 4. Merge and save ──────────────────────────────────────────────
        X_merged   = np.concatenate([x_orig, x_adv_filtered], axis=0)
        Y_merged   = np.concatenate([y_orig, y_adv_filtered], axis=0)

        retrain_dir = os.path.join(cfg['dir_path'], '..', 'datasets',
                                   cfg['dataset_name'], 'train', 'retrain_dataset')
        os.makedirs(retrain_dir, exist_ok=True)

        np.savetxt(os.path.join(retrain_dir, prefix + "_train_features.csv"),
                   X_merged, delimiter=",")
        # save labels CSV ...

        return retrain_dir
```

2. Register the retrainer in `src/retraining.py` under the matching feature extractor name:

```python
# src/retraining.py
_RETRAINER_REGISTRY = {
    "FrameBuilder": FrameBuilderRetrainer,
    "MyExtractor":  MyRetrainer,          # ← add this line
}
```

The key must exactly match the value of `cfg['feature_extractor']` (i.e., the class name you set in `feature_extractor:` in `config.yaml`).

3. No config change is needed for the retrainer itself — Stage 5 looks up the registry using the feature extractor name that is already in `cfg`.

---

## 10. IDS Model Implementations

All models live in `ids/` and inherit `IDS`. They are auto-discovered by `ids/__init__.py`.

### FrameInceptionResNet (`ids/FrameInceptionResNet.py`) — Primary Model

**Architecture:** Custom Inception-ResNet-V1 adapted for 29×29×1 binary input.

| Stage | Block | Input → Output |
|-------|-------|----------------|
| 1 | Stem | 29×29×1 → 13×13×128 |
| 2 | Inception-ResNet-A | 13×13×128 → 13×13×128 |
| 3 | Reduction-A | 13×13×128 → 6×6×448 |
| 4 | Inception-ResNet-B | 6×6×448 → 6×6×448 |
| 5 | Reduction-B | 6×6×448 → 2×2×896 |
| 6 | GlobalAvgPool | 2×2×896 → 896 |
| 7 | Dense + Softmax | 896 → 2 |

**Key methods:**

```python
def train(self, train_dataset_dir, cfg):
    # Loads frames CSV + labels CSV from train_dataset_dir
    # Reshapes: (num_frames, 29, 29, 1)
    # Trains for cfg["epochs"] epochs with BatchLossHistory callback
    # Saves batch-level loss history

def test(self, cfg):
    # Loads from datasets/<name>/test/<test_dataset_dir>/
    # Returns (preds, labels)

def load(self, path):
    # Loads .h5, recompiles with Adam optimizer

def load_frames_and_labels(self, frames_csv, labels_csv, h, w):
    # Returns X shaped (N, h, w, 1), Y shaped (N,)
```

**Custom callback:**

```python
class BatchLossHistory(Callback):
    def on_train_batch_end(self, batch, logs):
        self.history.append((self.current_iteration, logs["loss"]))
```

### Shannon / CorrectedShannon (`ids/shannon.py`)

Entropy-based anomaly detector. No neural network.

```python
def train(self, normal_data, cfg):
    # Compute per-window Shannon entropy of normal traffic
    # Learn mean_entropy and std_entropy

def test(self, test_data, cfg):
    # Flag window as attack if entropy outside mean ± k_factor * std
```

Parameters: `time_window=0.032768`, `k_factor=5.25`.

### RandomForest / DecisionTree

Thin wrappers around `sklearn.ensemble.RandomForestClassifier` and `DecisionTreeClassifier`. Use `Stat` feature extractor output directly.

### MLP (`ids/MLP.py`)

PyTorch Sequential: `Input(4) → Dense(128, ReLU) → Dense(128, ReLU) → Dense(4, Softmax)`. Uses early stopping on validation loss.

---

## 11. Utility Modules

### `src/utilities.py`

**Hex / Binary conversion:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `hex_to_bits` | `(hex_value, num_bits) -> str` | Hex string → zero-padded binary string |
| `bits_to_hex` | `(binary_str) -> str` | Binary string → hex |
| `int_to_bin` | `(int_num) -> str` | Integer → binary string |
| `pad` | `(value, length) -> str` | Left-pad string with zeros |
| `hex_to_dec` | lambda | Hex string → int |

**DataFrame helpers:**

| Function | Description |
|----------|-------------|
| `transform_data(df)` | Convert 'ID' & 'Payload' hex columns to decimal |
| `shift_columns(df)` | Right-align data columns based on DLC value |
| `df_to_csv(df, path)` | Save DataFrame to CSV |

**Sequence / window helpers:**

| Function | Description |
|----------|-------------|
| `sequencify_data(X, y, seq_size)` | Convert flat samples to overlapping windows |
| `sequencify(dataset, target, start, end, window)` | LSTM-style windowing |
| `balance_data(X_seq, y_seq)` | Downsample majority class |

### `src/dataset_analysis.py`

Standalone statistical analysis. Called by `run_analysis.py`.

| Function | Plot produced |
|----------|---------------|
| `plot_can_id_distribution()` | Bar chart — Top 20 CAN IDs by frequency |
| `plot_message_rate()` | Line chart — Messages per time bin |
| `plot_interarrival()` | Histogram — Inter-arrival time distribution |
| `plot_payload_histograms()` | Per-byte value distribution |
| `plot_canid_vs_time()` | Scatter — CAN ID over time |
| `plot_payload_entropy()` | Entropy distribution across payloads |
| `plot_byte_correlation()` | Heatmap — Byte-to-byte correlation |
| `plot_canid_periodicity()` | Periodicity analysis per CAN ID |
| `plot_distinct_ids()` | Cumulative unique IDs over time |
| `plot_attack_distribution()` | Pie/bar — Benign vs Attack class balance |

### `src/common_imports.py`

Centralized import hub. Import from here rather than directly from libraries:

```python
from src.common_imports import os, sys, np, pd, plt
from src.common_imports import accuracy_score, confusion_matrix, f1_score
from src.common_imports import tf, keras                   # TF_AVAILABLE checked
from src.common_imports import torch, nn, optim, DataLoader  # TORCH_AVAILABLE checked
```

---

## 12. Dynamic Class Loading Pattern

All major component directories (`ids/`, `features/feature_extractors/`, `splitters/`, `attacks/attack_handler/`) use the same `__init__.py` auto-discovery pattern:

```python
# <package>/__init__.py
import os, importlib, inspect

__all__ = []
__all_classes__ = []

package_name = __name__
current_dir  = os.path.dirname(__file__)

for filename in os.listdir(current_dir):
    if filename.endswith('.py') and not filename.startswith('_'):
        module_name = filename[:-3]
        module = importlib.import_module(f'.{module_name}', package_name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == f'{package_name}.{module_name}':
                globals()[name] = obj
                __all__.append(name)
                __all_classes__.append(obj)
```

**Effect:** Any class defined in a new `.py` file dropped into these directories is automatically available. Factory functions search `__all_classes__` by name (case-insensitive), so no registration step is needed.

---

## 13. Data Formats

### Raw CSV Input

```
timestamp,    can_id, dlc, b0,   b1,   b2,   b3,   b4,   b5,   b6,   b7,   flag
0.000000000,  0264,   8,   00,   00,   00,   00,   00,   00,   00,   00,   R
0.001024000,  0268,   8,   00,   00,   00,   00,   00,   00,   00,   00,   T
```

- `can_id`: 4-character hex (no `0x` prefix typical, but `0x` handled)
- `flag`: `T` = attack, `R` or `0` or `F` = normal

### FrameBuilder CSV Output

```
# features/Frames/<name>_frames.csv  — shape (num_frames, 841) — flattened 29×29
frame_id, p0, p1, ..., p840
0,        0,  1,  ..., 0

# features/Frames/<name>_labels.csv
frame_id, label
0,        1
```

### Train/Test NPZ

```python
np.savez(path,
    frames = X,   # (N, 29, 29, 1)   uint8
    labels = Y,   # (N,)             int (0 or 1)
)
```

### Adversarial Examples NPZ

```python
np.savez(path,
    final_test = adv_frames,   # (N, 29, 29, 1)  perturbed
    y_test     = labels,       # (N,)            1=attack, 0=benign
    x_test     = orig_frames,  # (N, 29, 29, 1)  originals for comparison
)
```

---

## 14. Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
Pillow>=10.0.0
PyYAML>=5.4.0
joblib>=1.3.0
tensorflow>=2.12.0          # Required for FrameInceptionResNet and GeneticAttack
```

Install:
```bash
pip install -r requirements.txt
```

---

## 15. Quick Reference Table

| Component | File(s) | Base Class | Config Key | Auto-discovered |
|-----------|---------|------------|------------|-----------------|
| Pipeline entry | `driver.py` | — | — | — |
| Config | `src/config.yaml` | — | — | — |
| Preprocessor | `datasets/<name>/preprocess_dataset.py` | `DataPreprocessor` | `dataset_name` | No (path-based) |
| Feature extractor | `features/feature_extractors/*.py` | `FeatureExtractor` | `feature_extractor` | Yes |
| Data splitter | `splitters/*.py` | `BaseSplitter` | `split_mode` | Yes |
| IDS model | `ids/*.py` | `IDS` | `training.model` | Yes |
| Training | `src/train.py` | — | — | — |
| Testing | `src/test.py` | — | — | — |
| Evaluation | `src/evaluate.py` | — | — | — |
| Attack | `attacks/attack_handler/*.py` | `Attack` | `adv_attack` | Yes |
| Genetic attack impl | `attacks/Genetic_algorithm/*.py` | `GeneticAttack` | `attack_type` | Via registry |
| Defense/Retrainer | `defense/retrainers/*.py` | `BaseDefense` | (feature_extractor key) | Via registry |
| Utilities | `src/utilities.py` | — | — | — |
| Analysis | `src/dataset_analysis.py` | — | — | — |
| Imports hub | `src/common_imports.py` | — | — | — |
