# CAN Rakshak

An Intrusion Detection System (IDS) framework for CAN bus security. CAN Rakshak implements a complete adversarial research pipeline: dataset preprocessing → feature extraction → model training/testing → adversarial attack generation → adversarial retraining.

---

## Pipeline Overview

The pipeline runs in up to 4 stages, each independently toggleable via `src/config.yaml`:

```
Stage 1: Dataset Processing     — preprocess raw logs, extract features, split data
Stage 2: Testing & Evaluation   — train or test an IDS model
Stage 3: Adversarial Perturbation — generate adversarial examples using attack methods
Stage 4: Robust Training        — retrain model on clean + adversarial samples
```

---

## Project Structure

```
CANShield/
├── driver.py                   # Main entry point
├── requirements.txt
├── src/
│   ├── config.yaml             # Primary configuration
│   ├── preprocessing.py        # Loads dataset-specific preprocessor
│   ├── get_extractor.py        # Feature extractor factory
│   ├── get_splitter.py         # Data splitter factory
│   ├── get_ids.py              # IDS model factory
│   ├── get_attack.py           # Attack factory
│   ├── train.py                # Model training
│   ├── test.py                 # Model testing
│   ├── evaluate.py             # Metrics (accuracy, precision, recall, F1, ASR)
│   ├── retraining.py           # Adversarial retraining
│   ├── utilities.py            # Hex/binary conversion, balancing, normalization
│   ├── base_preprocessor.py    # Abstract base for dataset preprocessors
│   ├── common_imports.py       # Shared imports hub (TF, PyTorch, sklearn, stdlib)
│   ├── postprocessing.py       # Post-processing stub (to be implemented)
│   ├── dataset_analysis.py     # Statistical analysis of CAN data
│   └── run_analysis.py         # Analysis runner
├── ids/                        # IDS model implementations
├── features/
│   ├── feature_extractors/     # Feature extraction implementations
│   └── image/                  # Traffic image encoding/decoding utilities
├── splitters/                  # Data splitting strategies
├── attacks/                    # Adversarial attack implementations
├── defense/                    # Defense mechanisms (adversarial retraining)
├── datasets/                   # Dataset directories
└── models/                     # Saved model checkpoints
```

---

## Configuration

All pipeline settings are controlled through `src/config.yaml`:

```yaml
run_steps:
  dataset_processing: false       # Stage 1
  testing_and_evaluation: false   # Stage 2
  adversarial_perturbation: true  # Stage 3
  robust_training: true           # Stage 4

dataset_name: CarHackingDataset
file_name: DoS_target.csv

dataset_processing:
  preprocess: false
  split: false
  split_mode: default             # default | threeway
  split_ratio: 0.2
  feature_extraction: false
  feature_extractor: FrameBuilder # FrameBuilder | Stat  (PixNet: upcoming, pending publication)

testing_and_evaluation:
  model: FrameInceptionResNet
  model_name: CH_DoS_retrained
  mode: test                      # train | test
  epochs: 10
  train_dataset_dir: CH_DoS_train_frames
  test_dataset_dir: CH_DoS_test_frames

adversarial_perturbation:
  adv_attack: GeneticAdvAttack    # GeneticAdvAttack | FGSM
  attack_type: DoS                # DoS | Fuzzy | Spoof

robust_training:
  adv_retraining: true
  adv_samples: 2000
  adv_examples_path: '<path_to_.npz>'
```

---

## Supported Components

### IDS Models

| Model | Description |
|---|---|
| `FrameInceptionResNet` | InceptionResNetV1 (Keras) for 29×29 binary frames |
| `Resnet` | InceptionResNetV1 (PyTorch) for image-based input |
| `ResNet50` | ResNet-50 |
| `Densenet121` | DenseNet-121 |
| `Densenet161` | DenseNet-161 (surrogate model) |
| `ConvNeXtBase` | ConvNeXt Base |
| `RandomForest` | Scikit-learn Random Forest |
| `DecisionTree` | Decision Tree |
| `MLP` | PyTorch Multi-Layer Perceptron |
| `Shannon` | Shannon entropy-based detector |

### Feature Extractors

| Extractor | Description |
|---|---|
| `FrameBuilder` | Converts sequential CAN packets into 29×29 binary frames |
| `PixNet` | *(Novel work in progress — will be released upon publication)* |
| `Stat` | Statistical features from CAN traffic |

### Adversarial Attacks

| Attack | Type | Description |
|---|---|---|
| `GeneticAdvAttack` | Genetic | Genetic algorithm attack for DoS, Fuzzy, and Spoof traffic |
| `FGSM` | Evasion | Fast Gradient Sign Method with constrained perturbations |

### Datasets

| Dataset | Description |
|---|---|
| `CarHackingDataset` | Car hacking benchmark dataset |
| `CANIntrusionDataset` | Open CAN intrusion dataset |
| `CARLA` | CARLA simulation dataset |
| `MIRGU` | MIRGU vehicular network dataset |

### Data Splitters

| Splitter | Description |
|---|---|
| `default` | Standard 2-way train/test split |
| `threeway` | 3-way split (surrogate / target / test) for black-box attacks |

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** PyTorch ≥ 2.0, TorchVision ≥ 0.15, NumPy ≥ 1.24, Pandas ≥ 2.0, Scikit-learn ≥ 1.3, Matplotlib ≥ 3.7, Pillow ≥ 10.0, PyYAML ≥ 5.4, Joblib ≥ 1.3

> Note: Genetic and FGSM attacks also require TensorFlow/Keras.

---

## Usage

1. Place your dataset files in `datasets/<dataset_name>/`
2. Edit `src/config.yaml` to configure the run
3. Run the pipeline:

```bash
python driver.py
```

The pipeline will execute only the stages with their `run_steps` flag set to `true`.

---

## Dataset Directory Layout

Each dataset folder follows this structure:

```
datasets/<DatasetName>/
├── original_dataset/       # Raw input files (moved here by preprocessor)
├── modified_dataset/       # Preprocessed CSV files
├── features/
│   ├── Frames/             # 29×29 binary frame files
│   └── Images/             # Rendered images
├── train/                  # Training splits
├── test/                   # Test splits
└── Results/                # Evaluation outputs (metrics, confusion matrices)
```

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- True Negative Rate (TNR)
- Missed Detection Rate (MDR)
- Attack Success Rate (ASR) — for adversarial evaluation

---

## Extending the Framework

All core components use abstract base classes. To add a new component:

| Component | Base Class | Location |
|---|---|---|
| IDS Model | `IDS` | `ids/base.py` |
| Feature Extractor | `FeatureExtractor` | `features/feature_extractors/base.py` |
| Data Splitter | `BaseSplitter` | `splitters/base.py` |
| Dataset Preprocessor | `DataPreprocessor` | `src/base_preprocessor.py` |
| Attack | `Attack` / `EvasionAttack` / `GeneticAttack` | `attacks/attack_handler/base.py` |
| Defense / Retrainer | `BaseDefense` | `defense/base.py` |
