# CAN-Rakshak — Detailed Usage & Documentation

This document provides comprehensive details on using CAN-Rakshak: how to set up datasets, preprocess data, configure experiments, train/test models, and perform attacks.

---

## 📁 Repository Layout

```
CAN-Rakshak/
├── attacks/
│   ├── attack_handler/
│   │   ├── base.py
│   │   └── FGSM.py
│   └── FGSM/
│       ├── attack_utilities.py
│       ├── fgsm.py
│       └── generate_mask.py
├── datasets/
├── docs/
├── features/
│   ├── feature_extractors/
│   │   ├── base.py
│   │   ├── pixnet.py
│   │   └── stat_features.py
│   ├── image/
│   │   ├── data_frame.py
│   │   └── traffic_encoder.py
│   └── utilities.py
├── ids/
│   ├── base.py
│   ├── densenet161.py
│   ├── resnet.py
│   └── shannon.py
├── models/
├── splitters/
│   ├── base.py
│   ├── default.py
│   └── threeway.py
└── src/
    ├── attack_config.py
    ├── base_preprocessor.py
    ├── config.py
    ├── evaluate.py
    ├── get_attack.py
    ├── get_extractor.py
    ├── get_ids.py
    ├── get_splitter.py
    ├── main.py
    ├── preprocessing.py
    ├── test.py
    ├── train.py
    └── utilities.py
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/spring-iitd/CAN-Rakshak.git
   cd CAN-Rakshak
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## 🧩 Dataset Preparation

Each dataset should be placed in `datasets/<DATASET_NAME>/` and contain a preprocessing script to generate a processed CSV file.

**Example structure:**

```
datasets/
└── Car_Hacking_Dataset/
    ├── preprocess.py
    └── data.csv
```

---

### Preprocessing Contract

* The first line of `preprocess.py` must import the base preprocessor:

  ```python
  from src.base_preprocessor import *
  ```

* The preprocessing class must inherit `DataPreprocessor` and implement:

  ```python
  def preprocess_dataset(self, input_dir: str, output_csv: str):
      ...
  ```

* The output CSV must have the header:

  ```
  Timestamp, can_id, dlc, byte0, byte1, byte2, byte3, byte4, byte5, byte6, byte7, labels
  ```

* If `dlc < 8`, prepend zeros to ensure 8 byte columns.

---

### Example `preprocess.py`

```python
from base_preprocessor import *
import os

def CH_to_CANbusData(original_folder_path, modified_folder_path):
    """
    Converts a raw CAN dataset log file into standardized CSV format.

    Each output CSV file will contain:
    Timestamp, can_id, dlc, byte0..byte7, labels
    
    Note:
    - Access your raw input files from the original folder path (original_folder_path).
    - Save your processed CSV files into the modified folder path (modified_folder_path).
    """
    csv_file = os.path.join(
        modified_folder_path,
        "your csv file"
    )

    user_file = os.path.join(
        original_folder_path,
        "your data file"
    )
    print(f"Converting {user_file} -> {csv_file}")
    os.makedirs(modified_folder_path, exist_ok=True)
    os.makedirs(original_folder_path, exist_ok=True)


    # write your logic for processing
    with open(user_file, 'r') as infile, open(csv_file, 'w') as outfile:
        for line in infile:
            columns = line.strip().split(",")
            timestamp = columns[0]
            can_id = columns[1]
            dlc = int(columns[2])

            # Pad missing databytes if dlc < 8
            data = (8 - dlc) * ["00"] + columns[3:3 + dlc]
            label = columns[-1].strip()

            data_str = ",".join(data)
            output_line = f"{timestamp},{can_id},{dlc},{data_str},{label}\\n"
            outfile.write(output_line)


class CarHackingPreprocessor(DataPreprocessor):
    """
    Example preprocessing class for CAN datasets.
    Takes raw log files from the input directory and converts them into
    processed CSVs stored in the output directory.

    original_folder_path: The folder path containing original raw CAN log files provided by the user.
    modified_folder_path: The folder path where processed CSV files will be stored after preprocessing.
    
    The user must ensure they access files from the original folder path (original_folder_path)
    and save processed CSV files into the modified folder path (modified_folder_path).
    """

    def preprocess_dataset(self, original_folder_path: str, modified_folder_path: str):
        print("Started preprocessing using the provided script...")
        
        CH_to_CANbusData(original_folder_path, modified_folder_path)
```

##

---

## 🧠 Training & Evaluation

1. **Execute:**

   ```bash
   python src/main.py
   ```

---

## ⚔️ Performing Attacks

To perform an attack:

1. Open `src/config.py` and set the desired attack name:

   ```python
   ATTACK_NAME = "fgsm"  # or another attack defined in attack_config.py
   ```
2. Modify `src/attack_config.py` to specify the parameters for your selected attack (e.g., target IDs, duration, rate).
3. Run the framework again using:

   ```bash
   python src/main.py
   ```
4. Results after the attack execution and evaluation will be saved under the corresponding dataset’s `Results/` folder.

---

## ✅ Best Practices

* Always validate your CSV format (12 columns, 8 databyte fields per row).
* Document dataset specifics inside `datasets/<DATASET_NAME>/README.md`.

---

## 🔄 Example Workflow Summary

```bash
git clone https://github.com/spring-iitd/CAN-Rakshak.git
cd CAN-Rakshak
pip install -r requirements.txt

mkdir -p datasets/Car_Hacking_Dataset/raw_files
# Add raw dataset files

python src/main.py
```

Results → `datasets/Car_Hacking_Dataset/Results/`

---

**CAN-Rakshak** — A comprehensive CAN IDS evaluation framework developed at IIT Delhi.

