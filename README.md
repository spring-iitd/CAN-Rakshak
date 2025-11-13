# CAN-Rakshak

CAN-Rakshak is a framework for evaluating and benchmarking CAN Intrusion Detection Systems (IDS). It provides tools for preprocessing CAN datasets, training IDS models, performing attacks, and evaluating their performance.

---

## 🚀 Quick Start

1. **Clone the repository:**

   ```bash
   git clone https://github.com/spring-iitd/CAN-Rakshak.git
   cd CAN-Rakshak
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset:**

   * Inside the cloned repo, go to the `datasets/` folder.
   * Create a folder with the name of your dataset (e.g., `Car_Hacking_Dataset`).
   * Add your dataset files and a preprocessing script inside it.
   * The preprocessing script must:

     * Include the line `from src.base_preprocessor import *`
     * Inherit the `DataPreprocessor` class.
     * Implement the abstract function `preprocess_dataset`.
     * Convert raw data into CSV format with columns:

       ```
       Timestamp, can_id, dlc, byte0..byte7, labels
       ```

       If `dlc` is 5, prepend 3 zeros to make 8 databytes.

4. **Modify configuration:**

   * Edit `src/config.py` to specify your dataset name, parameters, and (optionally) attack type.

5. **Train and Test IDS:**

   ```bash
   python3 src/main.py
   ```

   * The results after testing will be stored inside the `Results/` folder of your dataset.

6. **Perform an Attack:**

   * Set the attack name in `src/config.py` under `ATTACK_NAME`.
   * Modify `src/attack_config.py` to adjust attack parameters.
   * Re-run the IDS using:

     ```bash
     python3 src/main.py
     ```
   * Attack results will also be saved in your dataset’s `Results/` folder.

---

## 📘 Documentation

For complete setup, structure, and advanced details, visit the [📖 Full Documentation](docs/usage.md) or the [project wiki](https://github.com/spring-iitd/CAN-Rakshak/wiki).

---

## 📂 Repository Overview

```
CAN-Rakshak/
├── attacks/         # Attack modules and handlers
├── datasets/        # Datasets and results
├── docs/            # Full detailed documentation
├── features/        # Feature extraction utilities
├── ids/             # IDS model definitions
├── models/          # Saved models
├── splitters/       # Data split utilities
└── src/             # Core scripts (train, test, config, etc.)
```

---

## 🤝 Contributing

We welcome contributions! Please open an issue or pull request if you have improvements or new IDS/attack modules to add.

---

For any questions or setup assistance, refer to the full documentation:
👉 [CAN-Rakshak Docs](https://github.com/spring-iitd/CAN-Rakshak/wiki)

