# config/paths.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

NORMAL_DATA = os.path.join(DATASET_DIR, "Normal_Data - standardized.csv")
ATTACK_DATA = os.path.join(DATASET_DIR, "standardized_DoS_attack.csv")

MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "shannon.pkl"
)