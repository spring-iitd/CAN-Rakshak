import os

# -------------------------
# Directory paths
# -------------------------
DIR_PATH = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Dataset configuration
# -------------------------
DATASET_NAME = "your dataset name here"

# write original filename for training and testing
FILE_NAME = "new_data.log"

# -------------------------
# Pipeline control flags
# -------------------------
# # whether preprocessing stage should be executed
PREPROCESS = True
# whether split stage should be executed
SPLIT = True

# -------------------------
# Split configuration
# -------------------------
# options: default, three or your custom split mode name
SPLIT_MODE = "default" 

SPLIT_RATIO = 0.2


# whether feature extraction should be done (for both training and testing)
FEATURE_EXTRACTION = True

# Options for FEATURE_EXTRACTOR : PixNet, Stat or your custom feature extractor name
FEATURE_EXTRACTOR = 'PixNet'

# -------------------------
# Model configuration
# -------------------------
# MODEL_NAME must include any one of these : Densenet161, ResNet, Shannon or your custom model name
MODEL_NAME = "your model name here" # e.g., Densenet161_demo
# Options for mode : train and test (Results will be saved in "Results" folder)
MODE = "train"
EPOCHS = 3 # Number of epochs for training

# The folder will be created inside train and test folders
TRAIN_DATASET_DIR = "train_images"

TEST_DATASET_DIR = "test_images"

# Options : FGSM, None or your custom attack name
# If Adv_attack is not None, then fill configuration in attack_config.py
ADV_ATTACK = None
