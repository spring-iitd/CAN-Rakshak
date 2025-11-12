import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import *
from src.test import test_model
from preprocessing import *
from train import *
from test import *
from evaluate import *
from get_splitter import get_splitter
from get_extractor import get_extractor
from get_attack import get_attack

def main():
    # Set dataset path
    dataset_path = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME)
    
    preprocess(dataset_path, PREPROCESS)  
    get_extractor(FEATURE_EXTRACTOR)
    get_splitter(dataset_path, mode=SPLIT_MODE, feature_extractor=FEATURE_EXTRACTOR)
    train_test = MODE.lower()
    model_path = os.path.join(DIR_PATH, "..", "models", MODEL_NAME)
    
    if train_test == 'train':
        train_model(MODEL_NAME, model_path, ADV_ATTACK)
    elif train_test == 'test':
        test_model(MODEL_NAME, model_path, ADV_ATTACK)
    else:
        raise Exception(f"Not supported {train_test}")
    
    get_attack(ADV_ATTACK)
    
        
        
if __name__ == "__main__":
    main()