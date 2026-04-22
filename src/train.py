from common_imports import os
from get_ids import get_model


def train_model(modelName, modelPath, cfg):
    model = get_model(modelName)
    dataset_path = os.path.join(cfg['dir_path'], "..", "datasets", cfg['dataset_name'])
    train_dataset_dir = os.path.join(dataset_path, "train", cfg['train_dataset_dir'])
    os.makedirs(train_dataset_dir, exist_ok=True)
    print("Starting Training")
    model.train(train_dataset_dir, cfg=cfg)

    model.save(modelPath)
    print(f"Model saved at {os.path.normpath(modelPath)}")


def retrain_model(modelPath, adversarial_dataset_path, cfg):
    model = get_model(cfg['model'])
    model.load(modelPath)
    print("Starting Retraining with Adversarial Examples")
    print("ADV Dataset path : ", adversarial_dataset_path)
    model.train(adversarial_dataset_path, cfg=cfg)

    base, ext = os.path.splitext(modelPath)
    retrained_model_path = base + "_retrained" + ext
    model.save(retrained_model_path)
    print(f"Retrained Model saved at {os.path.normpath(retrained_model_path)}")
