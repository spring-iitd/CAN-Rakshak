from evaluate import evaluation_metrics
from ..FGSM.fgsm import FGSM_attack
from .base import *
from contextlib import redirect_stdout, redirect_stderr
import os
from datetime import datetime


class FGSM(Attack):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def apply(self):
        cfg         = self.cfg
        attack_name = cfg['adv_attack'].lower()
        print(f"Selected adversarial attack: {attack_name}")

        surrogate_model_path = (
            os.path.join(cfg['dir_path'], "..", "models", cfg['surrogate_model'])
            if cfg['surrogate_model'] else None
        )
        target_model_path = (
            os.path.join(cfg['dir_path'], "..", "models", cfg['target_model'])
            if cfg['adv_attack_type'] == "blackbox"
            else surrogate_model_path
        )

        log_file_dir = os.path.join(cfg['dir_path'], "..", "datasets", cfg['dataset_name'], "log_files")
        os.makedirs(log_file_dir, exist_ok=True)
        timestamp = datetime.now().strftime("_%Y_%m_%d_%H%M%S")
        log_file  = os.path.join(log_file_dir, f"{attack_name}_attack{timestamp}.log")

        with open(log_file, "w") as f:
            with redirect_stdout(f), redirect_stderr(f):
                print("Making call to the attack : ", attack_name)
                preds, labels, output_path = FGSM_attack(surrogate_model_path, target_model_path, cfg)
                evaluation_metrics(preds, labels, cfg)
                return output_path
