from common_imports import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import synthetic_data


def get_augmentor(attack_type, cfg):
    for augmentor_class in synthetic_data.__all_classes__:
        if attack_type.lower() in augmentor_class.__name__.lower():
            print(f"  Augmentor      : {augmentor_class.__name__}")
            return augmentor_class(cfg)
    raise Exception(f"{attack_type} augmentor not yet implemented")
