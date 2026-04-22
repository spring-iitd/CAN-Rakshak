from common_imports import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import defense.retrainers

def get_defense_class(defense_method):
    for retrainer_class in defense.retrainers.__all_classes__:
        if defense_method.lower() == retrainer_class.__name__.lower():
            return retrainer_class()
    raise Exception(f"{defense_method} not yet implemented")
