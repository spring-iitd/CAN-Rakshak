from common_imports import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ids

def get_model(model_name):
    for model_class in ids.__all_classes__:
        if model_name.lower() == model_class.__name__.lower():
            print("Entered into get_model and found model : ", model_class.__name__)
            return model_class()
    raise Exception(f"{model_name} not yet implemented")