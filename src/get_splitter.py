from common_imports import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import splitters


def get_splitter(input_dir, mode, feature_extractor, cfg):
    for splitter_class in splitters.__all_classes__:
        if splitter_class.__name__.lower() in mode.lower():
            print(f"  Splitter       : {splitter_class.__name__}")
            return splitter_class(input_dir, feature_extractor, cfg).split()
    raise Exception(f"Unknown split mode: {mode}")
