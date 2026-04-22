from common_imports import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import features.feature_extractors


def get_extractor(feature_extractor, cfg):
    for extractor_class in features.feature_extractors.__all_classes__:
        if extractor_class.__name__.lower() in feature_extractor.lower():
            print(f"  Extractor      : {extractor_class.__name__}")
            return extractor_class(cfg)
    raise Exception(f"{feature_extractor} not yet implemented")
