from common_imports import os
from get_ids import get_model
from evaluate import evaluation_metrics

def test_model(modelName, modelPath, cfg, adv_attack=None, image=None, TestSplit=None):
    model = get_model(modelName)
    print(f"Loading model from {os.path.normpath(modelPath)}")
    model.load(modelPath)
    result = model.test(cfg=cfg)
    if isinstance(result, tuple) and len(result) == 2:
        preds, labels = result
        evaluation_metrics(preds, labels, cfg)
    print("Testing Completed")