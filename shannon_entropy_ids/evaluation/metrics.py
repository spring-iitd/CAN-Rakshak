# evaluation/metrics.py
import math
from sklearn.metrics import precision_score, recall_score, f1_score


def normalize_labels(y):
    """
    Converts labels to binary:
    0 = normal
    1 = attack

    Handles:
    - strings ('R', 'T', 'A')
    - booleans
    - ints
    - NaN
    """
    normalized = []

    for v in y:
        # Case 1: NaN → treat as normal
        if v is None or (isinstance(v, float) and math.isnan(v)):
            normalized.append(0)
            continue

        # Case 2: String labels
        if isinstance(v, str):
            v = v.strip().upper()
            if v in ["T", "A", "ATTACK", "1"]:
                normalized.append(1)
            else:
                normalized.append(0)
            continue

        # Case 3: Boolean
        if isinstance(v, bool):
            normalized.append(int(v))
            continue

        # Case 4: Numeric
        try:
            normalized.append(int(v))
        except Exception:
            normalized.append(0)

    return normalized


def evaluate(y_true, y_pred):
    y_true = normalize_labels(y_true)
    y_pred = normalize_labels(y_pred)

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
