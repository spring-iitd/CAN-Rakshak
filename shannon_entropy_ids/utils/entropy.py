# utils/entropy.py
import math
from collections import Counter

def shannon_entropy(data):
    if not data:
        return 0.0

    counts = Counter(data)
    total = len(data)
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
    )
