# model/shannon.py
import numpy as np
import pandas as pd
from utils.entropy import shannon_entropy

class ShannonIDS:

    def __init__(self, time_window, k_factor):
        self.time_window = time_window
        self.k_factor = k_factor
        self.mean_h = None
        self.std_h = None

    def fit(self, df):
        entropies = self._window_entropy(df)
        self.mean_h = np.mean(entropies)
        self.std_h = np.std(entropies)

    def apply(self, df):
        lower = self.mean_h - self.k_factor * self.std_h
        upper = self.mean_h + self.k_factor * self.std_h

        df["anomaly"] = False
        entropies = self._window_entropy(df, return_indices=True)

        for entropy, indices in entropies:
            if not (lower <= entropy <= upper):
                df.loc[indices, "anomaly"] = True

        return df

    def _window_entropy(self, df, return_indices=False):
        results = []
        t = df.timestamp.min()

        while t < df.timestamp.max():
            win = df[(df.timestamp >= t) & (df.timestamp < t + self.time_window)]
            bytes_ = [b for row in win.Byte_Values for b in row]
            if bytes_:
                h = shannon_entropy(bytes_)
                if return_indices:
                    results.append((h, win.index))
                else:
                    results.append(h)
            t += self.time_window

        return results
