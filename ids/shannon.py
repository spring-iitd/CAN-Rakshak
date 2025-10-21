
import os
from config import *
import pandas as pd
import numpy as np
from collections import Counter
import math
import os 
from ids.base import IDS
from config import *
from datetime import datetime 

class Shannon(IDS):

    def __init__(self, time_window: float = 0.1, k_factor: float = 5.0):
        """
        Initializes the analysis attack.

        Args:
            time_window (float): The duration in seconds of the sliding window
                                 for entropy calculation.
            k_factor (float): The number of standard deviations from the mean
                              to set the anomaly threshold.
        """
        self.time_window = time_window
        self.k_factor = k_factor
        self.mean_h_ = None  # Learned from data in fit()
        self.std_h_ = None   # Learned from data in fit()
        super().__init__()


    def _is_hex(self, s):
        """Checks if a string can be interpreted as a hexadecimal."""
        try:
            int(str(s), 16)
            return True
        except (ValueError, TypeError):
            return False



    def _calculate_shannon_entropy(self, data_list: list) -> float:
        """Calculates the Shannon entropy for a list of byte values."""
        if not data_list:
            return 0.0
        counts = Counter(data_list)
        total_symbols = len(data_list)
        entropy = -sum((c / total_symbols) * math.log2(c / total_symbols) for c in counts.values())
        return entropy

    def _get_window_entropies(self, df: pd.DataFrame) -> list:
        """Slides a time window over the DataFrame and calculates entropy for each."""
        start_time, end_time = df['timestamp'].min(), df['timestamp'].max()
        current_ts = start_time
        entropies = []
        while current_ts < end_time:
            window_end = current_ts + self.time_window
            window_df = df[(df['timestamp'] >= current_ts) & (df['timestamp'] < window_end)]
            all_bytes = [byte for byte_list in window_df['Byte_Values'] for byte in byte_list]
            if all_bytes:
                entropies.append(self._calculate_shannon_entropy(all_bytes))
            current_ts = window_end
        return entropies

    def fit_from_csv(self, normal_data_csv_path: str):
        """
        CUSTOM FIT METHOD: Learns the entropy baseline from a normal data CSV file.
        This must be called before 'apply'.
        """
        print(f"Fitting model from '{os.path.basename(normal_data_csv_path)}'...")
        columns = ["timestamp", "can_id", "dlc", "Raw_Data_Bytes", "label"]

        # Read the CSV without header, but apply custom column names
        df_normal = pd.read_csv(normal_data_csv_path, header=None, names=columns)
        
        normal_entropies = self._get_window_entropies(df_normal)
        
        if not normal_entropies:
            raise ValueError("Could not calculate entropy from the provided normal data.")
            
        self.mean_h_ = np.mean(normal_entropies)
        self.std_h_ = np.std(normal_entropies)
        print(f"Fit complete. Baseline Mean Entropy: {self.mean_h_:.4f}, Std Dev: {self.std_h_:.4f}")

    def train(self, X_train=None, Y_train=None, **kwargs):
        updated_csv_file = os.path.join(DIR_PATH,"..","datasets", DATASET_NAME, "modified_dataset", "normal_dataset.csv")
        self.fit_from_csv(updated_csv_file)   # normal data 
        

    def prepare_frame_list(self):
        attack_data_path = os.path.join(DIR_PATH,"..","datasets", DATASET_NAME, "modified_dataset", "user_dos_dataset.csv")
        print(f"\nLoading attack data from '{os.path.basename(attack_data_path)}' for 'apply' method...")
        df_attack = pd.read_csv(attack_data_path) # dos dataset
        
        # Convert hex strings in 'Raw_Data_Bytes' to actual bytes for the 'data' key
        df_attack['data'] = df_attack['Raw_Data_Bytes'].apply(
            lambda x: bytes([int(b.strip(), 16) for b in str(x).replace('[', '').replace(']', '').replace("'", '').replace(',', ' ').split(' ') if b.strip() and len(b.strip()) > 0])
        )
        
        # Convert DataFrame to a list of dictionaries ('frames')
        attack_frames = df_attack[['timestamp', 'data']].to_dict('records')
        print(f"Converted {len(attack_frames)} rows into 'frames' format.")
        return attack_frames

    

    def apply(self, frames: list[dict], **kwargs) -> list[dict]:
        """
        Analyzes frames for anomalies and adds an 'anomaly_detected' key.
        This method conforms to the StatisticalAttack structure.
        """
        if self.mean_h_ is None or self.std_h_ is None:
            raise RuntimeError("You must call a `fit` method (e.g., `fit_from_csv`) before using `apply`.")

        if not frames:
            return []

        print("Applying entropy analysis...")
        # Create a modifiable copy and convert to DataFrame for efficient analysis
        adv_frames = [f.copy() for f in frames]
        df_test = pd.DataFrame(adv_frames)
        # The 'data' key in frames should hold bytes. Convert to lists of ints.
        df_test['Byte_Values'] = df_test['data'].apply(lambda x: list(x))
        
        lower_thresh = self.mean_h_ - self.k_factor * self.std_h_
        upper_thresh = self.mean_h_ + self.k_factor * self.std_h_

        # Analyze data in windows
        start_time, end_time = df_test['timestamp'].min(), df_test['timestamp'].max()
        current_ts = start_time
        num_anomalies_found = 0
        
        while current_ts < end_time:
            window_end = current_ts + self.time_window
            window_indices = df_test.index[(df_test['timestamp'] >= current_ts) & (df_test['timestamp'] < window_end)].tolist()
            
            if window_indices:
                window_df = df_test.loc[window_indices]
                all_bytes = [byte for byte_list in window_df['Byte_Values'] for byte in byte_list]
                
                is_anomaly = False
                if all_bytes:
                    entropy_val = self._calculate_shannon_entropy(all_bytes)
                    if not (lower_thresh <= entropy_val <= upper_thresh):
                        is_anomaly = True
                        num_anomalies_found += 1
                
                # "Attack" all frames in this window by labeling them
                for idx in window_indices:
                    adv_frames[idx]['anomaly_detected'] = is_anomaly
            current_ts = window_end

        print(f"Analysis complete. Found {num_anomalies_found} anomalous windows.")
        return adv_frames

   
    def test(self, X_test=None, Y_test=None, **kwargs):
        attack_frames=self.prepare_frame_list()
        self.apply(attack_frames)

        pass

    def predict(self,X_test=None, **kwargs):
        pass

    
    def save(self, path):
        pass

    
    def load(self, path):
        pass
