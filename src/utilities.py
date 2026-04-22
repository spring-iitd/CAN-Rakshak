import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import torch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def hex_to_bits(hex_value, num_bits):
    """Convert hexadecimal value to binary string with specified number of bits."""
    return bin(int(hex_value, 16))[2:].zfill(num_bits)

def bits_to_hex(binary_str):
    """Convert binary string to hexadecimal."""
    return hex(int(binary_str, 2))[2:].upper()

def int_to_bin(int_num):
    """Converts an integer to binary string."""
    return bin(int_num)[2:]

def pad(value, length):
    """Pads a given value with leading zeros to match the desired length."""
    curr_length = len(str(value))
    zeros = '0' * (length - curr_length)
    return zeros + value

hex_to_dec = lambda x: int(x, 16)

def transform_data(data):
    """Transforms DataFrame by converting hex values in 'ID' and 'Payload' to decimal."""
    data['ID'] = data['ID'].apply(hex_to_dec)
    data['Payload'] = data['Payload'].apply(hex_to_dec)
    return data

def shift_columns(df):
    """Shifts specific columns in the DataFrame based on the 'dlc' value."""
    for dlc in [2, 5, 6]:
        target_columns = df.columns[3:]
        df[target_columns] = df[target_columns].astype(object)
        df.loc[df['dlc'] == dlc, target_columns] = (
            df.loc[df['dlc'] == dlc, target_columns]
            .shift(periods=8 - dlc, axis='columns', fill_value='00')
        )
    return df

def sequencify_data(X, y, seq_size=10):
    max_index = len(X) - seq_size + 1
    X_seq = []
    y_seq = []
    for i in range(0, max_index, seq_size):
        X_seq.append(X[i:i+seq_size])
        try:
            y_seq.append(1 if 1 in y[i:i+seq_size].values else 0)
        except AttributeError:
            y_seq.append(1 if 1 in y[i:i+seq_size] else 0)
    return np.array(X_seq), np.array(y_seq)

def balance_data(X_seq, y_seq):
    zero_indices = np.where(y_seq == 0)[0]
    one_indices = np.where(y_seq == 1)[0]
    num_zeros = len(zero_indices)
    np.random.seed(42)
    sampled_one_indices = np.random.choice(one_indices, num_zeros, replace=False)
    balanced_indices = np.concatenate([zero_indices, sampled_one_indices])
    np.random.shuffle(balanced_indices)
    X_seq_balanced = X_seq[balanced_indices]
    y_seq_balanced = y_seq[balanced_indices]
    return X_seq_balanced, y_seq_balanced

def sequencify(dataset, target, start, end, window):
    """Create a sequencified dataset for LSTM model."""
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset)
    for i in range(start, end+1):
        indices = range(i-window, i)
        X.append(dataset[indices])
        indicey = i - 1
        y.append(target[indicey])
    return np.array(X), np.array(y)

def df_to_csv(dataframe, output_file_path):
    dataframe.to_csv(output_file_path, index=False)
