# ============================================================
# common_imports.py
# Shared imports for the CAN_Rakshak pipeline.
#
# Usage in any module:
#   from common_imports import os, np, pd, tf, plt, datetime
#   from common_imports import confusion_matrix, f1_score, StandardScaler
#   from common_imports import nn, optim, DataLoader          # PyTorch
#   from common_imports import Model, Dense, load_model       # Keras
#
# Only import what your module actually uses — do not do
# `from common_imports import *` as that pollutes the namespace.
# ============================================================


# ----------------------------------------------------------
# Standard library
# ----------------------------------------------------------
import os
import sys
import csv
import abc
import random
import logging
import itertools
from abc import ABC, abstractmethod
from datetime import datetime


# ----------------------------------------------------------
# Numerical / data
# ----------------------------------------------------------
import numpy as np
import pandas as pd


# ----------------------------------------------------------
# Visualization
# ----------------------------------------------------------
import matplotlib.pyplot as plt


# ----------------------------------------------------------
# Scikit-learn
# ----------------------------------------------------------
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import joblib


# ----------------------------------------------------------
# TensorFlow / Keras
# Set TF_AVAILABLE = False if TensorFlow is not installed.
# ----------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential, load_model
    from tensorflow.keras.layers import (
        Input,
        Dense,
        Dropout,
        Flatten,
        BatchNormalization,
        Activation,
        Conv2D,
        MaxPooling2D,
        AveragePooling2D,
        GlobalAveragePooling2D,
        Concatenate,
        Add,
        Lambda,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import SparseCategoricalCrossentropy
    from tensorflow.keras.callbacks import EarlyStopping, Callback
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ----------------------------------------------------------
# PyTorch / TorchVision
# Set TORCH_AVAILABLE = False if PyTorch is not installed.
# ----------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset, Subset
    from torchvision import datasets, transforms
    from torchvision import models as tv_models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


