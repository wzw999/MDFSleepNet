"""
Preprocessing utilities for sleep EEG data
"""

import numpy as np
import os


def preprocess_generic_sleep_data():
    """
    Generic preprocessing function for sleep EEG data
    """
    print("Generic sleep data preprocessing")
    print("This script provides utilities for preprocessing various sleep datasets")
    
    # Add generic preprocessing functions here
    pass


def normalize_eeg_data(data, method='z-score'):
    """
    Normalize EEG data using different methods
    """
    if method == 'z-score':
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        normalized = (data - mean) / (std + 1e-8)
    elif method == 'min-max':
        min_val = np.min(data, axis=0, keepdims=True)
        max_val = np.max(data, axis=0, keepdims=True)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
    else:
        normalized = data
    
    return normalized


if __name__ == "__main__":
    preprocess_generic_sleep_data()
