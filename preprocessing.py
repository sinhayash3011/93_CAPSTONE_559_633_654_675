# preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def moving_average_filter(data, window_size=5):
    """
    Apply moving average filter for smoothing.
    Only applies to sensor features (not GPS outputs).
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode="same")

def preprocess_data(X_train, X_test, apply_smoothing=True):
    """
    Preprocess the dataset:
    1. Apply smoothing (optional)
    2. Normalize using StandardScaler
    """

    # Apply smoothing only on raw features (per column)
    if apply_smoothing:
        X_train = np.array([moving_average_filter(col) for col in X_train.T]).T
        X_test = np.array([moving_average_filter(col) for col in X_test.T]).T

    # Normalize / Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    # Quick test
    from dataset_loader import load_dataset, split_dataset

    dataset_path = "IMU.csv"  # replace with your dataset path
    df = load_dataset(dataset_path)

    X_train, X_test, y_train, y_test = split_dataset(df)

    X_train_prep, X_test_prep, scaler = preprocess_data(X_train, X_test)

    print("Before preprocessing:", X_train.shape, "→ After preprocessing:", X_train_prep.shape)
    print("First row (before):", X_train[0])
    print("First row (after):", X_train_prep[0])
