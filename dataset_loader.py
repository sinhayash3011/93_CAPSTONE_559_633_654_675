# dataset_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(filepath: str):
    """
    Load the dataset from CSV file.
    Expected columns: time, ax, ay, az, wx, wy, wz, Bx, By, Bz, latitude, longitude, altitude, speed
    """
    df = pd.read_csv(filepath)

    # Drop rows with missing values
    df = df.dropna()

    # Ensure time column is in datetime format
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    return df


def split_dataset(df, test_size=0.2, random_state=42):
    """
    Split dataset into train/test sets.
    Inputs: IMU + Magnetometer
    Outputs: GPS (latitude, longitude)
    """
    feature_cols = ["ax", "ay", "az", "wx", "wy", "wz", "Bx", "By", "Bz"]
    target_cols = ["latitude", "longitude"]

    X = df[feature_cols].values
    y = df[target_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage
    dataset_path = "IMU.csv"  # replace with your dataset path
    df = load_dataset(dataset_path)
    print("Dataset shape:", df.shape)

    X_train, X_test, y_train, y_test = split_dataset(df)
    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)
