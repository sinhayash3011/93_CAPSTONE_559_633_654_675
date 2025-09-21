"""
utils.py
Utility functions for metrics and plotting.
"""
import numpy as np
import matplotlib.pyplot as plt
from haversine import haversine

# Metrics
def haversine_rmse(y_true, y_pred):
    """
    Computes Haversine RMSE between true and predicted GPS coordinates.
    Args:
        y_true (np.ndarray): Ground truth GPS coordinates (lat, lon).
        y_pred (np.ndarray): Predicted GPS coordinates (lat, lon).
    Returns:
        float: RMSE in meters.
    """
    errors = [haversine(tuple(a), tuple(b)) for a, b in zip(y_true, y_pred)]
    return np.sqrt(np.mean(np.square(errors)))

def mean_absolute_error(y_true, y_pred):
    """
    Computes MAE between true and predicted values.
    """
    return np.mean(np.abs(y_true - y_pred))

def drift_per_km(y_true, y_pred):
    """
    Computes drift per km.
    """
    # Implement drift calculation
    pass

def inference_latency(model, input_data):
    """
    Measures inference latency for a model.
    """
    import time
    start = time.time()
    _ = model(input_data)
    return time.time() - start

def compute_flops(model, input_data):
    """
    Estimates FLOPs for a model (optional).
    """
    # Implement FLOPs calculation (use ptflops or similar)
    pass

# Plotting

def plot_trajectory(gt_coords, pred_coords, title="Trajectory Plot"):
    """
    Plots ground-truth vs predicted GPS trajectory.
    """
    plt.figure(figsize=(8,6))
    plt.plot(gt_coords[:,0], gt_coords[:,1], label='Ground Truth')
    plt.plot(pred_coords[:,0], pred_coords[:,1], label='Predicted')
    plt.legend()
    plt.title(title)
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.show()

# Add more plotting functions for error histograms, CDF, boxplots, etc.
