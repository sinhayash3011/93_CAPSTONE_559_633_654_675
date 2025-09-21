# enhanced_evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import time
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from haversine import haversine

def haversine_distance(y_true, y_pred):
    """
    Calculate Haversine distance (great circle distance) between points.
    
    Args:
        y_true: Array of true coordinates (latitude, longitude)
        y_pred: Array of predicted coordinates (latitude, longitude)
        
    Returns:
        Array of distances in kilometers
    """
    distances = []
    for i in range(len(y_true)):
        # Extract (lat, lon) pairs
        true_point = (y_true[i, 0], y_true[i, 1])
        pred_point = (y_pred[i, 0], y_pred[i, 1])
        
        # Calculate haversine distance
        dist = haversine(true_point, pred_point, unit='km')
        distances.append(dist)
        
    return np.array(distances)

def drift_per_km(y_true, y_pred, total_distance=None):
    """
    Calculate drift per kilometer traveled.
    
    Args:
        y_true: Array of true coordinates
        y_pred: Array of predicted coordinates
        total_distance: Total distance traveled in km (if None, calculated from y_true)
        
    Returns:
        Drift per km ratio
    """
    if total_distance is None:
        # Calculate total distance traveled along ground truth path
        total_distance = 0
        for i in range(1, len(y_true)):
            total_distance += haversine(
                (y_true[i-1, 0], y_true[i-1, 1]), 
                (y_true[i, 0], y_true[i, 1]), 
                unit='km'
            )
    
    # Calculate total error distance
    error_distances = haversine_distance(y_true, y_pred)
    avg_error = np.mean(error_distances)
    
    # Avoid division by zero
    if total_distance == 0:
        return float('inf')
    
    return avg_error / total_distance

def evaluate_model(y_true, y_pred, model_name, inference_time=None, training_time=None):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        model_name: Name of the model
        inference_time: Time taken for inference (optional)
        training_time: Time taken for training (optional)
        
    Returns:
        Dictionary of metrics
    """
    # Traditional regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # GPS-specific metrics
    distances = haversine_distance(y_true, y_pred)
    avg_distance_error = np.mean(distances)
    max_distance_error = np.max(distances)
    p90_distance_error = np.percentile(distances, 90)
    drift = drift_per_km(y_true, y_pred)
    
    print(f"\n📊 {model_name} Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")
    print(f"Avg Distance Error (km): {avg_distance_error:.4f}")
    print(f"90% Distance Error (km): {p90_distance_error:.4f}")
    print(f"Max Distance Error (km): {max_distance_error:.4f}")
    print(f"Drift per km: {drift:.4f}")
    
    if inference_time is not None:
        print(f"Inference Time (s): {inference_time:.4f}")
    if training_time is not None:
        print(f"Training Time (s): {training_time:.4f}")

    return {
        "model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "avg_distance_error": avg_distance_error,
        "p90_distance_error": p90_distance_error,
        "max_distance_error": max_distance_error,
        "drift_per_km": drift,
        "inference_time": inference_time,
        "training_time": training_time
    }

def plot_trajectory(y_true, y_pred, model_name, fig_size=(10, 8)):
    """
    Enhanced trajectory plot with error visualization.
    
    Args:
        y_true: Array of true GPS coordinates
        y_pred: Array of predicted GPS coordinates
        model_name: Name of the model
        fig_size: Figure size tuple
    """
    plt.figure(figsize=fig_size)
    
    # Plot ground truth and predictions
    plt.plot(y_true[:,0], y_true[:,1], 'b-', label="Ground Truth", linewidth=2)
    plt.plot(y_pred[:,0], y_pred[:,1], 'r-', label="Predicted", linewidth=2)
    
    # Add markers for start and end points
    plt.plot(y_true[0,0], y_true[0,1], 'go', markersize=10, label="Start")
    plt.plot(y_true[-1,0], y_true[-1,1], 'mo', markersize=10, label="End")
    
    # Draw lines connecting corresponding points to visualize error
    for i in range(0, len(y_true), max(1, len(y_true)//20)):  # Show subset of error lines to avoid clutter
        plt.plot([y_true[i,0], y_pred[i,0]], [y_true[i,1], y_pred[i,1]], 
                 'k-', alpha=0.2)
    
    plt.title(f"GPS Trajectory Comparison - {model_name}", fontsize=14)
    plt.xlabel("Latitude", fontsize=12)
    plt.ylabel("Longitude", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_error_distribution(y_true, y_pred, model_name, fig_size=(10, 8)):
    """
    Plot error distribution metrics.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        model_name: Name of the model
        fig_size: Figure size tuple
    """
    errors = haversine_distance(y_true, y_pred)
    
    plt.figure(figsize=fig_size)
    gs = GridSpec(2, 2)
    
    # Histogram of error distribution
    ax1 = plt.subplot(gs[0, 0])
    sns.histplot(errors, bins=30, kde=True, ax=ax1)
    ax1.set_title('Error Distance Distribution', fontsize=12)
    ax1.set_xlabel('Error (km)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    
    # CDF of errors
    ax2 = plt.subplot(gs[0, 1])
    errors_sorted = np.sort(errors)
    p = 1. * np.arange(len(errors)) / (len(errors) - 1)
    ax2.plot(errors_sorted, p)
    ax2.set_title('Cumulative Distribution of Errors', fontsize=12)
    ax2.set_xlabel('Error (km)', fontsize=10)
    ax2.set_ylabel('Cumulative Probability', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Error over time/sequence
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(errors)
    ax3.set_title('Error Over Time/Sequence', fontsize=12)
    ax3.set_xlabel('Sequence Index', fontsize=10)
    ax3.set_ylabel('Error (km)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Box plot of errors
    ax4 = plt.subplot(gs[1, 1])
    sns.boxplot(y=errors, ax=ax4)
    ax4.set_title('Error Distribution Boxplot', fontsize=12)
    ax4.set_ylabel('Error (km)', fontsize=10)
    
    plt.suptitle(f'Error Analysis - {model_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def plot_comparison_results(results, metric_columns=None, fig_size=(12, 10)):
    """
    Plot comparison of multiple models based on various metrics.
    
    Args:
        results: DataFrame containing results for all models
        metric_columns: List of metrics to plot (if None, plot standard metrics)
        fig_size: Figure size tuple
    """
    if metric_columns is None:
        # Default metrics to plot
        metric_columns = ['RMSE', 'MAE', 'avg_distance_error', 'drift_per_km']
    
    results_df = pd.DataFrame(results)
    
    # Sort by the first metric for consistency
    sorted_df = results_df.sort_values(by=[metric_columns[0]])
    
    # Create subplots - one for each metric
    fig = plt.figure(figsize=fig_size)
    num_metrics = len(metric_columns)
    
    # Calculate rows and cols for subplot grid
    cols = min(2, num_metrics)
    rows = (num_metrics + cols - 1) // cols  # Ceiling division
    
    for i, metric in enumerate(metric_columns):
        ax = fig.add_subplot(rows, cols, i+1)
        
        # Create horizontal bar chart
        bars = ax.barh(sorted_df['model'], sorted_df[metric], color='skyblue')
        
        # Add value annotations to the bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', va='center')
        
        ax.set_title(f'{metric}', fontsize=12)
        ax.set_xlabel('Value', fontsize=10)
        ax.invert_yaxis()  # To have best model at the top
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Model Comparison by Different Metrics', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def plot_tradeoff_analysis(results, x_metric='inference_time', y_metric='avg_distance_error', 
                           size_metric='training_time', fig_size=(10, 8)):
    """
    Plot tradeoff analysis between two metrics with size indicating a third metric.
    
    Args:
        results: DataFrame containing results for all models
        x_metric: Metric for x-axis
        y_metric: Metric for y-axis
        size_metric: Metric for point size
        fig_size: Figure size tuple
    """
    results_df = pd.DataFrame(results)
    
    # Filter out any missing values
    valid_results = results_df.dropna(subset=[x_metric, y_metric])
    
    if len(valid_results) < 2:
        print(f"Not enough data with both {x_metric} and {y_metric} metrics for tradeoff analysis")
        return
    
    plt.figure(figsize=fig_size)
    
    # Scale size_metric for better visualization if it exists
    sizes = None
    if size_metric in valid_results.columns and not valid_results[size_metric].isna().all():
        sizes = 100 * (valid_results[size_metric] / valid_results[size_metric].max())
        sizes = sizes.fillna(100)  # Default size if missing
    
    # Create scatter plot
    scatter = plt.scatter(valid_results[x_metric], valid_results[y_metric], 
              s=sizes, alpha=0.7)
    
    # Add labels for each point
    for i, model in enumerate(valid_results['model']):
        plt.annotate(model, 
                   (valid_results[x_metric].iloc[i], valid_results[y_metric].iloc[i]),
                   xytext=(5, 5), textcoords='offset points')
    
    plt.title(f'Tradeoff Analysis: {y_metric} vs {x_metric}', fontsize=14)
    plt.xlabel(x_metric, fontsize=12)
    plt.ylabel(y_metric, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # If using a size metric, add a note
    if sizes is not None:
        plt.figtext(0.5, 0.01, f'Note: Point size represents {size_metric}', 
                    ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()
