import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class SlidingWindowAnomalyDetector:
    """
    Overlapping Sliding Windows Anomaly Detection using threshold-based method.
    """
    
    def __init__(self, window_size: int = 1000, percentile: float = 95.0, method: str = "upper"):
            """
            Initalize the anolmaly detector.
            
            Args:
                window_size (int): Size of the sliding window.
                percentile (float): Percentile to determine the threshold (0-100).
                method (str): "upper" for upper-tail only, "two-sided" for both tails

            """
            self.window_size = window_size
            self.percentile = percentile
            self.method = method
            self.predictions = None
            self.thresholds = None

    def detect_anomalies(self, data: np.array) -> Tuple[np.array, np.array]:
        """
        Detect anomalies in the data using sliding window approach.
        
        Args:
            data (np.array): Input time series data.
        
        Returns:
            preictions: Binary array (1 for anomaly, 0 for normal).
            thresholds: Array of threshold values for each window.
        """
        n = len(data)
        predictions = np.zeros(n, dtype)
        thresholds = np.zeros(n)
        
        if n < self.window_size:
            raise ValueError(f"Data length ({n}) must be >= window_size ({self.window_size})")
        
        # Window 1: cover indices [0, W-1}
        window_data = data[0:self.window_size]
        threshold = np.percentile(window_data, self.percentile, methid = "linear")

        # Label all points in the first window
        if self.method == "upper":
            predictions[0:self.window_size] = (data[0:self.window_size] >= threshold).astype(int)
        elif self.method == "two-sided":
            lower_threshold = np.percentile(window_data, 100 - self.percentile, method = "linear")
            predictions[0:self.window_size] = ((data[0:self.window_size] >= threshold) | 
                                              (data[0:self.window_size] <= lower_threshold)).astype(int)
            
        # Store the threshold for the first window
        thresholds[0:self.window_size] = threshold
        
        # Subsequent windows: Window i covers indices [i-1, i+W-2]
        for i in range(1, n - self.window_size + 1):
            window_start = i
            window_end = i + self.window_size
            new_point_idx = window_end - 1 # Index of the new point entering the window
            
            # Compute the threshold for the current window
            window_data = data[window_start:window_end]
            threshold = np.percentile(window_data, self.percentile, method = "linear")
            threshold[new_point_idx] = threshold
            
            # Label only the new point in the current window
            if self.method == "upper":
                predictions[new_point_idx] = int(data[new_point_idx] >= threshold)
                
            elif self.method == "two-sided":
                lower_threshold = np.percentile(window_data, 100 - self.percentile, method = "linear")
                predictions[new_point_idx] = int((data[new_point_idx] >= threshold) or
                                                  (data[new_point_idx] <= lower_threshold))
                
        self.predictions = predictions
        self.thresholds = thresholds
        return predictions, thresholds
    
    def evaluate_performance(self, true_labels: np.array) -> dict:
        """
        Calculate the performance metrics
        
        Args:
            true_labels:Ground truth binary labels (1 for anomaly, 0 for normal).
            
        Returns:
            Dictionary containing TP, FP, FN, TN and other accuracy metrics.
            
        """
        if self.predictions is None:
            raise ValueError("Must run detect_anomalies() before evaluating performance.")
        
        tp = np.sum((self.predictions == 1) & (true_labels == 1))
        fp = np.sum((self.predictions == 1) & (true_labels == 0))
        fn = np.sum((self.predictions == 0) & (true_labels == 1))
        tn = np.sum((self.predictions == 0) & (true_labels == 0))

        # Total acutal anomalies
        p = tp + fn
        # Total actual normal points
        n = tn + fp
        
        normal_accuracy = tn / n if n > 0 else 0
        anomaly_accuracy = tp / p if p > 0 else 0
        
        return {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "P": p, "N": n,
            "normal_accuracy": normal_accuracy,
            "anomaly_accuracy": anomaly_accuracy,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / p if p > 0 else 0,
            "f1_score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            
        }
    
def load_and_preprocess_data(filepath:str) -> Tuple[np.array, np.array]:
    """
    Load and preprocess the time series data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        nitrate_data: Nitrate values
        ground_truth: Groung truth anomaly labels
        
    """
    
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}
        print(f"Columns: {list(df.columns)}")
        
    
        # Extract the nitrate column
        if "Nitrate" in df.columns:
            nitrate_data = df["Nitrate"].values
        else:
        
            # TRy the common variations
            nitrate_cols = [col for col in df.columns if "nitrate" in col.lower()]
            if nitrate_cols:
                nitrate_data = df[nitrate_cols[0]].values
            else:
                raise ValueError("Nitrate column not found in the dataset.")
                
        # Remove any remaining NaNs values
        valid_indices = ~np.isnan(nitrate_data)
        nitrate_data = nitrate_data[valid_indices]
        print(f"Nitrate data length after removing NaNs: {len(nitrate_data)}")
        
        # Create sysnthetic ground truth labels for demonstration
        # In practice, these should come from domain experts or labeled data
        ground_truth = create_synthetic_ground_truth(nitrate_data)
        
        return nitrate_data, ground_truth
        
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        
        # Generate synthetic data for demonstration
        print("Generating synthetic nitrate data....
        return generate_synthetic_data()
        
def create_synthetic_ground_truth(data: np.array, target_anomalies: int = 77) -> np.array:
    """
    Create synthetic ground truth labels based on extreme values in the data.
    This is for demonstration - in practice there should be real ground truth labels.
    """
    
    n = len(data)
    ground_truth = np.zeros(n, dtype=int)
    
    # Mark highest calues as anomalies
    threshold = np.percentile(data, 99.5)
    anomaly_indices = np. where(data >= threshold)[0]
    
    # If there are too many, only select the highest values
    if len(anomaly_indices) > target_anomalies:
        sorted_indices = anomaly_indices[np.argsort(data[anomaly_indices])[::-1]]
        anomaly_indices = sorted_indices[:target_anomalies]


    ground_truth[anomaly_indices] = 1
    
    print(f"Created ground truth with {np.sum(ground_truth)} anomalies.")
    return ground_truth
    
    
def generate_synthetic_data() -> Tuple[np.array, np.array]:
    """
    Generate synthetic nitrate time series data for testing.
    
    """
    
    np.random.seed(42)
    n = 5000
    
    # Base trend and seasonality and noise
    t = np.arange(n)
    trend = 0.0001 * t
    seasonal = 2 * np.sin(2 * np.pi * t / 365) +np.sin(2 * np.pi * t / 30)
    noise = np.random.normal(0, 0.5, n)
    
    # Base nitrate levels (typical range 0-10 mg/L)
    base_level = 3.0
    data = base_level + trend + seasonal + noise
    
    