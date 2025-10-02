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
    
    # Add synthetic anomalies
    anomaly_indices = np.random.choice(n, 77, replace=False)
    data[anomaly_indices] += np.random.exponential(3, len(anomaly_indices))
    
    # Create ground truth labels
    ground_truth = np.zeros(n, dtype=int)
    ground_truth[anomaly_indices] = 1
    
    return data, ground_truth
    
def plot_results(data: np.array, predictions: np.array, ground_truth: np.array = None,
                 thresholds: np.array = None, title: str = "Anomaly Detection Results"):
    """
    Plot the time series data with detected anomalies and ground truth if available.
    
    """
    
    fig, axes - plt.subplots(2, 1, figsize=(15, 10))
    
    # Main plot
    axes[0].plot(data, color='blue', alpha=0.7, linewidth=1, label='Nitrate Data')
    
    anomaly_indices = np.where(predictions == 1)[0]
    if len(anomaly_indices) > 0:
        axes[0].scatter(anomaly_indices, data[anomaly_indices],
                        color='red', s=30, alpha=0.8, label+f"Detected Anomalies ({len(anomaly_indices)})")
        
    if ground_truth is not None:
        true_anomaly_indices = np.where(ground_truth == 1)[0]
        axes[0].scatter(true_anomaly_indices, data[true_anomaly_indices],
                        color='orange', s=20, alpha=0.6, marker='x', 
                        label=f"True Anomalies ({len(true_anomaly_indices)})")
        
    axes[0].set_title(title)
    axes[0].set_xlabel("Time Index")
    axes[0].set_ylabel("Nitrate Level (mg/L)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    
    # Threshold plot
    if thresholds is not None:
        axes[1].plot(thresholds, color='green', alpha=0.7, label="Dynamic Threshold")
        axes[1].plot(data, color='blue', alpha=0.5, label='Nitrate Data')
        axes[1].set_xlabel("Time Index")
        axes[1].set_ylabel("Value")
        axes[1].set_title("Dynamic Threshold Evolution")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()
    
    
def optimize_parameters(data: np.array, ground_truth: np.array) -> Tuple[int, float]:
    """
    Optimize the window size and percentile parameters.
    
    """
    
    print("Optimizing parameters...")
    
    window_sizes = [500, 750, 1000, 1250, 1500]
    percentiles = [90, 92.5, 95, 97.5, 99, 99.5]
    
    best_score = 0
    best_params = (1000, 95.0)
    
    results = []
    
    for w in window_sizes:
        if w >= len(data):
            continue
            
        for p in percentiles:
            detector = SlidingWindowAnomalyDetector(window_size=w, percentile=p)
            predictions, _ = detector.detect_anomalies(data)
            metrics = detector.evaluate_performance(ground_truth)
            
            
            #Combined score of the weighted averafe of normal and anomaly accuracy
            score = 0.4 * metrics["normal_accuracy"] + 0.6 * metrics["anomaly_accuracy"]
            results.append((w, p, score, metrics))
            
            if score > best_score:
                best_score = score
                best_params = (w, p)
                
                print(f"Best parameters: window_size={best_params[0]}, percentile={best_params[1]}")
                print(f"Best combined score: {best_score:.4f}")

                return best_params
                
                
            def main():
                # Main execution function.
                print("=== Overlapping Sliding Windows Anomaly Detection ===")
                
                # Load and preprocess data
                filepath = "AG_NO3_fill_cells_remove_NAN.csv"  # Update path as needed
                data, ground_truth = load_and_preprocess_data(filepath)
                
                
                print(f"Data statistics:")
    print(f"  Length: {len(data)}")
    print(f"  Mean: {np.mean(data):.3f}")
    print(f"  Std: {np.std(data):.3f}")
    print(f"  Min: {np.min(data):.3f}")
    print(f"  Max: {np.max(data):.3f}")
    print(f"  True anomalies: {np.sum(ground_truth)}\n")
    
    # Optimize parameters (optional)
    # best_w, best_p = optimize_parameters(data, ground_truth)
    
    # Use chosen parameters
    chosen_window_size = 1000
    chosen_percentile = 97.5
    
    print(f"Chosen parameters:")
    print(f"  Window size: {chosen_window_size}")
    print(f"  Percentile: {chosen_percentile}")
    print(f"  Method: upper-tail detection")
    
    print(f"\nRationale:")
    print(f"  - Window size {chosen_window_size}: Large enough to capture local patterns")
    print(f"    while remaining adaptive to changes. Represents ~{chosen_window_size/len(data)*100:.1f}% of data.")
    print(f"  - Percentile {chosen_percentile}%: Conservative threshold focusing on truly")
    print(f"    extreme values while maintaining sensitivity to anomalies.\n")
    
    # Create detector and run detection
    detector = SlidingWindowAnomalyDetector(
        window_size=chosen_window_size,
        percentile=chosen_percentile,
        method="upper"
    )
    
    print("Running anomaly detection...")
    predictions, thresholds = detector.detect_anomalies(data)
    
    # Evaluate performance
    metrics = detector.evaluate_performance(ground_truth)
    
    print(f"\n=== Performance Metrics ===")
    print(f"True Positives (TP):  {metrics['TP']}")
    print(f"False Positives (FP): {metrics['FP']}")
    print(f"False Negatives (FN): {metrics['FN']}")
    print(f"True Negatives (TN):  {metrics['TN']}")
    print(f"Total Positives (P):  {metrics['P']}")
    print(f"Total Negatives (N):  {metrics['N']}")
    print()
    print(f"Normal Event Detection Accuracy:  {metrics['normal_accuracy']:.3f} ({metrics['normal_accuracy']*100:.1f}%)")
    print(f"Anomaly Event Detection Accuracy: {metrics['anomaly_accuracy']:.3f} ({metrics['anomaly_accuracy']*100:.1f}%)")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-Score:  {metrics['f1_score']:.3f}")
    
    # Check target performance
    print(f"\n=== Target Performance Check ===")
    normal_target = metrics['normal_accuracy'] >= 0.80
    anomaly_target = metrics['anomaly_accuracy'] >= 0.75
    print(f"Normal accuracy ≥ 80%: {'✓' if normal_target else '✗'} ({metrics['normal_accuracy']*100:.1f}%)")
    print(f"Anomaly accuracy ≥ 75%: {'✓' if anomaly_target else '✗'} ({metrics['anomaly_accuracy']*100:.1f}%)")
    
    if normal_target and anomaly_target:
        print("🎉 Both accuracy targets achieved!")
    else:
        print("⚠️  Consider adjusting parameters to meet accuracy targets.")
    
    # Plot results
    plot_results(data, predictions, ground_truth, thresholds, 
                f"Nitrate Anomaly Detection (W={chosen_window_size}, q={chosen_percentile}%)")
    
    return detector, metrics

if __name__ == "__main__":
    detector, metrics = main()