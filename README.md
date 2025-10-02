# Homework-2---Overlapping-Sliding-Windows-Anomaly-Detection
In this assignment you will detect anomalies in a Nitrate time series using a threshold-based method with fixed-size, overlapping sliding windows (step size = 1). You will compute a q-percentile threshold within each window and classify the newly added data point as normal or anomaly. Ground truth indicates there are 77 anomaly events in total.

## Assignment Requirements

Threshold-based method only (no supervised models).
Fixed window size, step size = 1.
Percentiles computed with numpy.percentile(..., method="linear").
Use the Nitrate column from AG_NO3_fill_cells_remove_NAN.csv.
Find and justify an effective window size and initial threshold strategy (q).
Evaluate against the provided ground truth (total anomalies = 77).
Metrics to Report

Define and report FP, TP, FN, TN and two accuracy rates:

TP (True Positive): predicted anomaly & actually anomaly
FP (False Positive): predicted anomaly but actually normal
FN (False Negative): predicted normal but actually anomaly
TN (True Negative): predicted normal & actually normal
Let P = TP + FN (total anomalies = 77), N = TN + FP (total normals).

Normal event detection accuracy: TN / N
Anomaly event detection accuracy: TP / P
Target performance:

Normal accuracy ≥ 80%
Anomaly accuracy ≥ 75%

### Anomaly Detection and Threshold Evolution

<img width="1409" height="891" alt="Screenshot 2025-10-02 at 7 59 27 AM" src="https://github.com/user-attachments/assets/c64eb160-d465-46d3-89fc-017bc2cf9677" />



### W and q Selection Rationale

The window size (W) of 1000 was chosen based on the need to provide a large enough context to determine a reliable pattern with variations.  This also provides a stable threshold estimation while still being responsive to changes in the data.  The will allow a balance between sensitivity and stability.

The percentile of 97.5 was selected to help determine extreme values along with balancing the sensitivity to anomalies.  The 97.5% threshold captures the top 2.5% of values in each window, and is effective in identifying significant deviations from normal behavior.

### Results Summary

<img width="395" height="241" alt="Screenshot 2025-10-02 at 8 26 43 AM" src="https://github.com/user-attachments/assets/53866beb-4f8f-4157-88a0-63e47908984f" />

### Detailed Design

One-sidded (upper-tail only) thresholding approach was used to identify the Nitrate anomalies because they typically present as extremely high concentrations.

Labeling all points in the first window will provide an initial baseline for the threshold. This will help to ensure that there are no unlabeled points at the start of the time series.

With removing any of the remaining NaN values, the project will be able to run a cleaned dataset without any interruptions. Having a robust percentile computation method without missing values will help to ensure that the threshold is accurate.