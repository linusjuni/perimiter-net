# Context: Implementation of Frame-Level Evaluation for Anomaly Detection

## 1. Project Status

* **Task:** Surveillance Anomaly Detection (Binary: Normal vs. Anomaly).
* **Dataset:** UCF-Crime (High-Res Frames).
* **Model:** 3D ResNet-18 (R3D-18) pretrained on Kinetics-400.
* **Current State:**
  * We have a working training pipeline using **Focal Loss** and **AMP**.
  * We currently evaluate using **Clip-Level AUC** (i.e., treating every 16-frame clip as an independent data point).
  * *Current Metric:* `sklearn.metrics.roc_auc_score(clip_labels, clip_probs)`.

## 2. The Problem: "Bag of Clips" vs. "Continuous Timeline"

Our current evaluation is insufficient for the surveillance use case (and standard benchmarks like Sultani et al.):

1. **Loss of Temporal Context:** Clip-level evaluation treats a 5-minute video as hundreds of unrelated samples. It fails to measure if the model detects the *start* and *end* of an anomaly.
2. **The "Scene Bias" Issue:** If a video takes place in a dark alley, the model might predict high anomaly scores for the *entire* duration (background bias). Clip-level accuracy might look high (because the video *is* an anomaly video), but the system is useless if it flags the 4 minutes of normal walking before the crime happens.
3. **Benchmark Standard:** The standard metric for UCF-Crime and ShanghaiTech is **Frame-Level AUC**.

## 3. The Objective: Frame-Level AUC

We need to implement a new evaluation script (`evaluate_frame_level.py`) that performs **Temporal Localization**.

### The Logic

Instead of batching random clips, we must:

1. **Reconstruct the Video:** Process an entire test video from Frame 0 to Frame $N$ using a sliding window (e.g., `stride=1` or `stride=16`).
2. **Generate a Signal:** Stitch the predicted probabilities together to form a continuous 1D signal over time: $S(t)$.
3. **Temporal Smoothing:** Apply a filter (e.g., Gaussian or Moving Average) to reduce noise/flickering in the predictions.
4. **Compare to Ground Truth:**
    * Load the frame-level binary masks (0=Normal, 1=Anomaly) provided by the dataset metadata.
    * Compute AUC by comparing the continuous Signal $S(t)$ against the Binary Mask $Y(t)$ across *all frames of all testing videos concatenated together*.

## 4. Implementation Requirements

I need Python code to implement this pipeline.

**Inputs:**

* `model`: The trained PyTorch R3D-18 model.
* `test_loader`: A specific DataLoader that returns full video sequences (or clips grouped by video ID).
* `ground_truth_path`: Path to the UCF-Crime temporal annotations (start/end frames for anomalies).

**Desired Outputs:**

* **Frame-Level AUC Score** (0.0 to 1.0).
* **ROC Curve Plot:** Visualizing the trade-off between False Positive Rate and True Positive Rate at the frame level.
* *(Optional)* **Visualizations:** A plot showing the "Anomaly Score vs. Time" for specific videos, overlaid with the Ground Truth region (to see if the spike matches the crime).

**Constraints:**

* Must handle **sliding window inference** (stitching 16-frame clip predictions into a timeline).
* Must be memory efficient (cannot load 1,000 full videos into RAM at once).
