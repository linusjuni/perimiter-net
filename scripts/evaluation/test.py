import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def main():
    # --- CONFIGURATION ---
    # Input: The .pkl file created by compute_late_fusion.py
    # Update this filename to match the timestamp of your run!
    predictions_file = "/work3/s225224/ucf-crime/experiments/late_fusion/mil_late_fusion_predictions_20251203_170008.pkl"
    
    # Ground Truth
    annotation_file = "/dtu/blackhole/10/187952/ucf-crime-blackhole/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
    
    # Output for plots
    plot_dir = "outputs/plots"
    
    # Settings
    stride = 16  # Must match what you used during extraction
    # ---------------------

    print("=" * 60)
    print(f"Evaluating Predictions: {Path(predictions_file).name}")
    print("=" * 60)
    
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Load Predictions
    if not os.path.exists(predictions_file):
        print(f"❌ Error: File not found: {predictions_file}")
        return

    with open(predictions_file, 'rb') as f:
        predictions = pickle.load(f)
    print(f"-> Loaded predictions for {len(predictions)} videos.")

    # 2. Load Ground Truth
    gt_map = parse_annotations(annotation_file)
    print(f"-> Loaded annotations for {len(gt_map)} test videos.")

    global_preds = []
    global_gt = []
    processed_count = 0
    
    # 3. Align and Expand
    print("Aligning predictions with ground truth...")
    
    for vid_name, scores in predictions.items():
        if vid_name not in gt_map:
            continue
            
        # Expansion: Clip Score -> Frame Score
        # We repeat each score 'stride' times to fill the timeline
        frame_scores = np.repeat(scores, stride)
        
        # Create Ground Truth Mask
        # We infer total frames from the prediction length to ensure matching shapes
        total_frames = len(frame_scores)
        gt_mask = create_gt_mask(total_frames, gt_map[vid_name])
        
        global_preds.extend(frame_scores)
        global_gt.extend(gt_mask)
        processed_count += 1

    # 4. Compute Metrics
    if len(global_gt) == 0:
        print("❌ Error: No overlapping videos found between predictions and annotations.")
        return

    print(f"-> Successfully processed {processed_count} videos.")
    print("Computing AUC...")
    
    auc = roc_auc_score(global_gt, global_preds)
    
    print("\n" + "="*40)
    print(f"FINAL FRAME-LEVEL RESULT")
    print("="*40)
    print(f"Total Frames: {len(global_gt):,}")
    print(f"AUC Score:    {auc:.4f}")
    print("="*40)
    
    # 5. Plot ROC Curve
    fpr, tpr, _ = roc_curve(global_gt, global_preds)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Frame-Level)')
    plt.legend(loc="lower right")
    
    save_path = os.path.join(plot_dir, "roc_curve_late_fusion.png")
    plt.savefig(save_path)
    print(f"✅ Saved ROC plot to: {save_path}")

# --- Helper Functions ---
def parse_annotations(annotation_path):
    """
    Parses UCF-Crime text file.
    Returns: { 'VideoName': [(start1, end1), (start2, end2)] }
    """
    gt_intervals = {}
    if not os.path.exists(annotation_path):
        print(f"Error: Annotation file not found at {annotation_path}")
        return {}

    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                name = parts[0].replace('.mp4', '')
                intervals = []
                # Parsing logic: start1, end1, start2, end2
                try:
                    s1, e1, s2, e2 = map(int, parts[2:6])
                    if s1 != -1: intervals.append((s1, e1))
                    if s2 != -1: intervals.append((s2, e2))
                    gt_intervals[name] = intervals
                except ValueError:
                    continue
    return gt_intervals

def create_gt_mask(total_frames, intervals):
    """Creates binary mask (0/1) for a video of length total_frames."""
    mask = np.zeros(total_frames, dtype=np.int32)
    for start, end in intervals:
        # Clip coordinates to be within the actual video length
        s = max(0, start)
        e = min(total_frames, end)
        if s < e:
            mask[s:e] = 1
    return mask

if __name__ == "__main__":
    main()