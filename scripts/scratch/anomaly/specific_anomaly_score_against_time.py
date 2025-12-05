import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.mil import MILModel

sns.set_style("whitegrid")
sns.set_palette("muted")

RGB_FEATURES_DIR = "/work3/s225224/ucf-crime/features/rgb/Test"
MOTION_FEATURES_DIR = "/work3/s225224/ucf-crime/features/motion/Test"

RGB_CHECKPOINT = "/work3/s225224/ucf-crime/checkpoints/mil/mil_rgb_20251204_131331/best_model.pth"
MOTION_CHECKPOINT = "/work3/s225224/ucf-crime/checkpoints/mil/mil_motion_20251204_131231/best_model.pth"

ANNOTATION_FILE = "/dtu/blackhole/10/187952/ucf-crime-blackhole/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
OUTPUT_DIR = "outputs/test_plots"

INPUT_DIM = 512
STRIDE = 16
ALPHA = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_VIDEO = "RoadAccidents133_x264"

def load_model(path):
    """Load a MIL model from checkpoint."""
    print(f"Loading: {path}")
    model = MILModel(input_dim=INPUT_DIM).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

def parse_annotations(annotation_path):
    """Parse and downsample annotations."""
    gt_intervals = {}
    SUBSAMPLE_RATE = 10 
    
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                name = parts[0].replace('.mp4', '')
                intervals = []
                try:
                    s1, e1, s2, e2 = map(int, parts[2:6])
                    if s1 != -1: 
                        intervals.append((s1 // SUBSAMPLE_RATE, e1 // SUBSAMPLE_RATE))
                    if s2 != -1: 
                        intervals.append((s2 // SUBSAMPLE_RATE, e2 // SUBSAMPLE_RATE))
                    gt_intervals[name] = intervals
                except ValueError:
                    continue
    return gt_intervals

def analyze_video(vid_name, rgb_model, motion_model):
    """Run inference and return scores."""
    rgb_path = os.path.join(RGB_FEATURES_DIR, f"{vid_name}.npy")
    motion_path = os.path.join(MOTION_FEATURES_DIR, f"{vid_name}.npy")
    
    if not os.path.exists(rgb_path) or not os.path.exists(motion_path):
        return None

    try:
        feat_rgb = np.load(rgb_path)
        feat_motion = np.load(motion_path)
        
        min_len = min(feat_rgb.shape[0], feat_motion.shape[0])
        if min_len == 0: 
            return None
        feat_rgb = feat_rgb[:min_len]
        feat_motion = feat_motion[:min_len]

        with torch.no_grad():
            inp_rgb = torch.tensor(feat_rgb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            scores_rgb = rgb_model(inp_rgb).detach().cpu().numpy().flatten()
            
            inp_motion = torch.tensor(feat_motion, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            scores_motion = motion_model(inp_motion).detach().cpu().numpy().flatten()

        fused = (ALPHA * scores_rgb) + ((1 - ALPHA) * scores_motion)
        
        return fused
    except Exception as e:
        print(f"Error on {vid_name}: {e}")
        return None

def plot_result(vid_name, scores, intervals):
    """Plot anomaly scores against time."""
    expanded = np.repeat(scores, STRIDE)
    smooth = gaussian_filter1d(expanded, sigma=16)
    
    colors = sns.color_palette("muted")
    
    plt.figure(figsize=(10, 5), dpi=600)
    
    if intervals:
        gt_plotted = False
        for start, end in intervals:
            end = min(end, len(expanded))
            if start < end:
                if not gt_plotted:
                    plt.axvspan(start, end, color='#ffcccc', alpha=0.5, label='Ground Truth')
                    gt_plotted = True
                else:
                    plt.axvspan(start, end, color='#ffcccc', alpha=0.5)
            
    plt.plot(expanded, color=colors[0], linewidth=1, alpha=0.7, label='Raw Score')
    plt.plot(smooth, color=colors[1], linewidth=2, label='Smoothed')
    
    display_name = vid_name.replace('_x264', '').replace('0', ' 0', 1) if 'Shooting' in vid_name else vid_name
    title = f"{display_name} (Max: {scores.max():.3f})"
    
    plt.title(title, fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Frame Index', fontsize=11)
    plt.ylabel('Anomaly Score', fontsize=11)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"{vid_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def main():
    """Main function to process a specific video."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading models...")
    rgb_model = load_model(RGB_CHECKPOINT)
    motion_model = load_model(MOTION_CHECKPOINT)
    gt_map = parse_annotations(ANNOTATION_FILE)
    
    print(f"Processing {TARGET_VIDEO}...")
    
    scores = analyze_video(TARGET_VIDEO, rgb_model, motion_model)
    if scores is None:
        print(f"Failed to process {TARGET_VIDEO}")
        return
    
    intervals = gt_map.get(TARGET_VIDEO, [])
    
    plot_result(TARGET_VIDEO, scores, intervals)
    print(f"Complete! Plot saved to {OUTPUT_DIR}/{TARGET_VIDEO}.png")

if __name__ == "__main__":
    main()