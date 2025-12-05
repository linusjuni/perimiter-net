import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.mil import MILModel

RGB_FEATURES_DIR = "/work3/s225224/ucf-crime/features/rgb/Test"
MOTION_FEATURES_DIR = "/work3/s225224/ucf-crime/features/motion/Test"

RGB_CHECKPOINT = (
    "/work3/s225224/ucf-crime/checkpoints/mil/mil_rgb_20251204_131331/best_model.pth"
)
MOTION_CHECKPOINT = (
    "/work3/s225224/ucf-crime/checkpoints/mil/mil_motion_20251204_131231/best_model.pth"
)

ANNOTATION_FILE = "/dtu/blackhole/10/187952/ucf-crime-blackhole/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
OUTPUT_DIR = "outputs/test_plots"

INPUT_DIM = 512
STRIDE = 16
ALPHA = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path):
    """Load a MIL model from checkpoint."""
    print(f"Loading: {path}")
    model = MILModel(input_dim=INPUT_DIM).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def parse_annotations(annotation_path):
    """Parse and downsample annotations."""
    gt_intervals = {}
    SUBSAMPLE_RATE = 10

    with open(annotation_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                name = parts[0].replace(".mp4", "")
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
            inp_rgb = (
                torch.tensor(feat_rgb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            )
            scores_rgb = rgb_model(inp_rgb).detach().cpu().numpy().flatten()

            inp_motion = (
                torch.tensor(feat_motion, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            )
            scores_motion = motion_model(inp_motion).detach().cpu().numpy().flatten()

        fused = (ALPHA * scores_rgb) + ((1 - ALPHA) * scores_motion)

        return fused
    except Exception as e:
        print(f"Error on {vid_name}: {e}")
        return None


def plot_result(vid_name, scores, intervals, has_annotations=True):
    """Plot anomaly scores against time."""
    expanded = np.repeat(scores, STRIDE)
    smooth = gaussian_filter1d(expanded, sigma=16)

    plt.figure(figsize=(10, 5), dpi=150)

    if has_annotations and intervals:
        for start, end in intervals:
            end = min(end, len(expanded))
            if start < end:
                plt.axvspan(
                    start, end, color="#ffcccc", alpha=0.5, label="Ground Truth"
                )

    plt.plot(expanded, color="lightblue", linewidth=1, label="Raw Score")
    plt.plot(smooth, color="darkblue", linewidth=2, label="Smoothed")

    title = f"{vid_name} (Max: {scores.max():.3f})"
    if not has_annotations:
        title += " [NORMAL]"

    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Frame Index")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.tight_layout()

    category = "anomalous" if has_annotations and intervals else "normal"
    category_dir = os.path.join(OUTPUT_DIR, category)
    os.makedirs(category_dir, exist_ok=True)

    save_path = os.path.join(category_dir, f"{vid_name}.png")
    plt.savefig(save_path)
    plt.close()


def main():
    """Main function to process videos."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading models...")
    rgb_model = load_model(RGB_CHECKPOINT)
    motion_model = load_model(MOTION_CHECKPOINT)
    gt_map = parse_annotations(ANNOTATION_FILE)

    print("Processing all test videos and generating plots...")
    all_files = sorted(list(Path(RGB_FEATURES_DIR).glob("*.npy")))

    success_count = 0
    failed_count = 0

    for f in tqdm(all_files, desc="Plotting videos"):
        vid_name = f.stem

        scores = analyze_video(vid_name, rgb_model, motion_model)
        if scores is None:
            failed_count += 1
            continue

        intervals = gt_map.get(vid_name, [])
        has_annotations = len(intervals) > 0

        plot_result(vid_name, scores, intervals, has_annotations)
        success_count += 1

    print("Complete!")
    print(f"   - Successfully plotted: {success_count} videos")
    print(f"   - Failed: {failed_count} videos")
    print(f"   - Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
