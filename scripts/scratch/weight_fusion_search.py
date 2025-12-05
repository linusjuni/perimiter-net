import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mil import MILModel


def load_model(checkpoint_path, input_dim, device):
    """Load a MIL model from checkpoint."""
    model = MILModel(input_dim=input_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def parse_annotations(annotation_path):
    """Parses UCF-Crime annotation file."""
    gt_intervals = {}
    with open(annotation_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                name = parts[0].replace(".mp4", "")
                intervals = []
                try:
                    s1, e1, s2, e2 = map(int, parts[2:6])
                    if s1 != -1:
                        intervals.append((s1, e1))
                    if s2 != -1:
                        intervals.append((s2, e2))
                    gt_intervals[name] = intervals
                except ValueError:
                    continue
    return gt_intervals


def create_gt_mask(total_frames, intervals):
    """Creates binary mask for a video."""
    mask = np.zeros(total_frames, dtype=np.int32)
    for start, end in intervals:
        s = max(0, start)
        e = min(total_frames, end)
        if s < e:
            mask[s:e] = 1
    return mask


def main():
    # --- CONFIGURATION ---
    # RGB Config
    rgb_features_dir = "/work3/s225224/ucf-crime/features/rgb/Test"
    rgb_checkpoint = "/work3/s225224/ucf-crime/checkpoints/mil/mil_rgb_20251204_130129/best_model.pth"

    # Motion Config
    motion_features_dir = "/work3/s225224/ucf-crime/features/motion/Test"
    motion_checkpoint = "/work3/s225224/ucf-crime/checkpoints/mil/mil_motion_20251204_130253/best_model.pth"

    # Ground Truth
    annotation_file = "/dtu/blackhole/10/187952/ucf-crime-blackhole/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"

    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        f"/work3/s225224/ucf-crime/experiments/late_fusion/weight_search_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Settings
    input_dim = 512
    stride = 16
    alpha_values = np.arange(0.0, 1.05, 0.05)  # 0.0, 0.05, 0.10, ..., 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ---------------------

    print("=" * 60)
    print("🔍 Late Fusion Weight Search")
    print("=" * 60)

    # 1. Load Models
    print("Loading models...")
    model_rgb = load_model(rgb_checkpoint, input_dim, device)
    model_motion = load_model(motion_checkpoint, input_dim, device)
    print("✅ Models loaded")

    # 2. Load Ground Truth
    gt_map = parse_annotations(annotation_file)
    print(f"-> Loaded annotations for {len(gt_map)} videos")

    # 3. Find Common Videos
    rgb_files = {f.stem: f for f in Path(rgb_features_dir).glob("*.npy")}
    motion_files = {f.stem: f for f in Path(motion_features_dir).glob("*.npy")}
    common_videos = sorted(
        list(set(rgb_files.keys()) & set(motion_files.keys()) & set(gt_map.keys()))
    )
    print(f"-> Found {len(common_videos)} videos with both features and annotations")

    # 4. Run Inference Once (Store Raw Scores)
    print("\nRunning inference on all videos...")
    video_data = {}  # {vid_name: {'rgb': scores, 'motion': scores, 'gt': mask}}

    for vid_name in tqdm(common_videos):
        try:
            # Load features
            feat_rgb = np.load(rgb_files[vid_name])
            feat_motion = np.load(motion_files[vid_name])

            # Sync lengths
            min_len = min(feat_rgb.shape[0], feat_motion.shape[0])
            if min_len == 0:
                continue

            feat_rgb = feat_rgb[:min_len]
            feat_motion = feat_motion[:min_len]

            # Inference
            with torch.no_grad():
                inp_rgb = (
                    torch.tensor(feat_rgb, dtype=torch.float32).unsqueeze(0).to(device)
                )
                scores_rgb = model_rgb(inp_rgb).squeeze().cpu().numpy()

                inp_motion = (
                    torch.tensor(feat_motion, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )
                scores_motion = model_motion(inp_motion).squeeze().cpu().numpy()

            # Expand to frame-level
            frame_scores_rgb = np.repeat(scores_rgb, stride)
            frame_scores_motion = np.repeat(scores_motion, stride)

            # Create GT mask
            total_frames = len(frame_scores_rgb)
            gt_mask = create_gt_mask(total_frames, gt_map[vid_name])

            video_data[vid_name] = {
                "rgb": frame_scores_rgb,
                "motion": frame_scores_motion,
                "gt": gt_mask,
            }

        except Exception as e:
            print(f"Error processing {vid_name}: {e}")

    print(f"✅ Processed {len(video_data)} videos")

    # 5. Sweep Over Alpha Values
    print("\n" + "=" * 60)
    print("Sweeping fusion weights...")
    print("=" * 60)

    results = []

    for alpha in alpha_values:
        global_preds = []
        global_gt = []

        for vid_name, data in video_data.items():
            # Fused scores: alpha * RGB + (1 - alpha) * Motion
            fused = alpha * data["rgb"] + (1 - alpha) * data["motion"]
            fused = uniform_filter1d(fused, size=8, mode='nearest')
            global_preds.extend(fused)
            global_gt.extend(data["gt"])

        auc = roc_auc_score(global_gt, global_preds)
        results.append({"alpha": alpha, "auc": auc})
        print(
            f"  α={alpha:.2f} (RGB={alpha * 100:5.1f}%, Motion={(1 - alpha) * 100:5.1f}%) → AUC: {auc:.4f}"
        )

    # 6. Find Best Alpha
    best_result = max(results, key=lambda x: x["auc"])
    print("\n" + "=" * 60)
    print(f"🏆 BEST RESULT")
    print("=" * 60)
    print(f"  Alpha:  {best_result['alpha']:.2f}")
    print(f"  RGB:    {best_result['alpha'] * 100:.1f}%")
    print(f"  Motion: {(1 - best_result['alpha']) * 100:.1f}%")
    print(f"  AUC:    {best_result['auc']:.4f}")
    print("=" * 60)

    # 7. Save Results
    # CSV
    csv_path = "weight_search_results.csv"
    with open(csv_path, "w") as f:
        f.write("alpha,rgb_weight,motion_weight,auc\n")
        for r in results:
            f.write(
                f"{r['alpha']:.2f},{r['alpha'] * 100:.1f},{(1 - r['alpha']) * 100:.1f},{r['auc']:.4f}\n"
            )
    print(f"✅ Saved results to: {csv_path}")

    # Plot with Seaborn
    sns.set_style("whitegrid")
    sns.set_palette("muted")

    alphas = np.array([r["alpha"] for r in results])
    aucs = np.array([r["auc"] for r in results])
    
    # Convert alpha to percentages for plotting
    rgb_weights = alphas * 100
    motion_weights = (1 - alphas) * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Main line plot with RGB weight as percentage
    sns.lineplot(x=rgb_weights, y=aucs, marker='o', linewidth=2, markersize=6, ax=ax)
    
    # Best alpha vertical line
    best_rgb_weight = best_result["alpha"] * 100
    ax.axvline(
        x=best_rgb_weight,
        color=sns.color_palette("muted")[3],  # muted red
        linestyle="--",
        linewidth=2,
        label=f"Best RGB={best_rgb_weight:.0f}%",
    )
    
    ax.set_xlabel("RGB Weight", fontsize=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
    ax.set_ylabel("Frame-Level AUC", fontsize=12)
    ax.set_title("Late Fusion Weight Search: RGB vs Motion", fontsize=14, pad=20)
    ax.legend(fontsize=11)
    
    # Set adaptive y-axis limits with some padding
    y_min, y_max = aucs.min(), aucs.max()
    y_range = y_max - y_min
    y_padding = y_range * 0.1  # 10% padding
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Add secondary x-axis for Motion weight
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    # Sample every 4th value for cleaner labels
    sample_indices = list(range(0, len(motion_weights), 4))
    ax2.set_xticks([motion_weights[i] for i in sample_indices])
    ax2.set_xticklabels([f"{motion_weights[i]:.0f}%" for i in sample_indices])
    ax2.set_xlabel("Motion Weight", fontsize=12)
    
    # Invert the secondary x-axis so motion weight decreases left to right
    ax2.invert_xaxis()

    plot_path = "weight_search_plot.png"
    plt.savefig(plot_path, dpi=600, bbox_inches="tight")
    print(f"✅ Saved plot to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()