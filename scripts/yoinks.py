import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

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

def interpolate_features(features: np.ndarray, segments: int = 32) -> np.ndarray:
    """Compress variable length video features into fixed segments (same as training)."""
    T, D = features.shape
    if T == segments:
        return features

    chunks = np.array_split(features, segments, axis=0)
    interpolated = np.zeros((segments, D), dtype=np.float32)

    for i, chunk in enumerate(chunks):
        if chunk.shape[0] > 0:
            interpolated[i] = np.max(chunk, axis=0)
        else:
            interpolated[i] = np.zeros(D)

    return interpolated


def expand_segment_scores_to_frames(
    segment_scores: np.ndarray, total_frames: int
) -> np.ndarray:
    """Expand 32 segment scores back to frame-level."""
    num_segments = len(segment_scores)
    frames_per_segment = total_frames / num_segments
    frame_scores = np.zeros(total_frames, dtype=np.float32)

    for i, score in enumerate(segment_scores):
        start = int(i * frames_per_segment)
        end = int((i + 1) * frames_per_segment)
        frame_scores[start:end] = score

    return frame_scores


def main():
    # --- CONFIGURATION ---
    # RGB Config
    rgb_features_dir = "/work3/s225224/ucf-crime/features/rgb/Test"
    rgb_checkpoint = "/work3/s225224/ucf-crime/checkpoints/mil/mil_rgb_20251203_181423/best_model.pth"

    # Motion Config
    motion_features_dir = "/work3/s225224/ucf-crime/features/motion/Test"
    motion_checkpoint = "/work3/s225224/ucf-crime/checkpoints/mil/mil_motion_20251203_181651/best_model.pth"

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
            
            # Store original clip counts for frame expansion
            num_clips_rgb = feat_rgb.shape[0]
            num_clips_motion = feat_motion.shape[0]
            
            # Interpolate to 32 segments (SAME AS TRAINING!)
            feat_rgb = interpolate_features(feat_rgb, segments=32)
            feat_motion = interpolate_features(feat_motion, segments=32)
            
            # Inference - model expects (1, 32, 512)
            with torch.no_grad():
                tensor_rgb = torch.from_numpy(feat_rgb).float().unsqueeze(0).to(device)
                tensor_motion = torch.from_numpy(feat_motion).float().unsqueeze(0).to(device)
                
                scores_rgb = model_rgb(tensor_rgb).squeeze().cpu().numpy()      # (32,)
                scores_motion = model_motion(tensor_motion).squeeze().cpu().numpy()  # (32,)
            
            # Expand 32 segment scores to frame-level
            # Use the number of clips * stride to get total frames
            total_frames_rgb = num_clips_rgb * stride
            total_frames_motion = num_clips_motion * stride
            total_frames = min(total_frames_rgb, total_frames_motion)
            
            frame_scores_rgb = expand_segment_scores_to_frames(scores_rgb, total_frames)
            frame_scores_motion = expand_segment_scores_to_frames(scores_motion, total_frames)
            
            # Create GT mask
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
    csv_path =  "weight_search_results.csv"
    with open(csv_path, "w") as f:
        f.write("alpha,rgb_weight,motion_weight,auc\n")
        for r in results:
            f.write(
                f"{r['alpha']:.2f},{r['alpha'] * 100:.1f},{(1 - r['alpha']) * 100:.1f},{r['auc']:.4f}\n"
            )
    print(f"✅ Saved results to: {csv_path}")

    # Plot
    alphas = [r["alpha"] for r in results]
    aucs = [r["auc"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(alphas, aucs, "b-o", linewidth=2, markersize=6)
    plt.axvline(
        x=best_result["alpha"],
        color="r",
        linestyle="--",
        label=f"Best α={best_result['alpha']:.2f}",
    )
    plt.xlabel("Alpha (RGB Weight)", fontsize=12)
    plt.ylabel("Frame-Level AUC", fontsize=12)
    plt.title("Late Fusion Weight Search: RGB vs Motion", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add secondary x-axis labels
    ax = plt.gca()
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(alphas[::4])
    ax2.set_xticklabels([f"{(1 - a) * 100:.0f}%" for a in alphas[::4]])
    ax2.set_xlabel("Motion Weight", fontsize=12)

    plot_path = "weight_search_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
