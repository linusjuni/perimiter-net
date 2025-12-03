import os
import glob
import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.feature_extractor import FeatureExtractor
from src.datasets.transforms import RGBVideoTransform, SobelMotionTransform


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Features for MIL")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to frames")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/work3/s225224/ucf-crime/features",
        help="Output directory for .npy files",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained R3D checkpoint"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rgb", "motion"],
        default="rgb",
        help="Feature type: rgb or motion",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Inference batch size"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Frame stride (16 = non-overlapping clips)",
    )
    return parser.parse_args()


def group_frames_by_video(data_dir):
    """Group frame files by video ID."""
    video_dict = defaultdict(list)

    for split in ["Train", "Test"]:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            continue

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue

            frames = glob.glob(os.path.join(class_path, "*.png"))

            for fpath in frames:
                fname = os.path.basename(fpath)
                try:
                    parts = fname.rsplit("_", 1)
                    vid_id = parts[0]
                    frame_num = int(parts[1].split(".")[0])
                    video_dict[vid_id].append((frame_num, fpath))
                except:
                    continue

    # Sort frames by frame number
    final_dict = {}
    for vid, frames in video_dict.items():
        frames.sort(key=lambda x: x[0])
        final_dict[vid] = [x[1] for x in frames]

    return final_dict


def process_video(extractor, frames, transform, batch_size, stride=16, clip_len=16):
    """Extract features from a single video."""
    video_features = []
    indices = range(0, len(frames) - clip_len + 1, stride)
    batch_tensor = []

    for start_idx in indices:
        clip_paths = frames[start_idx : start_idx + clip_len]

        # Load clip frames
        clip_imgs = []
        for p in clip_paths:
            img = cv2.imread(p)
            if img is None:
                img = np.zeros((240, 320, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            clip_imgs.append(img)

        clip_np = np.array(clip_imgs)

        try:
            tensor = transform(clip_np)
            batch_tensor.append(tensor)
        except Exception:
            continue

        # Process batch
        if len(batch_tensor) == batch_size:
            batch_stack = torch.stack(batch_tensor)
            feats = extractor.extract(batch_stack)
            video_features.append(feats)
            batch_tensor = []

    # Process remaining clips
    if len(batch_tensor) > 0:
        batch_stack = torch.stack(batch_tensor)
        feats = extractor.extract(batch_stack)
        video_features.append(feats)

    if len(video_features) > 0:
        return np.concatenate(video_features, axis=0)
    return None


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_root = os.path.join(args.output_dir, args.mode)
    os.makedirs(save_root, exist_ok=True)
    print(f"🚀 Extracting {args.mode.upper()} features to {save_root}")

    # Initialize model and transform
    extractor = FeatureExtractor(args.checkpoint, device=device)

    if args.mode == "rgb":
        transform = RGBVideoTransform(mode="val", crop_size=112, resize_size=128)
    else:
        transform = SobelMotionTransform(mode="val", crop_size=112, resize_size=128)

    # Group frames by video
    print("Grouping frames by video ID...")
    video_map = group_frames_by_video(args.data_dir)
    print(f"📹 Found {len(video_map)} unique videos.")

    # Process all videos
    for vid_name, frames in tqdm(video_map.items(), desc="Extracting features"):
        save_path = os.path.join(save_root, f"{vid_name}.npy")

        # Skip if already processed
        if os.path.exists(save_path):
            continue

        # Skip videos with insufficient frames
        if len(frames) < 16:
            continue

        features = process_video(
            extractor, frames, transform, args.batch_size, args.stride
        )

        if features is not None:
            np.save(save_path, features)

    print("✅ Feature extraction complete!")


if __name__ == "__main__":
    main()
