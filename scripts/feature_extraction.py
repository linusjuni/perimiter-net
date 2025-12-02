import os
import glob
import re
import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import sys
from collections import defaultdict

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.feature_extractor import FeatureExtractor
from src.datasets.transforms import RGBVideoTransform, SobelMotionTransform

def parse_args():
    parser = argparse.ArgumentParser(description="Extract Features for MIL")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to frames (e.g. /dtu/blackhole/.../Frames)")
    parser.add_argument("--output_dir", type=str, default="/work3/s225224/ucf-crime/features",
                        help="Root directory to save .npy files")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained R3D .pth file")
    parser.add_argument("--mode", type=str, choices=['rgb', 'motion'], default='rgb',
                        help="Which stream are we extracting?")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    # Stride=16 means non-overlapping clips (standard for MIL)
    parser.add_argument("--stride", type=int, default=16, help="Frame stride") 
    return parser.parse_args()

def group_frames_by_video(data_dir):
    """
    Scans the directory and groups flat PNGs into video lists.
    Returns: Dict { 'VideoName': [path_0, path_1, ...] }
    """
    video_dict = defaultdict(list)
    
    # Iterate over Train and Test splits
    for split in ['Train', 'Test']:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path): continue
        
        # Iterate over Class folders (Abuse, Arson...)
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path): continue
            
            print(f"Scanning {split}/{class_name}...")
            # Glob all PNGs
            frames = glob.glob(os.path.join(class_path, "*.png"))
            
            for fpath in frames:
                fname = os.path.basename(fpath)
                # Logic: Abuse001_x264_100.png -> VidID: Abuse001_x264, Frame: 100
                try:
                    parts = fname.rsplit("_", 1) # Split at last underscore
                    vid_id = parts[0]
                    frame_num = int(parts[1].split(".")[0])
                    video_dict[vid_id].append((frame_num, fpath))
                except:
                    continue # Skip malformed files

    # Sort frames for each video
    print("Sorting frames...")
    final_dict = {}
    for vid, frames in video_dict.items():
        # Sort by frame number
        frames.sort(key=lambda x: x[0])
        # Keep only paths
        final_dict[vid] = [x[1] for x in frames]
        
    return final_dict

def process_video(extractor, frames, transform, batch_size, stride=16, clip_len=16):
    """Runs inference on a single video."""
    video_features = []
    
    # We loop from 0 to N with stride
    indices = range(0, len(frames) - clip_len + 1, stride)
    
    batch_tensor = []
    
    for start_idx in indices:
        # 1. Load Clip
        clip_paths = frames[start_idx : start_idx + clip_len]
        
        clip_imgs = []
        for p in clip_paths:
            img = cv2.imread(p)
            if img is None: 
                img = np.zeros((240, 320, 3), dtype=np.uint8) # Fallback dimensions
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            clip_imgs.append(img)
        
        clip_np = np.array(clip_imgs)
        
        # 2. Transform -> Tensor
        # Using mode='val' for deterministic center crop
        # Handle exceptions in transform (e.g. corrupt images)
        try:
            tensor = transform(clip_np) 
            batch_tensor.append(tensor)
        except Exception as e:
            print(f"Transform error: {e}")
            continue
        
        # 3. If Batch Full, Infer
        if len(batch_tensor) == batch_size:
            batch_stack = torch.stack(batch_tensor) # (B, 3, 16, H, W)
            feats = extractor.extract(batch_stack)  # (B, 512)
            video_features.append(feats)
            batch_tensor = []

    # Process remaining
    if len(batch_tensor) > 0:
        batch_stack = torch.stack(batch_tensor)
        feats = extractor.extract(batch_stack)
        video_features.append(feats)
        
    if len(video_features) > 0:
        return np.concatenate(video_features, axis=0) # (Total_Clips, 512)
    return None

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Output Directory
    save_root = os.path.join(args.output_dir, args.mode)
    os.makedirs(save_root, exist_ok=True)
    print(f"🚀 Extracting {args.mode.upper()} features to {save_root}")

    # 2. Initialize Model
    extractor = FeatureExtractor(args.checkpoint, device=device)
    
    # 3. Initialize Transform
    if args.mode == 'rgb':
        # Resize to 128, Crop 112 (Standard)
        transform = RGBVideoTransform(mode='val', crop_size=112, resize_size=128)
    else:
        # Sobel Motion Transform
        transform = SobelMotionTransform(mode='val', crop_size=112, resize_size=128)

    # 4. Group Frames (The Fix for Critique #2)
    print("Grouping frames by video ID...")
    video_map = group_frames_by_video(args.data_dir)
    print(f"📹 Found {len(video_map)} unique videos.")

    # 5. Processing Loop
    for vid_name, frames in tqdm(video_map.items()):
        save_path = os.path.join(save_root, f"{vid_name}.npy")
        
        # Resume Check
        if os.path.exists(save_path):
            continue
            
        if len(frames) < 16:
            continue
            
        features = process_video(extractor, frames, transform, args.batch_size, args.stride)
        
        if features is not None:
            np.save(save_path, features)

    print("✅ Extraction Complete.")

if __name__ == "__main__":
    main()