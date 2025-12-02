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
    parser.add_argument("--stride", type=int, default=16, help="Frame stride (16=non-overlapping)")
    return parser.parse_args()

def load_video_frames(video_path, pattern):
    """Loads all frame paths for a video, sorted numerically."""
    # Pattern expects format like: VideoName_x264_FrameNum.png
    frames = sorted(glob.glob(os.path.join(video_path, "*.png")), 
                   key=lambda x: int(re.search(r'_(\d+)\.png', x).group(1)))
    return frames

def process_video(extractor, frames, transform, batch_size, stride=16, clip_len=16):
    """Runs inference on a single video."""
    video_features = []
    
    # Create batches
    # We loop from 0 to N with stride
    indices = range(0, len(frames) - clip_len + 1, stride)
    
    # We process in chunks (batches) to maximize GPU usage
    batch_tensor = []
    
    for start_idx in indices:
        # 1. Load Clip
        clip_paths = frames[start_idx : start_idx + clip_len]
        
        clip_imgs = []
        for p in clip_paths:
            img = cv2.imread(p)
            if img is None: 
                # Black frame fallback
                img = np.zeros((240, 320, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            clip_imgs.append(img)
        
        clip_np = np.array(clip_imgs)
        
        # 2. Transform -> Tensor
        # Using mode='val' for deterministic center crop
        tensor = transform(clip_np) 
        batch_tensor.append(tensor)
        
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
    # Structure: output_dir/rgb/ or output_dir/motion/
    save_root = os.path.join(args.output_dir, args.mode)
    os.makedirs(save_root, exist_ok=True)
    print(f"🚀 Extracting {args.mode.upper()} features to {save_root}")

    # 2. Initialize Model
    extractor = FeatureExtractor(args.checkpoint, device=device)
    
    # 3. Initialize Transform
    # IMPORTANT: We use 'val' mode to get CenterCrop (Deterministic features)
    if args.mode == 'rgb':
        transform = RGBVideoTransform(mode='val', crop_size=112, resize_size=128)
    else:
        transform = SobelMotionTransform(mode='val', crop_size=112, resize_size=128)

    # 4. Find all Video Folders
    # Looking for: data/Frames/Train/Class/VideoName
    # We walk the directory
    video_folders = []
    for split in ['Train', 'Test']:
        split_path = os.path.join(args.data_dir, split)
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path): continue
            
            # Get video folders inside class folder
            vids = [os.path.join(class_path, v) for v in os.listdir(class_path)]
            video_folders.extend(vids)
            
    print(f"📹 Found {len(video_folders)} videos to process.")

    # 5. Processing Loop
    for video_path in tqdm(video_folders):
        vid_name = os.path.basename(video_path)
        save_path = os.path.join(save_root, f"{vid_name}.npy")
        
        # Resume Check
        if os.path.exists(save_path):
            continue
            
        frames = load_video_frames(video_path, r'(.+?)_(\d+)\.png')
        if len(frames) < 16:
            continue
            
        features = process_video(extractor, frames, transform, args.batch_size, args.stride)
        
        if features is not None:
            np.save(save_path, features)

    print("✅ Extraction Complete.")

if __name__ == "__main__":
    main()