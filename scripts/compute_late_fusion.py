import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import pickle
from datetime import datetime
# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mil import MILModel

def main():
    # --- CONFIGURATION ---
    # 1. RGB Config
    rgb_features_dir = "/work3/s225224/ucf-crime/features/rgb/Test"
    rgb_checkpoint = "/work3/s225224/ucf-crime/checkpoints/mil/mil_rgb_20251203_162251/best_model.pth"
    
    # 2. Motion Config
    motion_features_dir = "/work3/s225224/ucf-crime/features/motion/Test"
    motion_checkpoint = "/work3/s225224/ucf-crime/checkpoints/mil/mil_motion_20251203_162137/best_model.pth"
    
    # 3. Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/work3/s225224/ucf-crime/experiments/late_fusion/mil_late_fusion_predictions_{timestamp}.pkl"
    
    # 4. Settings
    input_dim = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ---------------------

    print("=" * 60)
    print("🚀 Starting Late Fusion Inference")
    print("=" * 60)

    # 1. Load Models
    print(f"Loading RGB Model from: {rgb_checkpoint}")
    model_rgb = MILModel(input_dim=input_dim).to(device)
    model_rgb.load_state_dict(torch.load(rgb_checkpoint, map_location=device))
    model_rgb.eval()

    print(f"Loading Motion Model from: {motion_checkpoint}")
    model_motion = MILModel(input_dim=input_dim).to(device)
    model_motion.load_state_dict(torch.load(motion_checkpoint, map_location=device))
    model_motion.eval()

    # 2. Find Test Videos (Intersect both folders)
    rgb_files = {f.stem: f for f in Path(rgb_features_dir).glob("*.npy")}
    motion_files = {f.stem: f for f in Path(motion_features_dir).glob("*.npy")}
    
    # Only process videos that exist in BOTH streams
    common_videos = sorted(list(set(rgb_files.keys()) & set(motion_files.keys())))
    print(f"-> Found {len(common_videos)} videos common to both streams.")

    predictions = {} # Format: {video_name: score_array}

    print("Running Inference & Fusion...")
    for vid_name in tqdm(common_videos):
        try:
            # Load Features
            feat_rgb = np.load(rgb_files[vid_name])
            feat_motion = np.load(motion_files[vid_name])
            
            # Sanity Check: Sync lengths
            min_len = min(feat_rgb.shape[0], feat_motion.shape[0])
            if min_len == 0: continue
            
            feat_rgb = feat_rgb[:min_len]
            feat_motion = feat_motion[:min_len]

            # Inference
            with torch.no_grad():
                # Note: We pass full length features (N, 512). 
                # The MILModel works on any length sequence.
                inp_rgb = torch.tensor(feat_rgb, dtype=torch.float32).unsqueeze(0).to(device)
                scores_rgb = model_rgb(inp_rgb).squeeze().cpu().numpy()

                inp_motion = torch.tensor(feat_motion, dtype=torch.float32).unsqueeze(0).to(device)
                scores_motion = model_motion(inp_motion).squeeze().cpu().numpy()

            # --- LATE FUSION ---
            # Average the scores
            final_scores = (scores_rgb + scores_motion) / 2.0
            
            # Store result
            predictions[vid_name] = final_scores
            
        except Exception as e:
            print(f"Error processing {vid_name}: {e}")

    # 3. Save Results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(predictions, f)
        
    print(f"✅ Saved predictions for {len(predictions)} videos to: {output_path}")

if __name__ == "__main__":
    main()