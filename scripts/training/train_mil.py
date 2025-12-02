import os
import torch
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
import random

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.mil import MILModel
from src.utils.losses import MILRankingLoss
from src.utils.logger import get_logger

def main():
    # --- Configuration ---
    # ONLY look at the Training folder. We will split this internally.
    feature_dir_train_source = "/work3/s225224/ucf-crime/features/rgb/Train"
    
    base_checkpoint_dir = "/work3/s225224/ucf-crime/checkpoints/mil"
    
    # Hyperparameters
    input_dim = 512       
    lr = 1e-3             
    weight_decay = 0.005  
    epochs = 2000         
    batch_size = 60       
    segments = 32         
    val_split = 0.20      # 20% of training data used for validation
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ---------------------

    logger = get_logger(__name__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"mil_rgb_{timestamp}"
    save_dir = Path(base_checkpoint_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting MIL Training: {run_name}")

    # --- 1. Data Preparation (Split Train/Val) ---
    logger.info(f"Scanning features from: {feature_dir_train_source}")
    
    # Gather all files
    all_files = list(Path(feature_dir_train_source).glob("*.npy"))
    
    # Separate Normal and Anomaly to stratify the split
    normal_files = [f for f in all_files if "Normal" in f.name]
    anomaly_files = [f for f in all_files if "Normal" not in f.name]
    
    # Shuffle
    random.shuffle(normal_files)
    random.shuffle(anomaly_files)
    
    # Split Normal
    n_split = int(len(normal_files) * (1 - val_split))
    train_norm_files = normal_files[:n_split]
    val_norm_files = normal_files[n_split:]
    
    # Split Anomaly
    a_split = int(len(anomaly_files) * (1 - val_split))
    train_anom_files = anomaly_files[:a_split]
    val_anom_files = anomaly_files[a_split:]
    
    logger.info(f"Data Split:")
    logger.info(f"  Train: {len(train_norm_files)} Normal, {len(train_anom_files)} Anomaly")
    logger.info(f"  Val:   {len(val_norm_files)} Normal, {len(val_anom_files)} Anomaly")

    # Initialize Loaders with specific file lists
    train_loader = MILDataLoader(train_norm_files + train_anom_files, segments=segments)
    val_loader = MILDataLoader(val_norm_files + val_anom_files, segments=segments)

    # --- 2. Model & Loss ---
    model = MILModel(input_dim=input_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = MILRankingLoss() 

    # --- 3. Training Loop ---
    best_val_auc = 0.0
    
    for epoch in range(epochs):
        model.train()
        
        # Get Batch
        norm_in, anom_in = train_loader.get_batch(batch_size)
        norm_in, anom_in = norm_in.to(device), anom_in.to(device)
        
        # Forward
        preds_norm = model(norm_in) 
        preds_anom = model(anom_in) 
        
        # Loss
        loss, loss_dict = criterion(preds_norm, preds_anom)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation (every 50 epochs)
        if (epoch + 1) % 50 == 0:
            # Validate on Held-Out Training Data (Video-Level AUC)
            val_auc = evaluate(model, val_loader, device)
            
            log_str = f"Epoch {epoch+1}: Loss {loss.item():.4f} (Rank {loss_dict['rank']:.4f}) | Val AUC: {val_auc:.4f}"
            
            # Save Best Model based on Validation AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), save_dir / "best_model.pth")
                log_str += " [New Best]"
            
            logger.info(log_str)
            
            # Always save latest
            torch.save(model.state_dict(), save_dir / "latest_model.pth")

    logger.info(f"MIL Training Complete. Best Val AUC: {best_val_auc:.4f}")


class MILDataLoader:
    """
    Custom loader that manages 'Bags' of videos.
    Loads list of file paths into RAM.
    """
    def __init__(self, file_list, segments=32):
        self.file_list = file_list
        self.segments = segments
        self.normal_videos = [] 
        self.anomaly_videos = []
        self._load_data()
        
    def _load_data(self):
        # print(f"   Loading {len(self.file_list)} features into RAM...")
        for f in self.file_list:
            try:
                feats = np.load(f)
                if feats.shape[0] == 0: continue
                
                is_normal = "Normal" in f.name
                
                if is_normal:
                    self.normal_videos.append(feats)
                else:
                    self.anomaly_videos.append(feats)
            except Exception as e:
                print(f"Error loading {f.name}: {e}")

    def interpolate(self, features):
        """Compress T clips -> 32 segments."""
        T = features.shape[0]
        dim = features.shape[1]
        
        if T == self.segments: return features
        
        chunks = np.array_split(features, self.segments, axis=0)
        interpolated = np.zeros((self.segments, dim), dtype=np.float32)
        
        for i, chunk in enumerate(chunks):
            if chunk.shape[0] > 0:
                interpolated[i] = np.mean(chunk, axis=0)
            else:
                interpolated[i] = np.zeros(dim)
        return interpolated

    def get_batch(self, batch_size=60):
        """Returns separate batches for Normal and Anomaly bags."""
        half = batch_size // 2
        
        # Random Sampling with Replacement
        idx_norm = np.random.randint(0, len(self.normal_videos), half)
        idx_anom = np.random.randint(0, len(self.anomaly_videos), half)
        
        norm_batch = [self.interpolate(self.normal_videos[i]) for i in idx_norm]
        anom_batch = [self.interpolate(self.anomaly_videos[i]) for i in idx_anom]
        
        return torch.tensor(np.array(norm_batch), dtype=torch.float32), \
               torch.tensor(np.array(anom_batch), dtype=torch.float32)

def evaluate(model, loader, device):
    """Evaluate Video-Level AUC on the provided loader."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        # Evaluate Anomaly Videos (Label 1)
        for feats in loader.anomaly_videos:
            inp = torch.tensor(loader.interpolate(feats)).unsqueeze(0).to(device)
            # Video Score = Max score of any segment in the video
            score = torch.max(model(inp)).item()
            preds.append(score)
            labels.append(1)
            
        # Evaluate Normal Videos (Label 0)
        for feats in loader.normal_videos:
            inp = torch.tensor(loader.interpolate(feats)).unsqueeze(0).to(device)
            score = torch.max(model(inp)).item()
            preds.append(score)
            labels.append(0)
            
    if len(labels) == 0: return 0.5
    return roc_auc_score(labels, preds)

if __name__ == "__main__":
    main()