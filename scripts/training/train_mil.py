import torch
import numpy as np
import argparse
from pathlib import Path
import sys
from datetime import datetime
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.mil import MILModel
from src.utils.losses import MILRankingLoss
from src.utils.logger import get_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Train MIL Detective")
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/mil")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=60) # 30 Norm + 30 Anom
    return parser.parse_args()

class MILDataLoader:
    def __init__(self, feature_dir, mode='Train', segments=32):
        self.feature_dir = feature_dir
        self.mode = mode
        self.segments = segments
        self.normal_videos = [] 
        self.anomaly_videos = []
        self._load_data()
        
    def _load_data(self):
        print(f"Loading features from {self.feature_dir}...")
        files = list(Path(self.feature_dir).glob("*.npy"))
        
        for f in files:
            # Basic Split Logic:
            # We assume extract_features ran on ALL videos.
            # We need to filter based on 'mode' if possible, or just load all.
            # Ideally, you have separate folders features/Train and features/Test.
            # If not, this loads everything.
            
            try:
                feats = np.load(f)
                if feats.shape[0] == 0: continue
                
                is_normal = "Normal" in f.name or "Training_Normal" in f.name
                
                if is_normal:
                    self.normal_videos.append(feats)
                else:
                    self.anomaly_videos.append(feats)
            except:
                pass
        print(f"Loaded {len(self.normal_videos)} Normal, {len(self.anomaly_videos)} Anomaly.")

    def interpolate(self, features):
        """Compress T -> 32 segments."""
        T = features.shape[0]
        dim = features.shape[1]
        if T == self.segments: return features
        chunks = np.array_split(features, self.segments, axis=0)
        interpolated = np.zeros((self.segments, dim), dtype=np.float32)
        for i, chunk in enumerate(chunks):
            if chunk.shape[0] > 0:
                interpolated[i] = np.mean(chunk, axis=0)
        return interpolated

    def get_batch(self, batch_size=60):
        half = batch_size // 2
        idx_norm = np.random.randint(0, len(self.normal_videos), half)
        idx_anom = np.random.randint(0, len(self.anomaly_videos), half)
        
        norm_batch = [self.interpolate(self.normal_videos[i]) for i in idx_norm]
        anom_batch = [self.interpolate(self.anomaly_videos[i]) for i in idx_anom]
        
        return torch.tensor(np.array(norm_batch), dtype=torch.float32), \
               torch.tensor(np.array(anom_batch), dtype=torch.float32)

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        # Evaluate Anomaly Videos
        for feats in loader.anomaly_videos:
            inp = torch.tensor(loader.interpolate(feats)).unsqueeze(0).to(device)
            score = torch.max(model(inp)).item() # Video Score = Max Clip Score
            preds.append(score)
            labels.append(1)
        # Evaluate Normal Videos
        for feats in loader.normal_videos:
            inp = torch.tensor(loader.interpolate(feats)).unsqueeze(0).to(device)
            score = torch.max(model(inp)).item()
            preds.append(score)
            labels.append(0)
    return roc_auc_score(labels, preds)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger(__name__)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.output_dir) / f"mil_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    loader = MILDataLoader(args.feature_dir)
    
    # 2. Model
    model = MILModel(input_dim=512).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = MILRankingLoss()
    
    logger.info("Starting MIL Training...")
    
    for epoch in range(args.epochs):
        model.train()
        norm_in, anom_in = loader.get_batch(args.batch_size)
        norm_in, anom_in = norm_in.to(device), anom_in.to(device)
        
        preds_norm = model(norm_in)
        preds_anom = model(anom_in)
        
        loss, loss_dict = criterion(preds_norm, preds_anom)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 50 == 0:
            # Basic validation on the training set (Proxy)
            # Ideally use a separate Test Loader here
            auc = evaluate(model, loader, device)
            logger.info(f"Epoch {epoch+1}: Loss {loss.item():.4f} AUC {auc:.4f}")
            torch.save(model.state_dict(), save_dir / "mil_checkpoint.pth")

if __name__ == "__main__":
    main()