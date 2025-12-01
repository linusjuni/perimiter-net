# Project Scope Change: VIRAT → UCF-Crime Dataset

## Executive Summary

**Original Project:** Multi-class human activity recognition (walking, running, standing, carrying, gesturing, entering, exiting) using VIRAT Ground Camera Dataset for perimeter security.

**New Project:** Binary anomaly detection + multi-class anomaly classification using UCF-Crime dataset for surveillance security applications.

---

## Why We Changed

### VIRAT Dataset Problems Discovered

1. **Severe Class Imbalance:**
   - Walking: 48% (530 samples)
   - Running: 0.7% (8 samples)
   - Standing: 30%
   - Carrying: 15%
   - Gesturing: 6% (70 samples)
   - Entering: 0% (0 samples)
   - Exiting: 0% (0 samples)

2. **Two classes have ZERO samples** - unusable for training
3. **Very small dataset:** Only 1,748 total samples after filtering
4. **Multi-actor ambiguity:** 30% of data excluded due to label confusion

**Conclusion:** Training a 7-class (or even 5-class) action classifier on VIRAT would be extremely difficult and produce poor results due to extreme imbalance and tiny sample sizes.

---

## New Project Scope

### Focus: Real-Time Anomaly Detection in Surveillance Videos

**Goal:** Detect abnormal events (intrusions, crimes, suspicious behavior) vs normal surveillance activity using deep learning.

### Two-Phase Approach

#### Phase 1: Binary Anomaly Detection

- Class 0: Normal surveillance footage
- Class 1: Anomaly (any suspicious/criminal activity)
- **Primary metric:** AUC (Area Under ROC Curve)
- **Use case:** Real-time alerting system

#### Phase 2: Multi-Class Anomaly Classification

- 14 classes total: 1 Normal + 13 Anomaly types
- Anomaly types: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Road Accident, Robbery, Shooting, Shoplifting, Stealing, Vandalism
- **Primary metric:** Accuracy, per-class precision/recall
- **Use case:** Forensic analysis and detailed reporting

---

## Architecture (Unchanged - Still Valid!)

### Model 1: 3D CNN (R3D-18)

- Spatial-temporal feature learning from RGB clips
- Pretrained on Kinetics-400
- Captures motion dynamics directly

### Model 2: Two-Stream Fusion Network

- **Spatial stream:** Appearance features (what's happening)
- **Temporal stream:** Motion features (optical flow or frame differences)
- **Fusion:** Late fusion of both streams
- Better performance than single-stream

**Why both matter:** Some anomalies are motion-based (running, fighting), others are appearance-based (abandoned object, wrong person). Fusion captures both.

---

## UCF-Crime Dataset

### Dataset Characteristics

**Size:** 1,900 untrimmed surveillance videos (~128 hours total)

**Format:** Pre-extracted frames (every 10th frame from original videos)

**Class Distribution:**

- Normal: 1,610 videos (~950K frames)
- Anomalies: 290 videos distributed across 13 classes
- **Major imbalance:** Normal class dominates with ~20-40x more frames

**14 Total Classes:**

1. Normal Videos
2. Abuse
3. Arrest
4. Arson
5. Assault
6. Burglary
7. Explosion
8. Fighting
9. Road Accidents
10. Robbery
11. Shooting
12. Shoplifting
13. Stealing
14. Vandalism

---

## Dataset Download Process

### Method: Kaggle API on HPC

**Installation:**

```bash
# Install Kaggle CLI with uv
uv tool install kaggle

# Setup credentials
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
# Paste: {"username":"your_kaggle_username","key":"your_kaggle_api_key"}
chmod 600 ~/.kaggle/kaggle.json
```

**Download:**

```bash
# Create directory structure
mkdir -p /work3/s225224/ucf-crime/{data,checkpoints,experiments}

# Download dataset
cd /work3/s225224/ucf-crime/data/
kaggle datasets download -d odins0n/ucf-crime-dataset

# Unzip (takes ~1-2 hours, 12GB → ~18GB)
nohup unzip ucf-crime-dataset.zip &
```

---

## Dataset Location on HPC

### Directory Structure

``` bash
/work3/s225224/
├── perimeter-net/          # Original VIRAT project (39GB)
│   ├── checkpoints/
│   ├── data/               # VIRAT dataset
│   │   ├── annotations/
│   │   ├── splits/
│   │   └── videos/
│   ├── experiments/
│   └── verification_output/
└── ucf-crime/              # New anomaly detection project (~18GB)
    ├── checkpoints/        # Model checkpoints
    ├── data/               # UCF-Crime dataset
    │   ├── Train/
    │   │   ├── Abuse/
    │   │   ├── Arrest/
    │   │   ├── Arson/
    │   │   ├── Assault/
    │   │   ├── Burglary/
    │   │   ├── Explosion/
    │   │   ├── Fighting/
    │   │   ├── NormalVideos/  (~948K frames)
    │   │   ├── RoadAccidents/
    │   │   ├── Robbery/
    │   │   ├── Shooting/
    │   │   ├── Shoplifting/
    │   │   ├── Stealing/
    │   │   └── Vandalism/
    │   ├── Test/
    │   │   └── (same 14 classes)
    │   └── ucf-crime-dataset.zip  (12GB - can be deleted after extraction)
    └── experiments/
```

### Frame Counts (Confirmed)

- **Train/NormalVideos:** 947,768 frames
- **Train/Fighting:** 24,684 frames
- **Train/Robbery:** 41,493 frames
- **Train/Stealing:** 44,802 frames

### File Naming Convention

``` bash
{ClassName}{VideoNumber}_x264_{FrameNumber}.png

Examples:
- Fighting002_x264_1000.png
- Normal_Videos331_x264_71840.png
```

---

## How to Use the Dataset

### Quick Verification

```bash
# Check extraction completed
ls /work3/s225224/ucf-crime/data/Train/ | wc -l  # Should show: 14
ls /work3/s225224/ucf-crime/data/Test/ | wc -l   # Should show: 14

# Check frame counts
find /work3/s225224/ucf-crime/data/Train/NormalVideos/ -name "*.png" | wc -l
find /work3/s225224/ucf-crime/data/Train/Fighting/ -name "*.png" | wc -l

# Check sizes
du -sh /work3/s225224/ucf-crime/data/Train/
du -sh /work3/s225224/ucf-crime/data/Test/
```

---

## Dataloader Design Requirements

### Pattern: Follow VIRAT Structure

Your existing VIRAT dataloader structure:

```python
class VIRATDataset(Dataset):
    def __init__(self, root_dir, split='train', clip_len=16, transform=None)
    def _load_annotations()  # Parse metadata
    def _load_video_clip()   # Load frames
    def __getitem__(idx)     # Return (frames, label)
```

### UCF-Crime Dataloader Specifications

**Key Differences from VIRAT:**

1. **Frames already extracted** - no video decoding needed
2. **Sample consecutive frames** from the pre-extracted set
3. **Two dataset modes:** Binary and Multi-class

**Core Requirements:**

#### 1. Initialization

```python
class UCFCrimeDataset(Dataset):
    def __init__(
        self, 
        root_dir='/work3/s225224/ucf-crime/data',
        split='train',           # 'train' or 'test'
        clip_len=16,            # Number of frames per clip
        mode='binary',          # 'binary' or 'multiclass'
        transform=None,
        stride=8                # Stride for sampling clips
    ):
```

#### 2. Label Mapping

```python
# Binary mode
BINARY_LABELS = {
    'NormalVideos': 0,
    'Abuse': 1, 'Arrest': 1, 'Arson': 1, 'Assault': 1,
    'Burglary': 1, 'Explosion': 1, 'Fighting': 1,
    'RoadAccidents': 1, 'Robbery': 1, 'Shooting': 1,
    'Shoplifting': 1, 'Stealing': 1, 'Vandalism': 1
}

# Multi-class mode
MULTICLASS_LABELS = {
    'NormalVideos': 0, 'Abuse': 1, 'Arrest': 2, 'Arson': 3,
    'Assault': 4, 'Burglary': 5, 'Explosion': 6, 'Fighting': 7,
    'RoadAccidents': 8, 'Robbery': 9, 'Shooting': 10,
    'Shoplifting': 11, 'Stealing': 12, 'Vandalism': 13
}
```

#### 3. Sample Structure

```python
def _load_samples(self):
    """
    Parse directory structure to create list of clips.
    
    Each sample: (class_name, video_name, start_frame_idx, end_frame_idx)
    
    For each video (identified by unique prefix like "Fighting002_x264"):
    - Find all frames belonging to that video
    - Create non-overlapping clips of length clip_len with stride
    
    Returns: List of tuples (class_folder, frame_paths)
    """
```

#### 4. Frame Loading

```python
def _load_clip(self, frame_paths):
    """
    Load clip_len consecutive frames.
    
    Input: List of frame paths (sorted by frame number)
    Output: numpy array of shape (T, H, W, C) where T=clip_len
    
    - Read each frame with cv2.imread() or PIL
    - Stack into temporal sequence
    - Handle edge cases (fewer frames than clip_len → repeat last frame)
    """
```

#### 5. Transform Application

```python
def __getitem__(self, idx):
    """
    Return single clip sample.
    
    Returns:
        frames: torch.Tensor of shape (C, T, H, W) after transform
        label: int (0 or 1 for binary, 0-13 for multiclass)
    """
    # 1. Get sample metadata (class, video, frame range)
    # 2. Load clip_len frames
    # 3. Apply transform (from VIRAT: RGBVideoTransform)
    # 4. Map class name to label based on mode
    # 5. Return (frames, label)
```

### Frame Sampling Strategy

**Parse filenames to group by video:**

```python
# From: Fighting002_x264_1000.png
# Extract: video_id = "Fighting002_x264"
#          frame_num = 1000

# Group frames by video_id, then sample clips with stride
```

**Use stride for temporal sampling:**

```python
# For video with 100 frames, clip_len=16, stride=8:
# Clip 1: frames [0, 16)
# Clip 2: frames [8, 24)
# Clip 3: frames [16, 32)
# etc.
```

### Handling Class Imbalance

#### Option 1: Weighted Sampling in DataLoader

```python
from torch.utils.data import WeightedRandomSampler

# Compute class weights
class_counts = [count for each class]
weights = 1.0 / class_counts
sample_weights = [weights[label] for each sample]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)

train_loader = DataLoader(dataset, batch_size=16, sampler=sampler)
```

#### Option 2: Weighted Loss Function

```python
# In training script
class_weights = torch.tensor([1.0, 20.0])  # Weight anomaly class higher
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

#### Option 3: Focal Loss (Recommended for severe imbalance)

```python
# Handles class imbalance better than weighted CE
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()
```

---

## Example Frame Paths for Implementation

### Training Split

``` bash
/work3/s225224/ucf-crime/data/Train/Fighting/Fighting002_x264_0.png
/work3/s225224/ucf-crime/data/Train/Fighting/Fighting002_x264_10.png
/work3/s225224/ucf-crime/data/Train/Fighting/Fighting002_x264_20.png
...
/work3/s225224/ucf-crime/data/Train/NormalVideos/Normal_Videos331_x264_71840.png
```

### Parsing Strategy

```python
import re
from pathlib import Path

def parse_frame_filename(filename):
    """
    Parse: Fighting002_x264_1000.png
    Returns: ('Fighting002_x264', 1000)
    """
    match = re.match(r'(.+?)_(\d+)\.png', filename)
    if match:
        video_id = match.group(1)
        frame_num = int(match.group(2))
        return video_id, frame_num
    return None, None

# Group frames by video
frames_by_video = {}
for frame_path in sorted(Path(class_dir).glob('*.png')):
    video_id, frame_num = parse_frame_filename(frame_path.name)
    if video_id not in frames_by_video:
        frames_by_video[video_id] = []
    frames_by_video[video_id].append((frame_num, frame_path))

# Sort frames within each video
for video_id in frames_by_video:
    frames_by_video[video_id].sort(key=lambda x: x[0])
```

---

## Key Takeaways

✅ **More tractable project:** Binary classification simpler than 14-way with extreme imbalance  
✅ **Still relevant:** Perimeter security focus maintained  
✅ **Architecture stays:** R3D-18 and two-stream both applicable  
✅ **Better dataset:** Actually designed for surveillance anomaly detection  
✅ **Clear benchmarks:** Can compare to published results (75-98% AUC for binary)  
✅ **Realistic scope:** Achievable within project timeline  

---

## Next Implementation Steps

1. **Build UCF-Crime dataloader** (pattern after VIRAT)
   - Parse directory structure
   - Group frames by video
   - Sample clips with stride
   - Handle binary vs multi-class modes

2. **Adapt training script** for binary classification
   - Use weighted loss or focal loss
   - Track AUC metric (not just accuracy)
   - Add ROC curve plotting

3. **Train baseline R3D-18** on binary task
   - Start with frozen backbone (only train classifier)
   - Use pretrained Kinetics-400 weights

4. **Implement two-stream fusion**
   - Spatial stream: RGB frames
   - Temporal stream: Optical flow or frame differences
   - Late fusion strategy

5. **Compare results**
   - Binary: R3D vs Two-stream (AUC comparison)
   - Multi-class: If time permits

6. **Analysis and reporting**
   - Per-class performance
   - Confusion matrices
   - Failure case analysis
   - ROC curves

The code structure from VIRAT (transforms, training utils, evaluation loops) all transfers directly!

---

## Important Notes

### Dataset Sharing

- Your co-student can access the dataset at: `/work3/s225224/ucf-crime/data/`
- Grant read permissions: `chmod -R g+rX /work3/s225224/ucf-crime/`
- Or they can download their own copy using the same Kaggle API method

### Space Management

- Keep the zip file (`ucf-crime-dataset.zip`) if you want a backup
- Delete it to free up 12GB: `rm /work3/s225224/ucf-crime/data/ucf-crime-dataset.zip`
- Total extracted size: ~18GB

### Performance Expectations

- Binary anomaly detection: Target 85-90% AUC
- Multi-class classification: Target 60-75% accuracy
- Some classes (Arrest, Burglary) are known to be difficult due to label ambiguity
