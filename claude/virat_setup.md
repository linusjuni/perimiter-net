# VIRAT Dataset Setup for Perimeter-Net

## Overview

This project uses the **VIRAT Video Dataset** for training an activity classification model for perimeter security. The dataset contains surveillance videos with annotated human activities.

## Data Source

- **Dataset**: VIRAT Video Dataset Release 2.0 (Ground Camera)
- **Annotations**: DIVA program annotations from GitLab
- **Website**: <https://viratdata.org/>
- **Annotation Repo**: <https://gitlab.kitware.com/viratdata/viratannotations>

## Directory Structure

```bash
/work3/s225224/perimeter-net/
├── data/
│   ├── videos/          # 79 video files (.mp4)
│   ├── annotations/     # 79 annotation files (.activities.yml)
│   ├── splits/          # train.txt, val.txt, test.txt
│   └── raw/
│       └── viratannotations/  # Cloned annotation repository
├── checkpoints/         # Model checkpoints during training
└── experiments/         # Training logs and results
```

## Setup Steps

### 1. Create Directory Structure

```bash
mkdir -p /work3/s225224/perimeter-net/data/{videos,annotations,splits,raw}
mkdir -p /work3/s225224/perimeter-net/{checkpoints,experiments/{logs,results}}
```

### 2. Download Annotations

```bash
cd /work3/s225224/perimeter-net/data/raw/
git clone https://gitlab.kitware.com/viratdata/viratannotations.git
```

### 3. Download Videos

**Prerequisites**:

- Sign the [VIRAT Video Dataset Protection Agreement](https://viratdata.org/resources/VIRAT-Video-Data-Set-Protection-Agreement-1-4-11.pdf)
- Install girder-cli: `pip install girder-client --break-system-packages`

**Download command**:

```bash
cd /work3/s225224/perimeter-net/data/videos/
girder-cli --api-url https://data.kitware.com/api/v1 download 56f581ce8d777f753209ca43 .
```

This downloads all videos from the VIRAT Ground Dataset folder on Kitware (~35 GB, ~35 minutes).

### 4. Copy Annotations

```bash
cd /work3/s225224/perimeter-net/data/
\cp raw/viratannotations/train/*.activities.yml annotations/
\cp raw/viratannotations/validate/*.activities.yml annotations/
```

### 5. Create Data Splits

Run the split creation script:

```bash
cd ~/projects/perimiter-net/
uv run scripts/create_splits.py
```

This filters to only include videos that exist and creates train/val/test splits.

## Dataset Details

### Statistics

- **Total videos**: 79 (out of 119 annotated, 40 missing from download)
- **Total samples**: 1,748 activity clips
  - Train: 1,102 samples (45 videos)
  - Val: 228 samples (18 videos)
  - Test: 418 samples (16 videos)

### Activity Classes (7 total)

1. `activity_walking`
2. `activity_running`
3. `activity_standing`
4. `activity_carrying`
5. `activity_gesturing`
6. `Entering`
7. `Exiting`

### Video Specifications

- **Resolution**: 1920x1080 (Full HD)
- **Format**: MP4
- **Clip length**: 16 frames per sample

## Verification

Check that everything is set up correctly:

```bash
# Check video count
ls /work3/s225224/perimeter-net/data/videos/*.mp4 | wc -l  # Should be 79

# Check annotation count
ls /work3/s225224/perimeter-net/data/annotations/*.yml | wc -l  # Should be 79

# Check split files
wc -l /work3/s225224/perimeter-net/data/splits/*.txt
# Output: 45 train, 18 val, 16 test

# Test dataloader
cd ~/projects/perimiter-net/
uv run python -m scripts.test_dataloader
```

## Important Notes

### Missing Videos

The original DIVA annotations reference 119 videos, but only 79 are available in the public Kitware folder. The `create_splits.py` script automatically filters to only include available videos.

### Annotation Format

- Annotations use YAML format (one list per file)
- Each activity has: `act2` (activity type), `timespan` (frame range), `actors` (person IDs)
- Multi-actor activities are filtered out (only single-person activities used)
- Clips must be ≥16 frames to be valid

### Storage

- **Location**: `/work3/s225224/` (DTU HPC scratch space)
- **Quota**: 100 GB
- **Current usage**: ~38 GB
- **Backup**: Code is in Git, but data is NOT backed up (scratch space)
