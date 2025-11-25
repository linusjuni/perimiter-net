# Project Progress & Design Summary

## 1. Project Goal
- Deep learning system for perimeter security using the VIRAT dataset.
- Detect and classify human activities (walking, running, standing, carrying, gesturing, entering, exiting) in surveillance video.
- Reduce false alarms and provide context-aware threat assessment.

## 2. Dataset Setup
- VIRAT Ground Camera Dataset (79 videos, 1,748 samples after filtering).
- Train/val/test splits (45/18/16 videos).
- 7 activity classes.
- Each sample is a 16-frame video clip, full-frame (not cropped to person).
- Multi-actor activities (~30% of data) excluded for label clarity.
- Data stored on DTU HPC at `/work3/s225224/perimeter-net/`.
- **Design Choice:** Accept full-frame classification as a limitation (label ambiguity), document in report, and propose person detection/tracking as future work.

## 3. Data Pipeline & Transforms
- Implemented `RGBVideoTransform` class in `src/datasets/transforms.py`.
- Train augmentations: RandomResizedCrop (conservative scale), RandomHorizontalFlip, ColorJitter (brightness/contrast), normalization (Kinetics/ImageNet stats).
- Val/Test augmentations: Resize, CenterCrop, normalization.
- Transforms convert `(T, H, W, C)` numpy array to `(C, T, H, W)` tensor for model compatibility.
- Transforms are applied in `VIRATDataset` via the `transform` argument.
- **Design Choice:** Apply transforms to the entire clip for spatial consistency. Use Kinetics normalization stats for compatibility with pretrained models.

## 4. Testing & Validation
- Quick test script validated that dataset and transforms produce correct shapes and value ranges.
- Training/validation clips are `(3, 16, 112, 112)` tensors, normalized as expected.

## 5. Training Utilities
- `src/utils/training_utils.py`:
	- `save_checkpoint` and `load_checkpoint` (device-aware, with logging).
	- `AverageMeter` for tracking running averages.
	- `EarlyStopping` for stopping training when validation metric stalls (with logging).
	- `accuracy` for top-k accuracy calculation.
- Logging using custom logger for key events (saving/loading checkpoints, early stopping, etc.).
- **Design Choice:** Simple, effective, PyTorch-style utilities. Logging only where it adds value.

## 6. Training & Evaluation Loops
- Functions: `train_epoch` and `evaluate` (in utils).
- Clean, reusable loops for one epoch of training/evaluation.
- Logging at key points (every 10 batches, epoch summary).
- No progress bars (per user preference).
- Metrics tracked using `AverageMeter`.

## 7. Model Architecture Planning
- R3D-18 (3D ResNet) from torchvision.
- Use Kinetics-400 pretrained weights.
- Freeze backbone (stem + layer1-4), only train final classifier (FC layer).
- Replace FC layer to output 7 classes.
- Device-aware (move to GPU).
- Logging for model setup, freezing, and parameter counts.
- Planned: `R3DClassifier` class in `src/models/r3d.py` with a factory function for easy instantiation.
- **Design Choice:** Start with R3D-18 for baseline, simple fine-tuning (only classifier). Modular design for easy extension (e.g., unfreezing backbone, switching to other architectures).

## 8. Other Design Notes
- No YAML config files: All hyperparameters defined in code for simplicity.
- Batch size: To be determined based on GPU memory (Tesla V100-PCIe 32GB).
- Optimizer: AdamW (modern, decoupled weight decay), learning rate `1e-4`, simple weight decay.
- Scheduler: CosineAnnealingLR (smooth decay).
- Class imbalance: Option to use weighted loss if needed.
- Logging: Prefer logging statements over progress bars.

## 9. Future Work / Placeholders
- Two-Stream Network: Placeholders for spatial and temporal transforms (for future extension).
- YOLOv8 + DeepSORT: Planned for demo integration, not full deployment.
- Evaluation Metrics: Confusion matrix, per-class accuracy (to be added).

## 10. Next Steps
- Implement and test the `R3DClassifier` class.
- Write the main training script (`scripts/train_r3d.py`) using all utilities and loops.
- Add evaluation metrics and reporting.
- (Optional) Integrate tracking and two-stream models.
