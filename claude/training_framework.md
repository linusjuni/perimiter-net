# Project Progress & Design Summary

## 1. Project Goal

- Deep learning system for anomaly detection using the **UCF-Crime dataset** (weakly supervised).
- Binary classification: Normal vs. Anomaly (13 crime types collapsed into single "Anomaly" class).
- Focus on accurate anomaly detection with comprehensive evaluation metrics for imbalanced data.

## 2. UCF-Crime Dataset

- **Implementation:** `src/datasets/ucf.py` (`UCFCrimeDataset`)
- **Structure:** Train/Test splits with video-level labels (weakly supervised)
- **Clip Extraction:** Sliding window with configurable stride
  - **Training:** `stride=clip_len` (16 frames) for non-overlapping clips → ~77K samples
  - **Validation:** `stride=clip_len` for fast evaluation
  - **Design Rationale:** Eliminates massive temporal redundancy (1.2M → 77K samples), reduces epoch time from ~11 hours to ~37 minutes (18x speedup)
- **Labels:** Binary (0=Normal, 1=Anomaly)
- **Preprocessing:** Frames loaded as RGB, resized and cropped via transforms

## 3. Data Pipeline & Transforms

- **Implementation:** `RGBVideoTransform` class in `src/datasets/transforms.py`
- **Train augmentations:** RandomResizedCrop (conservative scale), RandomHorizontalFlip, ColorJitter (brightness/contrast), Kinetics normalization
- **Val/Test augmentations:** Resize, CenterCrop, Kinetics normalization
- **Output format:** `(C, T, H, W)` tensor `(3, 16, 112, 112)` compatible with 3D CNNs
- **Design Choice:** Kinetics normalization for pretrained model compatibility, spatial consistency across entire clip

## 4. Model Architecture

- **Model:** R3D-18 (3D ResNet-18) from torchvision
- **Pretrained:** Kinetics-400 weights
- **Architecture:** Frozen backbone (stem + layer1-4), trainable FC classifier
- **Output:** 2 classes (Normal, Anomaly)
- **Regularization:** Dropout (0.5) in classifier
- **Implementation:** `src/models/r3d.py` with factory function `create_r3d_classifier()`
- **Design Rationale:** Efficient baseline with transfer learning, frozen backbone prevents overfitting on weakly supervised data

## 5. Training Configuration

- **Batch size:** 32 (optimal for V100 32GB)
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-2)
- **Scheduler:** CosineAnnealingLR (smooth decay over epochs)
- **Loss:** Focal Loss (alpha=[0.25, 0.75], gamma=2.0) for class imbalance
- **Mixed Precision:** AMP with GradScaler for ~30% speedup
- **Early Stopping:** Patience=5, metric=AUC (better than accuracy for imbalanced data)
- **Training speed:** ~0.9 sec/batch, ~37 min/epoch, ~6.5 hours for 10 epochs

## 6. Training & Evaluation Loops

- **Training:** `train_epoch()` in `src/utils/training.py`
  - AMP with autocast and gradient scaling
  - Adaptive logging interval (logs ~100 times per epoch)
  - Returns: `(loss, accuracy)` tuple
  - Tracks learning rate in logs
  
- **Evaluation:** `evaluate()` in `src/utils/evaluation.py`
  - Returns `EvaluationMetrics` dataclass with comprehensive metrics
  - AMP support for consistency with training
  - Pre-allocated arrays for memory efficiency
  - Adaptive logging (every 50 batches)

## 7. Evaluation Metrics

- **Dataclass:** `EvaluationMetrics` in `src/utils/evaluation.py`
- **Core metrics:** Loss, Accuracy, AUC-ROC (primary metric for model selection)
- **Per-class metrics:** Precision, Recall, F1 for both Normal and Anomaly classes
- **Binary-specific:** FPR (False Positive Rate), FNR (False Negative Rate)
- **Confusion matrix:** TN, FP, FN, TP with computed properties
- **Design Rationale:** Type-safe dataclass with rich metrics for imbalanced data analysis, AUC-based model selection more robust than accuracy

## 8. Training Utilities

- **Implementation:** `src/utils/training_utils.py`
- **Components:**
  - `save_checkpoint()` / `load_checkpoint()`: Device-aware, saves best_auc + full metrics dict
  - `AverageMeter`: Running average tracker
  - `EarlyStopping`: Monitors validation AUC (mode='max', patience=5)
  - `accuracy()`: Top-k accuracy calculation
- **Logging:** Custom logger for key events (checkpoints, early stopping, metric improvements)

## 9. Loss Function

- **Implementation:** `FocalLoss` in `src/utils/losses.py`
- **Parameters:** alpha=[0.25, 0.75] (lower weight for majority class), gamma=2.0
- **Rationale:** Addresses class imbalance by down-weighting easy examples, focuses learning on hard examples
- **Alternative:** Can switch to weighted CrossEntropyLoss if needed

## 10. Key Design Decisions

- **Stride optimization:** Non-overlapping clips (stride=clip_len) eliminates redundancy, 18x training speedup
- **AUC over accuracy:** Better metric for imbalanced data, used for model selection and early stopping
- **Frozen backbone:** Prevents overfitting on weakly supervised data, faster training
- **Focal Loss:** Handles class imbalance better than standard cross-entropy
- **Dataclass metrics:** Type-safe, self-documenting, easy serialization
- **No config files:** Hyperparameters in code for simplicity and transparency
- **Mixed precision:** ~30% speedup with minimal code changes

## 11. Performance Benchmarks

- **Original (stride=1):** 1.2M samples, ~11 hours/epoch, 77,638 batches
- **Optimized (stride=16):** 77K samples, ~37 min/epoch, 2,451 batches
- **Speedup:** 18x faster training, 16x fewer samples
- **Coverage:** Still sees every frame once per epoch
- **10 epoch training:** ~6.5 hours total (vs. ~110 hours originally)

## 12. Future Enhancements

- **Unfreeze backbone:** Fine-tune after initial convergence for potential accuracy gains
- **Test-time augmentation:** Dense sampling (stride=4) for final evaluation
- **Multi-scale evaluation:** Test multiple resolutions
- **Temporal attention:** Add attention mechanisms for better temporal modeling
- **YOLOv8 + DeepSORT:** Planned for demo integration (real-time tracking)
- **Deployment:** Model optimization (ONNX, TensorRT) for inference speedup
