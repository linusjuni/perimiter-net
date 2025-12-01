# UCF-Crime Dataset: Comprehensive Description

## Overview

The UCF-Crime dataset is a large-scale benchmark dataset for real-world anomaly detection in surveillance videos. It was introduced by Sultani, Chen, and Shah in their 2018 CVPR paper "Real-world Anomaly Detection in Surveillance Videos" and has become the standard benchmark for weakly supervised video anomaly detection research.

## Dataset Purpose and Design Philosophy

UCF-Crime was specifically designed to address the limitations of previous anomaly detection datasets, which typically:

- Featured acted or staged scenarios rather than real events
- Contained only a few anomaly types
- Were recorded in controlled environments
- Lacked realistic diversity in scenes, lighting, and camera angles

The dataset focuses on **weakly supervised learning**, where only video-level labels are provided (indicating whether a video contains an anomaly) rather than expensive frame-level or temporal annotations. This design choice reflects real-world constraints where obtaining detailed temporal annotations for surveillance footage is prohibitively expensive and time-consuming.

## Dataset Statistics

- **Total videos:** 1,900 long, untrimmed surveillance videos
- **Total duration:** Approximately 128 hours of footage
- **Training set:** 1,610 videos (800 normal + 810 anomalous)
- **Test set:** 290 videos (150 normal + 140 anomalous)
- **Video characteristics:** Variable length (ranging from tens of seconds to several minutes), captured from real-world CCTV cameras
- **Resolution:** Variable, reflecting real surveillance camera quality
- **Sources:** Real-world surveillance footage from YouTube and LiveLeak

## Class Distribution

The dataset contains **14 total classes**:

- **1 Normal class:** Regular daily activities captured by surveillance cameras
- **13 Anomaly classes:** Criminal or abnormal activities

### Severe Class Imbalance

The dataset exhibits significant class imbalance, which is realistic for surveillance scenarios:

- **Normal videos:** ~1,610 videos (~85% of dataset)
- **Anomaly videos:** ~290 videos (~15% of dataset)
- Within anomaly classes, distribution varies significantly
- Normal frames outnumber anomalous frames by approximately 20:1 to 40:1 ratio

This imbalance is intentional and reflects real-world surveillance where normal activities vastly outnumber anomalous events.

## The 13 Anomaly Categories

Each anomaly category was selected for its impact on public safety:

### 1. **Abuse**

Physical or verbal abuse between individuals. Includes domestic violence, elder abuse, and assault-like behaviors involving vulnerable individuals.

### 2. **Arrest**

Police officers arresting individuals. Includes handcuffing, detainment, and police intervention scenarios.
**Note:** This class is known to be challenging due to label ambiguity—distinguishing between normal police presence and actual arrests can be difficult.

### 3. **Arson**

Intentional fire-setting. Includes individuals deliberately starting fires in buildings, vehicles, or outdoor areas.

### 4. **Assault**

Physical attacks or violent confrontations between individuals. Distinguished from "Fighting" by being more one-sided or severe.

### 5. **Burglary**

Breaking and entering into buildings or vehicles with intent to steal. Includes forced entry, lock picking, and property intrusion.
**Note:** This class is challenging because the anomalous action (breaking in) may appear similar to normal entry in some cases.

### 6. **Explosion**

Explosions or detonations. Includes both indoor and outdoor explosion events.

### 7. **Fighting**

Physical altercations between two or more people. Includes street fights, bar fights, and aggressive physical confrontations.

### 8. **Road Accidents**

Vehicle accidents including car crashes, collisions, and traffic incidents. Captures various accident scenarios from different angles.

### 9. **Robbery**

Theft involving force or threat of force. Includes armed robbery, mugging, and violent theft scenarios. Distinct from "Stealing" by the presence of confrontation or weapons.

### 10. **Shooting**

Firearms being discharged. Includes shootings in various contexts (streets, buildings, etc.).

### 11. **Shoplifting**

Theft in retail environments. Shows individuals stealing items from stores without confrontation.

### 12. **Stealing**

Non-confrontational theft in non-retail settings. Includes pickpocketing, bag snatching, and opportunistic theft.

### 13. **Vandalism**

Deliberate destruction or damage to public or private property. Includes graffiti, property defacement, and intentional damage.

### Normal Videos

Normal class videos contain typical daily surveillance footage including:

- People walking, standing, or sitting in normal contexts
- Regular traffic flow
- Normal shopping behavior
- Everyday activities in public spaces
- Indoor and outdoor scenes
- Day and nighttime footage
- Various weather conditions

## Video Structure and Characteristics

### Untrimmed Nature

Videos are **long and untrimmed**, meaning:

- Anomalies may occur at any point in the video
- Significant portions of anomalous videos contain normal activity
- Temporal localization of anomalies is required during testing
- Average video contains both normal frames and anomalous segments

### Realism Factors

The dataset captures realistic surveillance challenges:

- **Variable lighting:** Day, night, artificial lighting, shadows
- **Camera quality:** Ranges from low to high resolution
- **Viewing angles:** Overhead, street-level, indoor, outdoor perspectives
- **Occlusions:** People and objects may be partially obscured
- **Background clutter:** Busy urban environments, crowds, traffic
- **Camera motion:** Some cameras have slight movement or shake
- **Distance variation:** Subjects appear at various distances from camera

## Annotation Format

### Training Set

- **Video-level labels only:** Each video is labeled as "Normal" (0) or "Anomaly" (1)
- **No temporal annotations:** The exact timing of anomalies is NOT provided
- **No spatial annotations:** No bounding boxes or spatial localization
- **Purpose:** Supports weakly supervised learning via Multiple Instance Learning (MIL)

### Test Set

- **Frame-level annotations:** Exact frames containing anomalies are marked
- **Temporal boundaries:** Start and end frames of anomalous events are provided
- **Purpose:** Enables precise evaluation of temporal localization performance

## Pre-extracted Frame Format

Many implementations of UCF-Crime work with pre-extracted frames rather than raw videos. The typical frame extraction strategy:

### Frame Extraction Convention

- **Sampling rate:** Every 10th frame from original videos is extracted
- **Naming convention:** `{ClassName}{VideoNumber}_x264_{FrameNumber}.png`
  - Example: `Fighting002_x264_1000.png`
  - Example: `Normal_Videos331_x264_71840.png`
- **Format:** PNG images
- **Organization:** Frames organized into class subdirectories

### Example Frame Counts (Training Set)

- **Normal_Videos:** ~948,000 frames
- **Fighting:** ~24,684 frames
- **Robbery:** ~41,493 frames
- **Stealing:** ~44,802 frames

This pre-extraction reduces I/O overhead and enables faster experimentation.

## Intended Use Cases

### Primary Task 1: Binary Anomaly Detection

**Goal:** Classify whether a video contains any anomaly (regardless of type)

- **Classes:** 2 (Normal vs Anomaly)
- **Metric:** Frame-level AUC (Area Under ROC Curve)
- **Application:** Real-time alerting systems for security personnel
- **Challenge:** Severe class imbalance (20:1 normal to anomaly ratio)

### Primary Task 2: Multi-Class Anomaly Recognition

**Goal:** Identify the specific type of anomaly occurring

- **Classes:** 14 (1 Normal + 13 Anomaly types)
- **Metrics:** Accuracy, per-class precision/recall, mean Average Precision
- **Application:** Forensic analysis, detailed incident reporting
- **Challenge:** Some classes have very few samples and high intra-class variation

### Research Applications

- Weakly supervised learning algorithm development
- Multiple Instance Learning (MIL) frameworks
- Temporal action localization
- Video understanding in surveillance contexts
- Zero-shot or few-shot anomaly detection

## Known Challenges and Limitations

### 1. Label Ambiguity

- **Arrest class:** Difficulty distinguishing normal police presence from actual arrests
- **Burglary class:** Breaking in can appear similar to legitimate entry
- **Fighting vs Assault:** Subjective boundaries between categories
- **Context dependency:** Same action might be normal or anomalous depending on context

### 2. Severe Class Imbalance

- Normal frames outnumber anomalous frames by 20:1 to 40:1
- Within anomalies, some classes are much rarer than others
- Requires specialized techniques: focal loss, weighted sampling, class balancing
- Standard cross-entropy loss performs poorly without modification

### 3. Intra-Class Variation

- Wide variation in how same anomaly appears across different videos
- Different camera angles, distances, lighting conditions
- Cultural and contextual differences in activities
- Variable duration of anomalous events (seconds to minutes)

### 4. Temporal Localization Difficulty

- Anomalies can be brief (a few seconds) within long videos (several minutes)
- Transition between normal and anomalous behavior can be gradual
- Multiple anomalies may occur in a single video
- Background activity continues during anomalies

### 5. Quality Variations

- Real-world surveillance footage has variable quality
- Some videos are low resolution or poorly lit
- Motion blur during rapid actions
- Compression artifacts in some videos

### 6. Evaluation Challenges

- Frame-level evaluation requires precise temporal boundaries
- Clip-based predictions must be mapped to frame-level scores
- Different evaluation protocols across research papers make comparison difficult

## Benchmark Performance

### Historical Performance (Binary Detection, Frame-level AUC)

**Early Methods (2018-2020):**

- Sultani et al. (2018) - MIL baseline: 75.41% AUC
- AR-Net (2020): 82.12% AUC
- RTFM (2021): 84.30% AUC

**Recent State-of-the-Art (2023-2025):**

- MGFN (2023): 86.87% AUC
- DSANet (2024): 89.44% AUC
- Multimodal methods: 85-87% AUC
- Ensemble methods: 80-88% AUC

**Performance Notes:**

- 85-90% AUC is considered excellent performance
- 80-85% AUC is competitive
- <80% AUC indicates need for improvement
- Perfect AUC (100%) is unrealistic due to label ambiguity

### Typical Baseline Performance

- Random classifier: ~50% AUC
- Simple frame difference methods: 60-65% AUC
- Pretrained 3D CNN (R3D-18) baseline: 78-83% AUC
- Two-stream networks: 83-87% AUC

## Standard Evaluation Protocol

### Metrics

1. **Frame-level AUC:** Primary metric for binary detection
   - Measures ability to rank anomalous frames higher than normal frames
   - Threshold-independent
   - Handles class imbalance well

2. **Average Precision (AP):** Alternative metric
   - Summarizes precision-recall curve
   - Used in some recent papers

3. **mAP at IoU thresholds:** For temporal localization
   - Measures precision of temporal boundaries
   - IoU thresholds typically 0.1, 0.3, 0.5

4. **Per-class metrics:** For multi-class evaluation
   - Accuracy, precision, recall per anomaly type
   - Identifies which anomalies are most difficult

### Evaluation Procedure

1. Train model on training set (video-level labels only)
2. Predict anomaly scores for all frames in test videos
3. Compare predictions to ground truth frame-level annotations
4. Compute AUC using frame-level predictions and labels
5. Optional: Aggregate clip predictions to video-level for video-level AUC

## Why UCF-Crime is Valuable

### 1. Realism

Unlike acted or synthetic datasets, UCF-Crime contains real surveillance footage of actual events, making it highly relevant for real-world deployment.

### 2. Scale

With 1,900 videos and 128 hours of footage, it provides sufficient data for training deep learning models while remaining manageable.

### 3. Weakly Supervised Setting

Video-level labels reflect realistic annotation constraints, encouraging development of methods that don't require expensive frame-level labeling.

### 4. Diversity

13 different anomaly types spanning various criminal and abnormal activities provide comprehensive coverage of security concerns.

### 5. Benchmark Status

As the standard benchmark since 2018, UCF-Crime enables direct comparison with extensive prior work, facilitating research progress tracking.

### 6. Practical Relevance

The dataset addresses real security concerns, making research directly applicable to surveillance systems, smart cities, and public safety applications.

## Comparison to Other Datasets

### UCF-Crime vs UCSD Ped1/Ped2

- **UCF-Crime:** Real anomalies, diverse types, weakly supervised
- **UCSD:** Pedestrian-focused, simpler anomalies, smaller scale
- **Advantage:** UCF-Crime is more realistic and challenging

### UCF-Crime vs ShanghaiTech

- **UCF-Crime:** Weakly supervised, criminal activities, outdoor & indoor
- **ShanghaiTech:** Pixel-level annotations, campus setting, ~13 hours
- **Advantage:** UCF-Crime better reflects real surveillance constraints

### UCF-Crime vs XD-Violence

- **UCF-Crime:** 13 anomaly types, video-level labels, single modality
- **XD-Violence:** Violence-focused, audio+visual, larger scale
- **Advantage:** XD-Violence has multimodal data; UCF-Crime has broader anomalies

## Typical Experimental Setup

### Data Split

- **Training:** Use provided training set (800 normal + 810 anomaly videos)
- **Validation:** Hold out 10-20% of training set for hyperparameter tuning
- **Testing:** Use provided test set (150 normal + 140 anomaly videos)

### Common Preprocessing

- Extract frames at fixed intervals (e.g., every 10th frame)
- Sample temporal clips of 16-32 frames
- Resize frames to 112×112 or 224×224 pixels
- Normalize with ImageNet statistics

### Training Strategies

- **Multiple Instance Learning (MIL):** Treat each video as a bag of clips
- **Focal Loss:** Address class imbalance
- **Pretrained Models:** Use Kinetics-400 pretrained 3D CNNs
- **Two-Stream Networks:** Combine RGB and optical flow

### Augmentation Considerations

- Horizontal flip (safe for surveillance)
- Random crops (simulate different viewpoints)
- Color jitter (lighting variations)
- Avoid: vertical flip, rotation (changes semantics)

## Common Pitfalls and Best Practices

### Pitfalls to Avoid

1. **Data leakage:** Ensure no overlap between train/test videos
2. **Ignoring imbalance:** Standard cross-entropy performs poorly
3. **Frame-level training:** Remember only video-level labels are available
4. **Overfitting to normal class:** Model may ignore anomalies
5. **Wrong evaluation:** Must use frame-level AUC, not accuracy

### Best Practices

1. **Use focal loss or weighted sampling** for class imbalance
2. **Pretrain on Kinetics-400** for better initialization
3. **Sample multiple clips per video** during training
4. **Validate with AUC** as primary metric
5. **Analyze per-class performance** to identify weaknesses
6. **Report both clip-level and video-level** AUC

## Technical Specifications Summary

| Property | Value |
|----------|-------|
| Total Videos | 1,900 |
| Total Duration | ~128 hours |
| Number of Classes | 14 (1 normal + 13 anomalies) |
| Training Videos | 1,610 |
| Test Videos | 290 |
| Annotation Type (Train) | Video-level |
| Annotation Type (Test) | Frame-level |
| Video Format | MP4 (original), PNG (extracted frames) |
| Typical Frame Rate | 30 fps (original) |
| Frame Extraction Rate | Every 10th frame (common) |
| Class Imbalance | ~20:1 (normal:anomaly) |
| Primary Metric | Frame-level AUC |
| Learning Paradigm | Weakly Supervised |
| Source | Real surveillance footage |
| Release Year | 2018 |

## Citation

If using the UCF-Crime dataset, cite the original paper:

```bibtex
@inproceedings{sultani2018real,
  title={Real-world anomaly detection in surveillance videos},
  author={Sultani, Waqas and Chen, Chen and Shah, Mubarak},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={6479--6488},
  year={2018}
}
```

## Conclusion

The UCF-Crime dataset represents a significant contribution to video anomaly detection research. Its focus on real-world surveillance footage, weakly supervised learning, and diverse anomaly types makes it an ideal benchmark for developing practical anomaly detection systems. While it presents significant challenges—particularly severe class imbalance and label ambiguity—these challenges reflect real-world constraints and drive development of robust, deployable solutions.

For researchers and practitioners working on surveillance video analysis, UCF-Crime provides a realistic testbed that bridges the gap between academic research and practical security applications. Its continued use as a benchmark ensures that advances in the field are measured against a consistent, challenging, and practically relevant standard.
