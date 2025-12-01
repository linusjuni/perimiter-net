# Deep Learning for Perimeter Security: Activity-Aware Intrusion Detection

**Authors:**
* Theodor Dornonville de la Cour
* Linus Juni

## Motivation

Critical infrastructure requires robust perimeter security. Many current systems rely on manual monitoring or simple motion detection, causing high false alarm rates.
We propose a Deep Learning system that detects boundary crossings and classifies objects and activities (walking, running, carrying objects) to reduce false positives and provide context-aware threat assessment.

## Background

We use YOLOv8 for person detection, define virtual perimeters for boundary monitoring, and train a 3D CNN / Two Stream Fusion activity classifier on the VIRAT surveillance dataset [1]. DeepSORT [2] provides tracking across frames. Optionally, we integrate acoustic event detection for multi-modal analysis.

## Milestones

* **Weeks 1-2:** Setup VIRAT dataset, implement perimeter detection with YOLOv8, extract activity clips
* **Weeks 3-4:** Train activity classifier (6-8 classes), integrate tracking, evaluate performance
* **Week 5:** Integrate components, design alert system, add audio detection if time permits
* **Week 6:** Final evaluation, report writing, reproducible code

## References

[1] Oh, S., et al. (2011). A large-scale benchmark dataset for event recognition in surveillance video. CVPR 2011.
[2] Wojke, N., et al. (2017). Simple online and realtime tracking with a deep association metric. ICIP 2017.