# PerimeterNet

## 🚨 Deep Learning for Activity-Aware Intrusion Detection

**PerimeterNet** is an AI system for critical infrastructure security that detects and classifies human activity at perimeter boundaries (e.g., fences). It aims to reduce false alarms by assessing the context of intrusions.

### 🎯 Objective
Achieve high-accuracy, real-time activity classification to enable context-aware threat alerting.

## Train the 3D activity model (once data is present)

Open `scripts/train_activity_classifier.py` and press “Run Python File” in your IDE. Adjust the constants at the top (data root, epochs, batch size, resize) if needed.

Expected data layout under `--data-root`:
```
videos/*.mp4
annotations/*.activities.yml
splits/train.txt
splits/val.txt
splits/test.txt
```
