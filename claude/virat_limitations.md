# Limitations: Full-Frame Activity Classification

## Current Approach

Our dataloader uses **full surveillance frames** without person-specific bounding boxes.

## The Ambiguity Problem

**What we have:**
- Input: Full video frame (may contain multiple people)
- Label: Activity performed by ONE specific person (e.g., "Person #135 is walking")

**What we DON'T have:**
- Which person in the frame the label refers to
- Bounding box coordinates to crop that person

**Example scenario:**
```
Frame contains:
- Person #135: walking (small, in background) ← This is our label
- Person #47: standing (large, in foreground)
- Person #82: sitting (medium, middle)

Model sees: All three people
Label says: "activity_walking"
```

The model must infer which person the label refers to, introducing training ambiguity.

---

## Data Filtering

To reduce ambiguity, we **filter out multi-actor activities**:
```python
if 'actors' in act_data and len(act_data['actors']) > 1:
    continue  # Skip activities involving multiple people
```

**Impact:**
- ~30% of VIRAT activities are multi-actor and get discarded
- Remaining 70% are single-actor, but other people may still be visible in frame
- Total dataset: 1,748 samples (from 79 videos)

---

## Expected Performance Impact

**Best case:** When labeled person is the most prominent/active in frame
- Model learns to focus on salient motion
- Reasonable accuracy expected

**Worst case:** When labeled person is small/background and others are prominent
- Noisy training signal
- Model may learn incorrect associations

**Estimated accuracy:** 50-70% (rough guess without bounding boxes)

---

## Alternative: Bounding Box Approach

**What it would require:**
1. Load `.geom.yml` files with person bounding boxes
2. Crop each frame to person's bounding box
3. Resize crops to fixed size (224x224)
4. Modify dataloader, handle missing bboxes

**Pros:**
- Clean one-person-one-label mapping
- Higher accuracy expected
- Smaller inputs, faster training

**Cons:**
- Significant code changes
- More complex pipeline
- Some samples lack bounding boxes (as we observed)
- Reduces dataset size further

---

## Why We Chose Full-Frame Approach

**For this course project:**
1. **Time constraint**: Focus on model training, not data engineering
2. **Valid approach**: Full-frame action recognition is used in literature
3. **Realistic**: Deployment systems see full scenes, not pre-cropped people
4. **Baseline first**: Get results, then decide if bounding boxes are needed

---

## Acknowledgment in Report

> "We use full-frame inputs rather than person-specific crops. This introduces ambiguity in multi-person scenes where the activity label refers to one individual but the model observes multiple people. This limitation is acknowledged and could be addressed in future work through integration with person detection and tracking systems (YOLOv8 + DeepSORT)."

---

## Future Work

1. Integrate person detection (YOLOv8) + tracking (DeepSORT)
2. Crop tracked persons from frames
3. Associate detections with VIRAT person IDs
4. Train on clean person-level crops

This would eliminate ambiguity but requires a more complex pipeline - suitable for future iterations beyond the course project scope.