# AI Phone Detection Model for Proctored Exam Engine

## Overview
This module detects mobile phones in images/video frames for a proctored-exam prototype. It uses Ultralytics YOLO11n and returns phone bounding boxes with confidence scores.

> **Scope:** This deliverable covers the phone-detection AI model only. Backend, UI, exam workflow, and system integration are outside this module.

## Model Versions

### V1 — Baseline phone detector
- Base model: YOLO11n
- Training epochs: 30
- Dataset: Original phone-detection dataset
- Training image size: 416
- Validation metrics (recorded):
  - Precision: 92.4%
  - Recall: 75.4%
  - mAP@50: 78.2%
  - mAP@50–95: 66.3%

### V2 — Half-phone and negative-image candidate
- Base model: YOLO11n
- Training epochs: 50
- Dataset: Original dataset + 57 annotated half-phone images + 66 negative images
- Training image size: 640
- Inference image size used in webcam test: 416
- Validation metrics (recorded):
  - Precision: 86.5%
  - Recall: 75.4%
  - mAP@50: 78.5%
  - mAP@50–95: 56.2%

V2 is the final **candidate** for testing partial phones; metrics do not show that it is better on every measure. V1 and V2 were trained/evaluated with different data/configurations, so compare results cautiously.

## Files
- `models/phone_v1_best.pt` — baseline model weights (copy from V1 run's `weights/best.pt`)
- `models/phone_v2_best.pt` — V2 model weights (copy from V2 run's `weights/best.pt`)
- `data.yaml` — dataset configuration
- `webcam_test.py` — live webcam inference script
- `notebooks/` — training notebooks (if retained)
- `Evaluation_Report.pdf` — metrics and test limitations

## Requirements
- Python (the environment used for training/testing)
- Ultralytics
- OpenCV

Install dependencies:
```bash
pip install ultralytics opencv-python
```

## Run webcam inference
1. Copy the selected model weights to `models/phone_v2_best.pt` (or update the path in the script).
2. Run:
```bash
python webcam_test.py
```
3. Press `q` to close the webcam window.

## Important configuration
- Webcam confidence threshold: `0.50`
- Inference image size: `416`
- Webcam index: `0` (change if required)

## Limitations
- A confidence score is a model score, not a guarantee of correctness or overall accuracy.
- Validation metrics are not the same as real-world webcam performance.
- Partial/occluded phone detection is not guaranteed.
- Face/glasses and other objects may cause false positives.
- A phone that is fully hidden cannot be detected from the camera frame.
- Current dataset/validation set is relatively small; further testing on diverse exam-room conditions is recommended.
- Treat detections as review cues, not proof of cheating.

## Suggested final checks
- Test phone visible, partially visible phone, phone at different angles/distances, face-only, glasses, hands, laptop/book, and empty desk.
- Record true detections, missed phones, and false detections with sample counts.
