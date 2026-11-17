# Identity Monitoring Module

## 1. Project Overview

The Identity Monitoring module is a component of the Exam Engine project. It ensures that the person appearing in front of the webcam during an online examination is the registered candidate.

The system combines face detection, facial landmark extraction, face alignment, and facial recognition to perform identity verification.

## 2. Objectives

The module focuses on three main objectives:

1. **Face Detection:** Detect faces in webcam frames and monitor the number of people present.
2. **Facial Landmark Detection:** Extract 478 facial landmarks for each detected face.
3. **Identity Verification:** Verify whether a detected face belongs to a registered candidate using facial embeddings.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Face Detection | MediaPipe Face Landmarker |
| Facial Recognition | OpenCV SFace |
| Image Processing | OpenCV |
| Numerical Processing | NumPy |
| Data Analysis | Pandas |
| Visualization | Matplotlib |
| Evaluation Dataset | Labeled Faces in the Wild (LFW) |

## 4. Models Used

### MediaPipe Face Landmarker
- Detects faces in images and webcam frames.
- Extracts 478 facial landmarks per detected face.
- Provides facial geometry for face alignment.

### OpenCV SFace
- Extracts 128-dimensional facial embeddings.
- Aligns facial images using five facial reference points.
- Supports identity verification through cosine similarity.

Both models are pretrained. No additional model training is performed.

## 5. System Pipeline

### A. Face Detection
1. Capture webcam frames.
2. Convert frames from BGR to RGB.
3. Detect all visible faces using MediaPipe.
4. Extract facial landmarks and bounding boxes.
5. Determine the number of detected faces.

### B. Face Alignment and Embedding Extraction
1. Extract five facial reference points from landmarks.
2. Align the detected face using OpenCV SFace.
3. Generate a 128-dimensional facial embedding.
4. Apply L2 normalization.

### C. Identity Verification
1. Register the candidate using a reference photograph.
2. Generate and store the candidate's reference embedding.
3. Extract embeddings from detected webcam faces.
4. Calculate cosine similarity between reference and live embeddings.
5. Compare similarity against the configured threshold.

**Current identity threshold:** 0.3822 (calibrated using LFW DevTrain evaluation).

### D. Monitoring Status

The system independently monitors:

- NO_FACE_DETECTED
- SINGLE_FACE_DETECTED
- MULTIPLE_FACES_DETECTED

Identity verification separately reports:

- NOT_VERIFIED
- CANDIDATE_IDENTIFIED
- CANDIDATE_NOT_FOUND
- MULTIPLE_MATCHES_FOUND

## 6. Dataset

The module uses the Labeled Faces in the Wild (LFW) dataset for identity verification evaluation.

The dataset includes:
- Genuine identity pairs.
- Impostor identity pairs.
- Development training pairs.
- Development testing pairs.

Selected group photographs and test images are also used to evaluate face detection.

The complete LFW image collection is not included in this repository. Refer to the official dataset source for download instructions.

## 7. Evaluation Results

The SFace verification threshold was calibrated using the LFW DevTrain dataset.

| Metric | DevTest Result |
|---|---:|
| Accuracy | 99.19% |
| Precision | 100% |
| Recall | 98.33% |
| F1-score | 99.16% |
| ROC-AUC | 99.52% |
| FAR | 0% |
| FRR | 1.67% |

- Scorable DevTest pairs: 863
- Threshold: 0.3822
- Evaluation method: Cosine similarity

These results apply to the scorable LFW evaluation pairs and should not be interpreted as guaranteed real-world webcam accuracy.

## 8. Live Identity Monitoring

The live monitoring notebook integrates face detection and identity verification.

It provides:
- Real-time webcam face detection.
- Bounding-box visualization.
- 478 facial landmark visualization.
- Live face counting.
- Candidate identity matching.
- Similarity score display.
- No-face and multiple-face monitoring alerts.

## 9. Project Structure

```text
Identity_monitoring/
│
├── dataset/
├── models/
├── notebooks/
│   ├── 01_environment_setup.ipynb
│   ├── 02_face_detection.ipynb
│   ├── 03_face_alignment_and_embedding.ipynb
│   ├── 04_identity_verification_testing.ipynb
│   └── 05_lfw_csv_analysis.ipynb
│
├── outputs/
├── src/
│   └── live_identity_monitor.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 10. Installation and Execution

1. Clone the Exam Engine repository.
2. Navigate to the `Identity_monitoring` directory.
3. Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

4. Download the complete LFW dataset and place it in the expected dataset directory if running the full evaluation.
5. Ensure both pretrained model files are present in the `models/` directory.
6. Open the notebooks in VS Code or Jupyter.
7. Run the notebooks in sequence.
8. Open `src/live_identity_monitor.ipynb` to test the integrated webcam monitoring system.

## 11. Limitations

- Identity verification depends on lighting, camera quality, face orientation, and occlusion.
- Partially visible faces may not always be detected.
- The current implementation does not provide dedicated liveness or anti-spoofing verification.
- The current LFW-calibrated threshold requires further validation for actual examination environments.
- Live verification is periodically performed to reduce CPU usage.

## 12. Future Scope

- Multiple-student registration and identification.
- Persistent student embedding storage.
- Database integration for monitoring events.
- Improved handling of partially occluded faces.
- Liveness detection and anti-spoofing.
- Integration with the complete Exam Engine.

## 13. Acknowledgements

- Google MediaPipe
- OpenCV Zoo – SFace
- Labeled Faces in the Wild (LFW) Dataset

Model licenses and source references should be retained with the distributed model files.
