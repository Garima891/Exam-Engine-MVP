import cv2
from ultralytics import YOLO
from collections import deque

model_v2 = YOLO(
    "runs/detect/runs/phone_detection/phone_v2-2/weights/best.pt"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
CONFIDENCE = 0.50
confidence_history = deque(maxlen=5)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Webcam frame not received")
        break

    results = model_v2.predict(
        source=frame,
        conf=CONFIDENCE,
        imgsz=416,
        verbose=False
    )

    result = results[0]
    annotated_frame = result.plot()
    phone_count = len(result.boxes)

    if phone_count > 0:
        current_conf = float(result.boxes.conf.max())
        confidence_history.append(current_conf)
        smooth_conf = sum(confidence_history) / len(confidence_history)

        cv2.putText(
            annotated_frame,
            f"Confidence: {smooth_conf:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
    else:
        confidence_history.clear()

    cv2.putText(
        annotated_frame,
        f"Detected Phones: {phone_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Live Phone Detection - YOLO11n V2", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()