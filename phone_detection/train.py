from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="dataset/yolo_phone/data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    project="runs/phone_detection",
    name="phone_v2",
    patience=15,
    plots=True,
    degrees=10,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=0.5,
    close_mosaic=10
)