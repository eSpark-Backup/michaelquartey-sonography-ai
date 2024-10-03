from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

model.train(data="data.yaml", epochs=100, patience=10, batch=4)