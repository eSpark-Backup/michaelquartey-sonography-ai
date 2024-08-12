import cv2
from PIL import Image

from ultralytics import YOLO

model = YOLO("best.onnx")

results = model.predict(source="test_images", save=True)