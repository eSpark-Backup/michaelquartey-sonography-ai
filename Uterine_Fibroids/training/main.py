from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

model.train(data="paths.yaml", epochs=100, patience=10, batch=4)
# # metrics = model.val()  # evaluate model performance on the validation set
# # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# # path = model.export(format="onnx")  # export the model to ONNX format~
