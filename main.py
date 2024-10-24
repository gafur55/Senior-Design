from ultralytics import YOLO

model = YOLO("yolov8n.pt") # load the model

results = model.train(data="config.yaml", epochs=5)