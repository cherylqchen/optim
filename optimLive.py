from ultralytics import YOLO

model = YOLO("/Users/user/Downloads/Optim/colourDetector.pt")
model.predict(source="0",show=True,conf=0.5)