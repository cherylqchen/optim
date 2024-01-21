from ultralytics import YOLO

model = YOLO("/Users/user/Downloads/Optim/colourDetectorv2.pt")
model.predict(source="0",show=True,conf=0.5)