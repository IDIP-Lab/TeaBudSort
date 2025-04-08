from ultralytics import YOLO

model = YOLO('yolov12x.yaml')
model.load("best.pt")

model.export(
    format="onnx",
    dynamic=True,
    opset=12,
    imgsz=(640,640)
)