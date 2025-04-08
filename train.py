from ultralytics import YOLO

# load a model
model = YOLO('yolov12x.yaml')
model.load("yolov12x.pt")

# Train the model
results = model.train(
  data='tea.yaml',
  epochs=300,
  batch=16,
  imgsz=640,
  scale=0.9,  # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.2,  # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.6,  # S:0.15; M:0.4; L:0.5; X:0.6
  device="cuda:0",
)
