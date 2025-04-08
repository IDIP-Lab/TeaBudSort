from ultralytics import YOLO

model = YOLO('best.pt')
model.val(data='tea.yaml', save_json=True)