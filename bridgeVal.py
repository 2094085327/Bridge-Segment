from ultralytics import YOLO

# Load a model
model = YOLO('C:/Users/86188/Desktop/Bridge/ultralytics-main/runs/segment/train13/weights/best.pt')  # pretrained YOLOv8n model

source = 'C:/Users/86188/Desktop/Bridge/ultralytics-main/ultralytics/assets'

# Run batched inference on a list of images
results = model(source, stream=True)  # return a list of Results objects

# Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bbox outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Class probabilities for classification outputs

model.predict(source, save=True, imgsz=640, conf=0.7)