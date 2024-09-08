import cv2
import torch
import time
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
from ultralytics import YOLO

# Initialize YOLOv5 and YOLOv8 models
yolo5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
yolo8_model = YOLO('yolov8n.pt')

# Initialize Faster R-CNN model with ResNet50 backbone
faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn.eval()

# Load test image
image_path = 'test_image.jpg'
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Metrics storage
metrics = {
    'Speed': {},
    'Robustness': {},
    'Computational Efficiency': {}
}

# Define a function to time the execution
def time_execution(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    return result, elapsed_time

# Contour Detection
def contour_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    return contour_img

# YOLOv5 Detection
def yolo5_detection(image):
    results = yolo5_model(image)
    return np.squeeze(results.render())

# YOLOv8 Detection
def yolo8_detection(image):
    results = yolo8_model(image)
    return np.squeeze(results[0].plot())

# Faster R-CNN Detection
def faster_rcnn_detection(image):
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        predictions = faster_rcnn(image_tensor)[0]
    for box in predictions['boxes']:
        box = box.int().tolist()
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return image

# Calculate robustness (a proxy measure based on the number of detections)
def calculate_robustness(detections):
     # Placeholder for robustness calculation 
    return len(detections) 

# Calculate computational efficiency (inverse of the time taken)
def calculate_efficiency(time_taken):
    return 1 / time_taken if time_taken > 0 else 0

# Perform the detections and collect metrics
contour_img, contour_time = time_execution(contour_detection, img)
yolo5_img, yolo5_time = time_execution(yolo5_detection, img_rgb)
yolo8_img, yolo8_time = time_execution(yolo8_detection, img_rgb)
faster_rcnn_img, faster_rcnn_time = time_execution(faster_rcnn_detection, img_rgb)


metrics['Speed']['Contour Detection'] = contour_time
metrics['Speed']['YOLOv5'] = yolo5_time
metrics['Speed']['YOLOv8'] = yolo8_time
metrics['Speed']['Faster R-CNN'] = faster_rcnn_time

metrics['Robustness']['Contour Detection'] = calculate_robustness(contour_img)
metrics['Robustness']['YOLOv5'] = calculate_robustness(yolo5_img)
metrics['Robustness']['YOLOv8'] = calculate_robustness(yolo8_img)
metrics['Robustness']['Faster R-CNN'] = calculate_robustness(faster_rcnn_img)

metrics['Computational Efficiency']['Contour Detection'] = calculate_efficiency(contour_time)
metrics['Computational Efficiency']['YOLOv5'] = calculate_efficiency(yolo5_time)
metrics['Computational Efficiency']['YOLOv8'] = calculate_efficiency(yolo8_time)
metrics['Computational Efficiency']['Faster R-CNN'] = calculate_efficiency(faster_rcnn_time)

# Display metrics
for metric, values in metrics.items():
    print(f"\n{metric}:")
    for method, value in values.items():
        print(f"  {method}: {value:.2f}")

# Resize the images for consistent display size
contour_img_resized = cv2.resize(contour_img, (640, 320))
yolo5_img_resized = cv2.resize(yolo5_img, (640, 320))
yolo8_img_resized = cv2.resize(yolo8_img, (640, 320))
faster_rcnn_img_resized = cv2.resize(cv2.cvtColor(faster_rcnn_img, cv2.COLOR_RGB2BGR), (640, 320))

# Stack images horizontally for a grid layout
top_row = np.hstack((contour_img_resized, yolo5_img_resized))
bottom_row = np.hstack((yolo8_img_resized, faster_rcnn_img_resized))
final_img = np.vstack((top_row, bottom_row))

# Display the images in one window
cv2.imshow('Object Detection Comparison', final_img)

# Wait until any key is pressed to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
