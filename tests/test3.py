import cv2
import torch
import torchvision
from torchvision import transforms
import numpy as np
from sort import Sort  # Import SORT tracker

# Define detection classes
class FasterRCNNDetector:
    def __init__(self, device='cpu'):
        # Load a pre-trained Faster R-CNN model
        self.device = device
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def detect(self, frame):
        img_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            detections = self.model(img_tensor)[0]
        return detections

class YOLOv5Detector:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(self.device)
        self.model.eval()

    def detect(self, frame):
        results = self.model(frame)
        detections = results.pandas().xyxy[0]
        return detections

class YOLOv8Detector:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov8s', pretrained=True).to(self.device)
        self.model.eval()

    def detect(self, frame):
        results = self.model(frame)
        detections = results.pandas().xyxy[0]
        return detections

class SSDDetector:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = torchvision.models.detection.ssd300_vgg16(pretrained=True).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def detect(self, frame):
        img_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            detections = self.model(img_tensor)[0]
        return detections

def draw_boxes_with_highlight(frame, detections, click_point=None, threshold=0.5):
    closest_box = None
    min_dist = float('inf')

    for i in range(len(detections['boxes'])):
        score = detections['scores'][i]
        if score > threshold:
            box = detections['boxes'][i].cpu().numpy().astype(int)
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2

            # Draw all detected boxes
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(frame, f'Score: {score:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Highlight the box closest to the clicked point (if any)
            if click_point:
                dist = ((center_x - click_point[0]) ** 2 + (center_y - click_point[1]) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_box = box

    # Highlight the closest box if click_point exists
    if closest_box is not None:
        cv2.rectangle(frame, (closest_box[0], closest_box[1]), (closest_box[2], closest_box[3]), (0, 255, 0), 3)

    return frame

click_point = None

def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)

def main():
    cap = cv2.VideoCapture(0)  # Capture from default camera
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Prompt user to select a detection model
    print("Select the detection model: ")
    print("1: Faster R-CNN")
    print("2: YOLOv5")
    print("3: YOLOv8")
    print("4: SSD")

    model_choice = input("Enter the number of the model: ")

    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if model_choice == '1':
        detector = FasterRCNNDetector(device=device)
    elif model_choice == '2':
        detector = YOLOv5Detector(device=device)
    elif model_choice == '3':
        detector = YOLOv8Detector(device=device)
    elif model_choice == '4':
        detector = SSDDetector(device=device)
    else:
        print("Invalid choice. Exiting.")
        return

    tracker = Sort()  # Initialize SORT tracker

    cv2.namedWindow("Live Video Feed")
    cv2.namedWindow("Detected Objects")

    global click_point

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects on the captured frame
        detections = detector.detect(frame)

        # Convert detections into the format expected by SORT (x1, y1, x2, y2, score)
        if isinstance(detections, dict):
            dets = np.array([[*box.cpu().numpy(), score] for box, score in zip(detections['boxes'], detections['scores']) if score > 0.5])
        else:
            dets = np.array([[row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['confidence']] for _, row in detections.iterrows()])

        # Update the SORT tracker with the detections
        tracked_objects = tracker.update(dets)

        # Draw bounding boxes and highlight the one closest to the clicked point
        detected_frame = draw_boxes_with_highlight(frame.copy(), detections, click_point)

        # Draw tracked boxes
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj.astype(int)
            cv2.rectangle(detected_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(detected_frame, f'Track ID: {int(track_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the detection results in a separate window
        cv2.imshow("Detected Objects", detected_frame)

        # Set the mouse callback function
        cv2.setMouseCallback("Live Video Feed", mouse_callback)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
