import cv2
import torch
import torchvision
from torchvision import transforms
import numpy as np

class FasterRCNNDetector:
    def __init__(self):
        # Load a pre-trained Faster R-CNN model from torchvision
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def detect(self, frame):
        # Convert frame to a tensor
        img_tensor = self.transform(frame).unsqueeze(0)

        # Perform detection
        with torch.no_grad():
            detections = self.model(img_tensor)[0]

        return detections

def draw_boxes_with_highlight(frame, detections, click_point, threshold=0.5):
    closest_box = None
    min_dist = float('inf')

    for i in range(len(detections['boxes'])):
        score = detections['scores'][i]
        if score > threshold:
            box = detections['boxes'][i].numpy().astype(int)
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            dist = ((center_x - click_point[0]) ** 2 + (center_y - click_point[1]) ** 2) ** 0.5

            # Draw all detected boxes
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(frame, f'Score: {score:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Update closest box based on distance only
            if dist < min_dist:
                min_dist = dist
                closest_box = box

    # Highlight the closest box
    if closest_box is not None:
        cv2.rectangle(frame, (closest_box[0], closest_box[1]), (closest_box[2], closest_box[3]), (0, 255, 0), 3)

    return frame, closest_box

class ObjectTracker:
    def __init__(self):
        self.tracker = cv2.TrackerCSRT_create()
        self.initialized = False

    def initialize(self, frame, bbox):
        x, y, x1, y1 = bbox
        bbox = (x, y, abs(x-x1), abs(y+y1))
        self.tracker.init(frame, bbox)
        self.initialized = True

    def update(self, frame):
        if not self.initialized:
            return False, None
        success, bbox = self.tracker.update(frame)
        return success, bbox

click_point = None

def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)

def main():
    cap = cv2.VideoCapture(0)  # Capture from default camera

    detector = FasterRCNNDetector()
    tracker = ObjectTracker()

    cv2.namedWindow("Live Video Feed")
    cv2.namedWindow("Detected Objects")

    global click_point

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the live video feed
        cv2.imshow("Live Video Feed", frame)

        if click_point:
            # Detect objects on the captured frame
            detections = detector.detect(frame)

            # Draw all bounding boxes and highlight the one closest to the clicked point
            detected_frame, tracked_bbox = draw_boxes_with_highlight(frame.copy(), detections, click_point)

            # Initialize the tracker with the selected bounding box
            if tracked_bbox is not None:
                tracker.initialize(frame, tracked_bbox)

            # Show the detection results in a separate window
            cv2.imshow("Detected Objects", detected_frame)

            # Reset click_point to avoid repeated initialization
            click_point = None

        # Update the tracker and draw tracking results
        if tracker.initialized:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the live video feed with tracking results
        cv2.imshow("Live Video Feed", frame)

        # Set the mouse callback function
        cv2.setMouseCallback("Live Video Feed", mouse_callback)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
