import cv2
from detection import FasterRCNNDetector  
from tracker import ObjectTracker 
from utils import mouse_click, resize_image, detect_contour_around_click, extract_roi, adjust_bbox_to_original_scale
from config import FRAME_SKIP, IMAGE_RESIZE_DIMENSIONS, MAX_TRACKING_FAILURES

def main():
    cap = cv2.VideoCapture(0)
    detector = FasterRCNNDetector()
    tracker = ObjectTracker()

    click_info = {'click': None}
    object_detected = False  # Flag to check if the object has been detected
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_click, click_info)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to capture frame. Exiting...")
            break

        if click_info['click'] is not None and not object_detected:
            click_x, click_y = click_info['click']

            # Step 1: Pre-detection process
            contour_bbox = detect_contour_around_click(frame, (click_x, click_y), config={'CONTOUR_APPROXIMATION_METHOD': cv2.CHAIN_APPROX_SIMPLE})
            if contour_bbox is not None:
                roi = extract_roi(frame, contour_bbox)
                resized_roi = resize_image(roi, IMAGE_RESIZE_DIMENSIONS)
                
                # Step 2: Accurate detection with Faster R-CNN
                boxes, scores, labels = detector.detect(resized_roi)
                
                # Find the box that contains the original click location
                selected_box = None
                for box in boxes:
                    # Adjust box to original image scale
                    adjusted_box = adjust_bbox_to_original_scale(box, contour_bbox, IMAGE_RESIZE_DIMENSIONS, frame.shape)
                    
                    if click_x >= adjusted_box[0] and click_x <= adjusted_box[2] and click_y >= adjusted_box[1] and click_y <= adjusted_box[3]:
                        selected_box = adjusted_box
                        break
                
                if selected_box is not None:
                    tracker.initialize(frame, selected_box)
                    object_detected = True  # Set the flag to true after detection
                    click_info['click'] = None  # Reset click info
                    print("Tracker initialized successfully with box:", selected_box)
                else:
                    print("No object detected around the clicked location.")

        if object_detected and frame_count % FRAME_SKIP == 0:
            # Update tracking
            bbox = tracker.update(frame)
            if bbox is not None:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                tracker.failures += 1
                if tracker.failures >= MAX_TRACKING_FAILURES:
                    print("Re-detection required due to multiple tracking failures.")
                    object_detected = False  # Reset the flag to indicate detection is needed again

        cv2.imshow('Frame', frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
