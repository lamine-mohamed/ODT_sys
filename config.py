# config.py
import cv2
# Faster R-CNN Configuration
DETECTION_THRESHOLD = 0.5  # Confidence threshold for detections

# Tracking Configuration
TRACKING_TYPE = 'CSRT'  # Tracker type
FRAME_SKIP = 5  # Number of frames to skip before re-detecting

# Pre-detection Configuration
CONTOUR_APPROXIMATION_METHOD = cv2.CHAIN_APPROX_SIMPLE  # Method for contour approximation
CONTOUR_THRESHOLD = 0.02  # Threshold for contour detection accuracy

# Image Processing Configuration
IMAGE_RESIZE_DIMENSIONS = (300, 300)  # Resize dimensions for pre-detection

# Other Configurations
MAX_TRACKING_FAILURES = 3  # Max number of allowed tracking failures before re-detection
