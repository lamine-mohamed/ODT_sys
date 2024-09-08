# tracker.py

import cv2
from utils import convert_to_xywh

class ObjectTracker:
    def __init__(self, tracker_type='CSRT'):
        self.tracker = self._create_tracker(tracker_type)
        self.failures = 0
        self.is_tracking = False


    def _create_tracker(self, tracker_type):
        """Initialize the tracker based on the given type."""
        if tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        else:
            raise ValueError("Unsupported tracker type: {}".format(tracker_type))

    def initialize(self, frame, bbox):
        if frame is None or bbox is None:
            print("Initialization failed: Frame or Bounding Box is None.")
            return False

        if bbox[2] <= 0 or bbox[3] <= 0:
            print("Initialization failed: Invalid bounding box dimensions.")
            return False

        success = self.tracker.init(frame, convert_to_xywh(bbox))
        if success:
            self.is_tracking = True
        return success
    
    def update(self, frame):
        if not self.is_tracking or frame is None:
            print("Cannot update tracker: Not tracking or frame is None.")
            return None

        success, bbox = self.tracker.update(frame)
        if success:
            return bbox
        else:
            print("Tracker update failed.")
            return None