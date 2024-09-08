import cv2
import numpy as np

def mouse_click(event, x, y, flags, param):
    """Handle mouse click events. Record the click coordinates."""
    if event == cv2.EVENT_LBUTTONDOWN:
        param['click'] = (x, y)
        print(f"Clicked at: ({x}, {y})")  # Debug: Confirm the click is registered

def resize_image(image, dimensions):
    """Resize the image to the given dimensions."""
    if image is None or dimensions is None:
        raise ValueError("Image and dimensions must not be None.")
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_LINEAR)

def detect_contour_around_click(image, click_point, config):
    """Detect the largest contour around the clicked location.

    Args:
        image (ndarray): The input image.
        click_point (tuple): Coordinates of the mouse click.
        config (dict): Configuration dictionary containing contour parameters.

    Returns:
        tuple: Bounding box (x, y, w, h) around the detected contour or None if no contour is found.
    """
    if image is None or click_point is None:
        raise ValueError("Image and click_point must not be None.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, config['CONTOUR_APPROXIMATION_METHOD'])
    selected_contour = None

    for contour in contours:
        if cv2.pointPolygonTest(contour, click_point, False) >= 0:
            if selected_contour is None or cv2.contourArea(contour) > cv2.contourArea(selected_contour):
                selected_contour = contour

    if selected_contour is not None:
        x, y, w, h = cv2.boundingRect(selected_contour)
        return x, y, w, h
    else:
        return None

def extract_roi(image, bounding_box):
    """Extract Region of Interest (ROI) from the image based on the bounding box.

    Args:
        image (ndarray): The input image.
        bounding_box (tuple): The bounding box (x, y, w, h).

    Returns:
        ndarray: Cropped image representing the ROI.
    """
    if image is None or bounding_box is None:
        raise ValueError("Image and bounding_box must not be None.")
    
    x, y, w, h = bounding_box
    return image[y:y+h, x:x+w]

def convert_to_xywh(box):
    """Convert (x1, y1, x2, y2) format to (x, y, w, h) format.

    Args:
        box (tuple): Bounding box in (x1, y1, x2, y2) format.

    Returns:
        tuple: Bounding box in (x, y, w, h) format.
    """
    if box is None or len(box) != 4:
        raise ValueError("Box must be a tuple of four elements.")
    
    x1, y1, x2, y2 = box
    return x1, y1, x2-x1, y2-y1

def convert_to_xyxy(box):
    """Convert (x, y, w, h) format to (x1, y1, x2, y2) format.

    Args:
        box (tuple): Bounding box in (x, y, w, h) format.

    Returns:
        tuple: Bounding box in (x1, y1, x2, y2) format.
    """
    if box is None or len(box) != 4:
        raise ValueError("Box must be a tuple of four elements.")
    
    x, y, w, h = box
    return x, y, x+w, y+h

def adjust_bbox_to_original_scale(box, contour_bbox, resized_dimensions, original_dimensions):
    """Adjust bounding box from resized ROI back to original image scale.

    Args:
        box (tuple): Bounding box in (x1, y1, x2, y2) format relative to the resized ROI.
        contour_bbox (tuple): Bounding box of the contour in the original image.
        resized_dimensions (tuple): Dimensions of the resized ROI (width, height).
        original_dimensions (tuple): Dimensions of the original image (height, width, channels).

    Returns:
        tuple: Adjusted bounding box in (x1, y1, x2, y2) format relative to the original image.
    """
    x, y, w, h = contour_bbox
    resized_width, resized_height = resized_dimensions
    orig_height, orig_width = original_dimensions[:2]

    scale_x = w / resized_width
    scale_y = h / resized_height

    x1, y1, x2, y2 = box
    x1 = int(x + x1 * scale_x)
    y1 = int(y + y1 * scale_y)
    x2 = int(x + x2 * scale_x)
    y2 = int(y + y2 * scale_y)

    return x1, y1, x2, y2
