import cv2
from ultralytics import YOLO
from pathlib import Path

# --------------------------------------------------
# Load YOLOv8 pretrained model
# --------------------------------------------------
# Using general model first (no retraining yet)
MODEL_PATH = "yolov8n.pt"

_model = None

def load_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def detect_lesion(image_path_or_array):
    """
    Detects the most prominent lesion-like region.
    Returns cropped image (numpy array) or None.
    """

    model = load_model()

    # Load image
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
    else:
        img = image_path_or_array

    if img is None:
        return None

    results = model(img, conf=0.25)

    if len(results) == 0 or results[0].boxes is None:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        return None

    # Take largest box (most prominent region)
    areas = []
    for b in boxes:
        x1, y1, x2, y2 = b
        areas.append((x2 - x1) * (y2 - y1))

    idx = areas.index(max(areas))
    x1, y1, x2, y2 = map(int, boxes[idx])

    crop = img[y1:y2, x1:x2]

    # Reject tiny crops
    if crop.shape[0] < 50 or crop.shape[1] < 50:
        return None

    return crop
