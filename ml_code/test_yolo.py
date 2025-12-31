import sys
import os
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml_code.detection.yolo_detect import detect_lesion

crop = detect_lesion("a1.jpeg")

if crop is None:
    print("No lesion detected")
else:
    print("Lesion detected:", crop.shape)
    cv2.imshow("Crop", crop)
    cv2.waitKey(0)
