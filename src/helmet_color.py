"""Rough helmet color -> role mapping using HSV histograms inside helmet box."""
import cv2
import numpy as np

ROLE_MAP = {"white": "supervisor", "yellow": "worker", "red": "fire/safety", "blue": "visitor/engineer", "orange": "worker"}
# HSV ranges (H 0-180 in OpenCV)
COLOR_RANGES = {
    "white":  (np.array([0, 0, 180]), np.array([180, 40, 255])),
    "yellow": (np.array([20, 80, 80]), np.array([35, 255, 255])),
    "orange": (np.array([10, 80, 80]), np.array([20, 255, 255])),
    "red":    (np.array([0, 80, 80]), np.array([10, 255, 255])),  # + wrap-around handled below
    "blue":   (np.array([90, 80, 80]), np.array([130, 255, 255])),
}

def classify_helmet_color(frame_bgr, x1, y1, x2, y2):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown", "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    best, best_score = "unknown", 0
    total = hsv.shape[0] * hsv.shape[1]
    for color, (lo, hi) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, lo, hi)
        if color == "red":  # second red lobe H~170-180
            mask2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask, mask2)
        score = mask.sum() / 255 / total
        if score > best_score:
            best, best_score = color, score
    if best_score < 0.15:
        return "unknown", "unknown"
    return best, ROLE_MAP.get(best, "unknown")
