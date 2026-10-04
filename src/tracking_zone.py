"""ByteTrack + zone + dwell-time violation logic. This is the final-year novelty."""
from collections import defaultdict
import cv2

class ZoneViolationEngine:
    """Trigger violation only if a no-helmet ID stays in danger zone > dwell_sec."""
    def __init__(self, dwell_sec=3.0, fps=30):
        self.dwell_frames = int(dwell_sec * fps)
        self.counter = defaultdict(int)   # track_id -> consecutive violation frames
        self.fired = set()                # (track_id) already logged

    def in_zone(self, cx, cy, zone):
        x1, y1, x2, y2 = zone
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def update(self, track_id, has_helmet, cx, cy, zone):
        if track_id is None:
            return False
        bad = (not has_helmet) and self.in_zone(cx, cy, zone)
        self.counter[track_id] = self.counter[track_id] + 1 if bad else 0
        if self.counter[track_id] >= self.dwell_frames and track_id not in self.fired:
            self.fired.add(track_id)
            return True
        return False

    def reset_id(self, track_id):
        self.counter.pop(track_id, None)
        self.fired.discard(track_id)


def draw_zone(frame, zone, label="DANGER ZONE"):
    x1, y1, x2, y2 = zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame
