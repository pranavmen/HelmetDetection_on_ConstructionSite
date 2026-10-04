"""Streamlit demo: live PPE + tracking + zone violations + evidence log."""
import cv2, os, time
import streamlit as st
from ultralytics import YOLO

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.tracking_zone import ZoneViolationEngine, draw_zone
from src.helmet_color import classify_helmet_color
from src.db import log_violation, recent

st.set_page_config(page_title="PPE Compliance Monitor", layout="wide")
st.title("Zone-Aware PPE Compliance Monitor — Final Year Demo")

with st.sidebar:
    weights = st.text_input("Weights", "runs/train/yolo11s_shwd/weights/best.pt")
    conf = st.slider("Confidence", 0.1, 0.9, 0.5, 0.05)
    source = st.selectbox("Source", ["webcam", "video file", "image"])
    video_path = st.text_input("Video/image path (if file)", "")
    dwell = st.slider("Dwell seconds for violation", 1, 10, 3)
    run = st.checkbox("Run", False)

col1, col2 = st.columns([3, 1])
frame_ph = col1.empty()
stat_ph = col2.empty()
log_ph = st.empty()

def norm_cls(name: str) -> str:
    n = name.lower()
    if n in ("hat", "helmet", "helmet_with", "with_helmet"):
        return "helmet"
    if n in ("person", "head", "head_without", "without_helmet", "no_helmet", "no-helmet"):
        return "head"  # SHWD negative = bare head
    return n

def process_frame(model, frame, engine, zone):
    # SHWD has only helmet/head - violation = tracked HEAD inside zone
    res = model.track(frame, persist=True, conf=conf, verbose=False, tracker="bytetrack.yaml")[0]
    names = model.names
    annotated = res.plot()
    violations = n_helmet = n_head = 0
    if res.boxes is not None and res.boxes.id is not None:
        for b in res.boxes:
            cls = norm_cls(names[int(b.cls)])
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            tid = int(b.id) if b.id is not None else None
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if cls == "helmet":
                n_helmet += 1
                color, role = classify_helmet_color(frame, x1, y1, x2, y2)
                cv2.putText(annotated, f"{color}", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            elif cls == "head":
                n_head += 1
                # has_helmet=False always for head boxes; dwell logic suppresses flicker
                if engine.update(tid, False, cx, cy, zone):
                    snap = f"violations/track{tid}_{int(time.time())}.jpg"
                    os.makedirs("violations", exist_ok=True)
                    cv2.imwrite(snap, frame)
                    log_violation(tid, "no-helmet(head) in danger zone", snap, float(b.conf))
                    violations += 1
    annotated = draw_zone(annotated, zone)
    return annotated, violations, n_helmet, n_head

if run:
    if not os.path.exists(weights):
        st.warning(f"Weights not found: {weights}. Train first or use yolo11s.pt for smoke test.")
        model_path = "yolo11s.pt"  # will auto-download; classes differ but pipeline runs
    else:
        model_path = weights
    model = YOLO(model_path)
    engine = ZoneViolationEngine(dwell_sec=dwell, fps=30)

    if source == "image" and video_path:
        import numpy as np
        from PIL import Image
        img = np.array(Image.open(video_path).convert("RGB"))[:, :, ::-1]
        zone = (50, 50, img.shape[1]-50, img.shape[0]-50)
        ann, _, nh, nb = process_frame(model, img, engine, zone)
        frame_ph.image(ann[:, :, ::-1], channels="RGB")
        stat_ph.write(f"helmet: {nh} head: {nb}")
    else:
        cap_src = 0 if source == "webcam" else video_path
        cap = cv2.VideoCapture(cap_src)
        ok, f0 = cap.read()
        h, w = (f0.shape[:2] if ok else (480, 640))
        zone = (w//4, h//3, w*3//4, h)  # default danger zone; editable later
        total_v = 0
        while run and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            ann, v, nh, nb = process_frame(model, frame, engine, zone)
            total_v += v
            frame_ph.image(ann[:, :, ::-1], channels="RGB")
            stat_ph.metric("Zone violations", total_v)
            stat_ph.write(f"helmet: {nh} head: {nb}")
            rows = recent(10)
            log_ph.table([{"id": r[0], "ts": r[1], "track": r[2], "reason": r[3]} for r in rows])
        cap.release()
else:
    st.info("Tick Run in sidebar. Default zone = bottom-center. Edit zone coords in code for your CCTV view.")
    st.table([{"id": r[0], "ts": r[1], "track": r[2], "reason": r[3]} for r in recent(10)])
