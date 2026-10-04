"""Cap hard-negative mining: crop sub-0.9-conf 'helmet' predictions from a video.
Used for the cap-confusion experiment (see REPORT_ch3_ch4_draft.md Sec 4.5).
Result: 217 cap crops -> background fine-tune did NOT fix caps-as-helmets.
Kept for reproducibility of the negative result."""
import argparse, cv2
from ultralytics import YOLO

def main(weights, video, outdir="data/cap_neg", conf=0.45, max_conf=0.9, stride=10):
    import os
    os.makedirs(outdir, exist_ok=True)
    m = YOLO(weights)
    cap = cv2.VideoCapture(video)
    saved, fno = 0, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if fno % stride == 0:
            r = m.predict(f, conf=conf, verbose=False)[0]
            if r.boxes is not None:
                for j, b in enumerate(r.boxes):
                    if m.names[int(b.cls)] == "helmet" and float(b.conf) < max_conf:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        pad = 15
                        crop = f[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                        if crop.size > 0 and crop.shape[0] > 30:
                            cv2.imwrite(f"{outdir}/crop_{fno}_{j}.jpg", crop)
                            saved += 1
        fno += 1
    cap.release()
    print(f"crops saved: {saved} -> {outdir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/yolo11s_shwd_best.pt")
    p.add_argument("--video", required=True)
    p.add_argument("--outdir", default="data/cap_neg")
    a = p.parse_args()
    main(a.weights, a.video, a.outdir)
