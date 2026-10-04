"""Frame-level inference for image / video / webcam."""
import argparse, cv2
from ultralytics import YOLO

def run_image(weights, source, conf=0.5, save="outputs/pred.jpg"):
    model = YOLO(weights)
    r = model.predict(source, conf=conf, verbose=False)[0]
    im = r.plot()
    cv2.imwrite(save, im)
    print(f"Saved {save} | detections: {len(r.boxes)}")
    for b in r.boxes:
        print(f"  {model.names[int(b.cls)]} {float(b.conf):.2f} {b.xyxy.tolist()}")

def run_video(weights, source=0, conf=0.5):
    model = YOLO(weights)
    cap = cv2.VideoCapture(source)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, conf=conf, verbose=False)[0]
        cv2.imshow("PPE - q to quit", r.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", default="0")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--image-mode", action="store_true")
    a = p.parse_args()
    src = int(a.source) if a.source.isdigit() else a.source
    if a.image_mode:
        run_image(a.weights, src)
    else:
        run_video(a.weights, src, a.conf)
