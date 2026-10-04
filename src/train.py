"""Train YOLOv8s/YOLO11s on SHWD. Tuned for RTX 3050 6GB."""
import argparse
from ultralytics import YOLO

def main(model="yolo11s.pt", data="data/shwd/data.yaml", epochs=80, imgsz=640, batch=8, name="yolo11s_shwd"):
    # SHWD note: ~12:1 head:helmet imbalance. Ultralytics focal/cls loss handles it.
    # If helmet recall is low, add: cls=1.5, box=7.5, or oversample helmet images.
    m = YOLO(model)
    m.train(
        data=data, epochs=epochs, imgsz=imgsz, batch=batch,
        patience=20, workers=4, cache=False,  # cache=True needs >16GB RAM
        optimizer="AdamW", lr0=0.001,
        augment=True, mosaic=1.0, mixup=0.1, copy_paste=0.1,
        device=0, amp=True,  # mixed precision for 6GB
        project="runs/train", name=name, exist_ok=True,
        val=True, plots=True,
    )
    print(f"Done. Best: runs/train/{name}/weights/best.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11s.pt", help="yolo11s.pt | yolov8s.pt | yolo11n.pt")
    p.add_argument("--data", default="data/shwd/data.yaml")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--name", default="yolo11s_shwd")
    a = p.parse_args()
    main(a.model, a.data, a.epochs, a.imgsz, a.batch, a.name)
