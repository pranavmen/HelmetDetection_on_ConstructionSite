"""Evaluate: mAP + per-class P/R + FPS benchmark on RTX 3050."""
import argparse, time
from ultralytics import YOLO

def main(weights, data="data/shwd/data.yaml", imgsz=640):
    model = YOLO(weights)
    metrics = model.val(data=data, imgsz=imgsz, plots=True)
    print(f"mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
    print(f"P: {metrics.box.mp:.4f}  R: {metrics.box.mr:.4f}")

    # FPS benchmark (GPU warmup + 100 runs)
    import torch, numpy as np
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(10):
        _ = model.predict(dummy, verbose=False, device=0)
    t0 = time.time()
    n = 100
    for _ in range(n):
        _ = model.predict(dummy, verbose=False, device=0)
    fps = n / (time.time() - t0)
    print(f"Approx FPS (RTX3050, imgsz={imgsz}): {fps:.1f}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="configs/data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    a = p.parse_args()
    main(a.weights, a.data, a.imgsz)
