"""Generalization test: evaluate best weights on held-out TEST split + save annotated samples."""
import argparse, glob, os, shutil
from ultralytics import YOLO

def main(weights="runs/train/yolo11s_shwd/weights/best.pt", data="data/shwd/data.yaml", n_samples=20, conf=0.5):
    model = YOLO(weights)
    print("== Test-split metrics (unseen data) ==")
    metrics = model.val(data=data, split="test", verbose=False)
    print(f"test mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
    print(f"test P: {metrics.box.mp:.4f}  R: {metrics.box.mr:.4f}")

    out = "outputs/generalization"
    os.makedirs(out, exist_ok=True)
    imgs = sorted(glob.glob("data/shwd/test/images/*"))[:n_samples]
    res = model.predict(imgs, conf=conf, verbose=False)
    for i, r in enumerate(res):
        r.save(filename=os.path.join(out, f"gen_{i}.jpg"))
    print(f"Saved {len(res)} annotated samples to {out}/")
    print("To test YOUR OWN photos: put them in data/own_photos/ and rerun with --own data/own_photos")

    own = "data/own_photos"
    if os.path.isdir(own) and glob.glob(os.path.join(own, "*")):
        r2 = model.predict(sorted(glob.glob(os.path.join(own, "*")))[:50], conf=conf, verbose=False)
        for i, r in enumerate(r2):
            r.save(filename=os.path.join(out, f"own_{i}.jpg"))
        print(f"Saved {len(r2)} own-photo predictions to {out}/")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/train/yolo11s_shwd/weights/best.pt")
    p.add_argument("--data", default="data/shwd/data.yaml")
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--conf", type=float, default=0.5)
    a = p.parse_args()
    main(a.weights, a.data, a.n_samples, a.conf)
