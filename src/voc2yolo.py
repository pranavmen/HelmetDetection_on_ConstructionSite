"""Convert original SHWD VOC2028 -> YOLO. Maps hat->0(helmet), person->1(head)."""
import argparse, os, xml.etree.ElementTree as ET, shutil, random

MAP = {"hat": 0, "helmet": 0, "person": 1, "head": 1}

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in MAP:
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        cx, cy = ((x1 + x2) / 2 / w), ((y1 + y2) / 2 / h)
        bw, bh = ((x2 - x1) / w), ((y2 - y1) / h)
        boxes.append((MAP[name], cx, cy, bw, bh))
    return boxes

def main(voc, out, val_ratio=0.15, test_ratio=0.1, seed=42):
    img_dir = os.path.join(voc, "JPEGImages")
    ann_dir = os.path.join(voc, "Annotations")
    imgs = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    random.Random(seed).shuffle(imgs)
    n = len(imgs)
    n_test, n_val = int(n * test_ratio), int(n * val_ratio)
    splits = {"test": imgs[:n_test], "valid": imgs[n_test:n_test + n_val], "train": imgs[n_test + n_val:]}
    for sp, files in splits.items():
        os.makedirs(os.path.join(out, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(out, sp, "labels"), exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(img_dir, f), os.path.join(out, sp, "images", f))
            xml = os.path.join(ann_dir, os.path.splitext(f)[0] + ".xml")
            label = os.path.join(out, sp, "labels", os.path.splitext(f)[0] + ".txt")
            if not os.path.exists(xml):
                open(label, "w").close()
                continue
            with open(label, "w") as fp:
                for c, cx, cy, bw, bh in parse_xml(xml):
                    fp.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    with open(os.path.join(out, "data.yaml"), "w") as fp:
        fp.write(f"path: {os.path.abspath(out)}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: helmet\n  1: head\n")
    print(f"Done: {n} images -> {out}. Train {len(splits['train'])}, valid {len(splits['valid'])}, test {len(splits['test'])}")
    print("SHWD is imbalanced (~12:1 head:helmet). Keep all data; YOLO handles it, but watch recall on helmet.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--voc", required=True, help="path to VOC2028 folder")
    p.add_argument("--out", default="data/shwd")
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.1)
    a = p.parse_args()
    main(a.voc, a.out, a.val_ratio, a.test_ratio)
