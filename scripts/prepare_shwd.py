"""Prepare SHWD without Roboflow API key. Normalizes a downloaded ZIP/folder to data/shwd YOLO format."""
import argparse, os, zipfile, shutil, glob, random

def unzip_all(src_dir="data/downloads", out_tmp="data/_tmp"):
    os.makedirs(src_dir, exist_ok=True)
    zips = glob.glob(os.path.join(src_dir, "*.zip"))
    if not zips:
        print(f"No zips in {src_dir}/. Download one of:")
        print("  A) https://universe.roboflow.com/project-oec9x/shwd-uvzhy/dataset/3 -> Download -> YOLOv8 -> ZIP (free login, no API key)")
        print("  B) https://www.kaggle.com/datasets/vodan37/yolo-helmethead -> Download (YOLO helmet/head)")
        print("  C) https://www.kaggle.com/datasets/trainingdatapro/helmet-detection -> Download")
        print(f"Then put ZIP in {src_dir}/ and rerun.")
        return []
    os.makedirs(out_tmp, exist_ok=True)
    for z in zips:
        print(f"Unzipping {z} ...")
        with zipfile.ZipFile(z, "r") as f:
            f.extractall(out_tmp)
    return zips

def find_images(root):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True) if p.lower().endswith(exts)]

def find_label_for(img_path, root):
    # try sibling labels/ dir or same dir .txt
    base = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    candidates = [
        os.path.join(os.path.dirname(img_path), base),
        os.path.join(os.path.dirname(os.path.dirname(img_path)), "labels", base),
        os.path.join(os.path.dirname(img_path).replace("images", "labels"), base),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # fallback: search by filename
    hits = glob.glob(os.path.join(root, "**", base), recursive=True)
    return hits[0] if hits else None

def main(downloads="data/downloads", tmp="data/_tmp", out="data/shwd", val_ratio=0.15, test_ratio=0.1, seed=42):
    unzip_all(downloads, tmp)
    imgs = find_images(tmp if os.path.exists(tmp) else downloads)
    if not imgs:
        print("No images found. Check ZIP contents.")
        return
    print(f"Found {len(imgs)} images. Normalizing to {out} ...")
    random.Random(seed).shuffle(imgs)
    n = len(imgs)
    n_test, n_val = int(n * test_ratio), int(n * val_ratio)
    splits = {"test": imgs[:n_test], "valid": imgs[n_test:n_test + n_val], "train": imgs[n_test + n_val:]}
    for sp, files in splits.items():
        os.makedirs(os.path.join(out, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(out, sp, "labels"), exist_ok=True)
        for src_img in files:
            dst_img = os.path.join(out, sp, "images", os.path.basename(src_img))
            # avoid collision
            if os.path.exists(dst_img):
                name, ext = os.path.splitext(os.path.basename(src_img))
                dst_img = os.path.join(out, sp, "images", f"{name}_{abs(hash(src_img)) % 100000}{ext}")
            shutil.copy(src_img, dst_img)
            lbl = find_label_for(src_img, tmp)
            dst_lbl = os.path.join(out, sp, "labels", os.path.splitext(os.path.basename(dst_img))[0] + ".txt")
            if lbl and os.path.exists(lbl):
                shutil.copy(lbl, dst_lbl)
            else:
                open(dst_lbl, "w").close()
    # If source already had data.yaml, keep classes; else write SHWD default
    yaml_src = glob.glob(os.path.join(tmp, "**", "data.yaml"), recursive=True)
    if yaml_src:
        print(f"Source data.yaml found: {yaml_src[0]}. Copying.")
        shutil.copy(yaml_src[0], os.path.join(out, "_source_data.yaml"))
    with open(os.path.join(out, "data.yaml"), "w") as fp:
        fp.write(f"path: {os.path.abspath(out)}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: helmet\n  1: head\n")
    print(f"Done. Train {len(splits['train'])}, valid {len(splits['valid'])}, test {len(splits['test'])}")
    print("Check a few labels: classes must be 0/1 only. If you see 2 (person/vest), tell me - config needs 3 classes.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--downloads", default="data/downloads")
    p.add_argument("--tmp", default="data/_tmp")
    p.add_argument("--out", default="data/shwd")
    a = p.parse_args()
    main(a.downloads, a.tmp, a.out)
