"""SHWD dataset setup: 3 options. Recommended: Roboflow YOLO version (already split + YOLO format)."""
import argparse, os

ROBOFLOW_CODE = '''
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")  # free at roboflow.com
project = rf.workspace("project-oec9x").project("shwd-uvzhy")
ds = project.version(3).download("yolov8", location="data/shwd")
# gives data/shwd/{train,valid,test}/images + labels + data.yaml
'''

def main(method="info"):
    print("SHWD = 7581 imgs, 9044 helmet(hat) + 111514 head(no-helmet). Classes: helmet, head.")
    print("Splits in Roboflow mirror: 5277 train / 1514 valid / 757 test (70/20/10).")
    print()
    print("Option A - Roboflow (recommended, YOLO-ready):")
    print(ROBOFLOW_CODE)
    print("Option B - Original VOC (njvisionpower/Safety-Helmet-Wearing-Dataset):")
    print("  1. Download from GitHub releases / GoogleDrive / BaiduDrive")
    print("  2. Unzip to data/shwd_voc/VOC2028 (Annotations + JPEGImages + ImageSets)")
    print("  3. Run: python src/voc2yolo.py --voc data/shwd_voc/VOC2028 --out data/shwd")
    print()
    print("Option C - Kaggle mirrors: search 'SHWD safety helmet wearing YOLO' on kaggle.com")
    print("  kaggle datasets download -d <mirror> -p data/ ; then convert if VOC.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="info")
    a = p.parse_args()
    if a.method == "roboflow":
        try:
            from roboflow import Roboflow
        except ImportError:
            raise SystemExit("pip install roboflow")
        key = os.environ.get("ROBOFLOW_API_KEY")
        if not key:
            raise SystemExit("Set ROBOFLOW_API_KEY env var")
        rf = Roboflow(api_key=key)
        rf.workspace("project-oec9x").project("shwd-uvzhy").version(3).download("yolov8", location="data/shwd")
    else:
        main(a.method)
