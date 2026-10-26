# HelmetDetectionOnConstructionSite
Zone-aware real-time PPE compliance monitoring for construction sites using YOLO11 + multi-object tracking.

Most helmet-detection projects stop at per-frame bounding boxes. This system goes further:
tracked worker identities (ByteTrack) + danger-zone + dwell-time logic trigger a violation only when a
bare head stays inside the zone, with evidence snapshots logged to SQLite.

## Results (RTX 3050 6GB laptop, 640px)

YOLO11s — validation split (1581 images):

| Class  | Precision | Recall | mAP50 | mAP50-95 |
|--------|-----------|--------|-------|----------|
| helmet | 0.945     | 0.932  | 0.973 | 0.652    |
| head   | 0.909     | 0.869  | 0.922 | 0.601    |
| all    | 0.927     | 0.900  | 0.948 | 0.626    |

YOLO11s — held-out test split (1766 unseen images):

| Precision | Recall | mAP50 | mAP50-95 |
|-----------|--------|-------|----------|
| 0.941     | 0.920  | 0.961 | 0.642    |

Ablation — YOLOv8s vs YOLO11s (same data, 80 epochs, identical hypers):

| Model   | P     | R     | mAP50 | mAP50-95 | Params | Weights | FPS |
|---------|-------|-------|-------|----------|--------|---------|-----|
| YOLO11s | 0.927 | 0.900 | 0.948 | 0.626    | 9.4M   | 19MB    | 31  |
| YOLOv8s | 0.919 | 0.907 | 0.944 | 0.620    | 11.1M  | 22.5MB  | 76  |

YOLO11s is the final model (newer C2PSA attention, smaller). See `REPORT_ch3_ch4_draft.md` for method + full analysis.

## Dataset
SHWD-style PPE data (Kaggle `mustafatayyipbayram/ppe-detection`, 7035 images):
train 3688 / val 1581 / test 1766. Classes verified visually: `0=head (no-helmet)`,
`1=helmet`. Train instances: 14,087 helmet / 4,169 head.

## Pipeline
```
CCTV / webcam / video / image -> YOLO11s detector -> ByteTrack IDs -> zone + dwell check
  -> SQLite violation log + snapshot -> Flask API + one-page web dashboard (frontend/)
```

## Quickstart
```powershell
python -m venv venv; .\venv\Scripts\activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install ultralytics opencv-python flask pandas pillow pyyaml tqdm lapx imageio-ffmpeg

# dataset: Kaggle mustafatayyipbayram/ppe-detection -> data/downloads -> normalize
python scripts/prepare_shwd.py   # or place data manually under data/shwd/{train,val,test}

# train (6GB-safe defaults: batch 8, AMP, 640px)
python src/train.py --model yolo11s.pt --epochs 80 --imgsz 640 --batch 8

# evaluate + FPS
python src/evaluate.py --weights runs/train/yolo11s_shwd/weights/best.pt

# test-split generalization + sample predictions
python scripts/generalization_test.py

# live demo — one-page dashboard (detection, live camera, video analysis, evidence log)
python server.py            # then open http://localhost:8000
```

## Structure
```
server.py                  # Flask API: image/live/video detection + evidence endpoints
frontend/                  # one-page dashboard (index.html, style.css, app.js)
src/train.py               # training (RTX 3050 safe defaults)
src/evaluate.py            # mAP + FPS benchmark
src/detect.py              # image / video / webcam inference
src/tracking_zone.py       # ByteTrack + dwell-time zone logic
src/helmet_color.py        # helmet color -> role
src/db.py                  # SQLite evidence log
src/voc2yolo.py            # VOC -> YOLO converter (original SHWD)
scripts/download_dataset.py
scripts/prepare_shwd.py    # normalize any downloaded ZIP to data/shwd
scripts/generalization_test.py
configs/data.yaml
REPORT_ch3_ch4_draft.md    # report Chapters 3-4 draft
```

## Notes
- `weights/yolo11s_shwd_best.pt` (19MB) is included - clone and run the demo directly:
  `python server.py`, then open `http://localhost:8000`.
- `data/`, `runs/`, other `*.pt` are git-ignored. Retrain with the commands above to reproduce.
- Put your own site photos in `data/own_photos/` and rerun `generalization_test.py`.
- Demo: image / video-file mode is the reliable path; webcam needs a local camera.
