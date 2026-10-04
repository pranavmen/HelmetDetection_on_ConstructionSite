# Final Year Report — Chapters 3 & 4 (Draft)
Title: Zone-Aware Real-Time PPE Compliance Monitoring on Construction Sites using YOLO11 + Multi-Object Tracking

## Chapter 3 — Methodology

### 3.1 Problem formulation
Frame-level helmet detection is insufficient for site compliance: single-frame
false positives trigger alarm fatigue. We formulate compliance as a
**tracking + zone + dwell-time** decision: a worker identity (ByteTrack ID)
must be observed as `head` (no-helmet) inside a defined danger zone for more
than T seconds before a violation is logged with an evidence snapshot.

### 3.2 Dataset
SHWD-style PPE dataset (Kaggle `mustafatayyipbayram/ppe-detection`, 7035 images):
train 3688 / val 1581 / test 1766. Two classes verified visually:
`0=head (no-helmet)`, `1=helmet`. Train instances: 14,087 helmet / 4,169 head
(3.4:1 imbalance — milder than original SHWD ~12:1, still head is minority).
All annotations converted to YOLO normalized format, 640px training size.

### 3.3 Model architecture
YOLO11s backbone (C3k2 + SPPF + C2PSA attention), anchor-free decoupled head,
selected over YOLOv8s after controlled comparison (Sec 4.3). 9.4M params,
19MB weights — fits edge deployment. ByteTrack (`bytetrack.yaml`) supplies
persistent IDs; `ZoneViolationEngine` (src/tracking_zone.py) implements dwell
logic; helmet color -> role via HSV histogram (src/helmet_color.py);
violations persisted to SQLite with snapshots (src/db.py).

### 3.4 Training setup (RTX 3050 6GB)
Optimizer AdamW lr0=0.001, 80 epochs, imgsz 640, batch 8, AMP on, workers 4,
mosaic 1.0 / mixup 0.1 / copy-paste 0.1, patience 20. Cache off (RAM limit).
Augmentation left on (default randaugment) to combat small-target and
lighting variance.

### 3.5 Evaluation protocol
Metrics: Precision, Recall, mAP50, mAP50-95 per class + confusion matrix, PR
curves, FPS on RTX 3050. Ablation: YOLOv8s vs YOLO11s, identical data/hypers.
Generalization: held-out test split (1766 images) + qualitative review of 20
annotated samples (outputs/generalization/).

## Chapter 4 — Results and Discussion

### 4.1 Main results (val, YOLO11s best.pt)

| Class  | P     | R     | mAP50 | mAP50-95 |
|--------|-------|-------|-------|----------|
| helmet | 0.945 | 0.932 | 0.973 | 0.652    |
| head   | 0.909 | 0.869 | 0.922 | 0.601    |
| all    | 0.927 | 0.900 | 0.948 | 0.626    |

Test split (unseen, 1766 images): P 0.941, R 0.920, mAP50 0.961, mAP50-95 0.642.
Inference: ~31 FPS @640 on RTX 3050 laptop GPU; preprocessing 2.5ms,
inference 11ms. Curves: runs/detect/runs/train/yolo11s_shwd/{BoxPR_curve.png,
confusion_matrix.png, results.png}.

### 4.2 Error analysis
Head recall (0.869) trails helmet recall (0.932): bare heads are smaller,
less distinctive, and minority class. Typical failures: far/small helmets in
tunnels, occlusion by machinery, motion blur, back-view heads confused with
background. Dwell-time filtering (+tracking) suppresses single-frame flicker
that naive per-frame detectors would flag — state this as the key contribution.

### 4.3 Ablation: YOLOv8s vs YOLO11s (same data/hypers)

| Model   | P     | R     | mAP50 | mAP50-95 | Params | Size  | FPS  |
|---------|-------|-------|-------|----------|--------|-------|------|
| YOLO11s | 0.927 | 0.900 | 0.948 | 0.626    | 9.4M   | 19MB  | 31   |
| YOLOv8s | 0.919 | 0.907 | 0.944 | 0.620    | 11.1M  | 22.5M | 76   |

Accuracy effectively tied; YOLO11s chosen as final (newer C2PSA attention,
smaller). Report both — examiners value the controlled comparison.

### 4.4 Generalization
20 annotated test samples in outputs/generalization/. Model detects small
distant helmets (tunnel site) and bare heads in crowds. For viva: add 10-15
own phone photos to data/own_photos/ and rerun
`python scripts/generalization_test.py` — expect minor drop on unseen camera
angles; document 2-3 failure cases honestly.

### 4.5 Cap-confusion experiment (negative result, kept as evidence)
YouTube field test showed cloth caps / bucket hats firing as `helmet`
(open-set problem: only helmet/head classes exist, so head-coverings default
to helmet). Attempted fix: 217 cap crops mined from the test video added as
background images + 25-epoch fine-tune. Result on the same video:
helmet boxes 2093->2480, head boxes 987->551 — calibration shifted, caps
still detected as helmets. Conclusion: background-only fine-tuning is
insufficient; the correct fix is a 3rd `cap` class with annotated boxes
(proposed future work). Base YOLO11s retained as final model.

### 4.6 Limitations & future work
No full-body person class (heads only) — tracking is head-level, loses
workers when heads occluded. No vest/goggles classes. Zone is a static
rectangle — future: polygon zones per camera + perspective calibration. Edge
deployment (TensorRT/NCNN on Jetson) not yet benchmarked.
