# Project Explanation — Helmet Detection on Construction Sites

> **One-line summary:** An AI-based safety-compliance system that watches construction-site camera
> footage, detects whether each worker is wearing a safety helmet, and raises a **violation alert
> with photographic evidence** only when a bare-headed worker stays inside a danger zone for a
> continuous few seconds — instead of flashing alarms on every single frame.

*(Use this document to walk your teacher through the project. Sections are ordered the way you
should present them: Problem → Solution → Working → Results → Live Demo.)*

---

## 1. The Problem

Construction sites are among the most hazardous workplaces, and head injuries are one of the most
serious risks. Site rules require every worker to wear a safety helmet, but compliance is usually
checked **manually** by a safety officer walking around.

Problems with manual checking:

- A site has many cameras — a human cannot watch all of them continuously.
- Checking is periodic, so violations happen unnoticed between rounds.
- Recording violations (photo + time) for reports is slow and error-prone.

The naive software alternative — a detector that raises an alarm whenever it sees a bare head in
any frame — has its own problem: **false alarms**. A worker bending down, a passing visitor, or one
blurry frame can trigger the alarm. If an alarm rings 50 times a day and 45 are false, people stop
listening to it. This is called **alarm fatigue**.

## 2. What This Project Does

This project builds a complete PPE (Personal Protective Equipment) compliance monitoring system:

1. **Detects** helmets and bare heads in video frames using a custom-trained YOLO11s deep-learning
   model.
2. **Tracks** each person across frames so the system knows *which* worker is who (ByteTrack
   algorithm) — it does not treat every frame independently.
3. **Applies a safety rule:** a violation is logged only when the **same tracked person's bare
   head stays inside the danger zone continuously for a dwell time (default 3 seconds)**.
4. **Logs evidence automatically:** each violation stores the timestamp, track ID, reason, model
   confidence, and a cropped snapshot of the offending worker into a **SQLite database** — ready
   for audit reports.
5. **Bonus — helmet colour → role:** the system estimates the helmet colour (HSV colour analysis)
   and maps it to a site role: white = supervisor, yellow/orange = worker, blue =
   visitor/engineer, red = fire/safety personnel.

So the project is not just an object detector — it is a **decision system** that turns raw model
output into meaningful, low-false-alarm safety alerts with an audit trail.

## 3. How It Works (Pipeline)

```
CCTV / webcam / uploaded video / image
        │
        ▼
┌─────────────────────┐
│  YOLO11s detector   │  → finds "helmet" and "head" boxes with confidence scores
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  ByteTrack tracker  │  → assigns a persistent ID to each worker frame-to-frame
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  Zone + dwell-time rule (the core novelty)  │
│  bare head inside zone, same ID, ≥ 3 sec    │
│  → VIOLATION (single alarm, not per-frame)  │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐     ┌──────────────────────────┐
│  SQLite evidence DB │ ◄── │ Snapshot image of worker │
└─────────────────────┘     └──────────────────────────┘
        │
        ▼
  Flask web server (localhost:8000) → one-page web dashboard (HTML/CSS/JS)
```

Key components:

| Component | Technology | Role in the project |
|---|---|---|
| Object detection | YOLO11s (Ultralytics) | Finds helmet / bare-head boxes at ~31 FPS |
| Tracking | ByteTrack | Keeps a stable ID per worker across frames |
| Zone + dwell logic | `src/tracking_zone.py` (custom) | Filters single-frame false alarms |
| Helmet colour → role | `src/helmet_color.py` (HSV histograms) | Identifies supervisor/worker/visitor |
| Evidence storage | SQLite (`src/db.py`) | Timestamped violation log + snapshot |
| Web backend | Python + Flask (`server.py`) | REST API for image / live / video analysis |
| Web frontend | Plain HTML / CSS / JavaScript | One-page demo dashboard (no framework) |

## 4. Dataset

- **SHWD-style PPE detection dataset** (Kaggle: `mustafatayyipbayram/ppe-detection`), **7,035
  images** of construction workers, annotated in YOLO format.
- Two classes: `helmet` and `head` (no-helmet). Class labels were **verified visually** after
  conversion.
- Split: **train 3,688 / validation 1,581 / test 1,766** images (held-out test set never seen
  during training).
- Training instances: 14,087 helmets / 4,169 bare heads.
- Images resized to 640×640 for training.

## 5. Model Training

- Model: **YOLO11s** (small variant) — chosen for the best accuracy/size trade-off.
- Settings: 80 epochs, batch size 8, 640px image size, Automatic Mixed Precision (AMP), on an
  RTX 3050 6GB laptop GPU.
- Trained weights are only **19 MB (9.4M parameters)** — small enough to deploy on modest
  hardware.

### Ablation study (comparing architectures on identical data & settings)

| Model | Precision | Recall | mAP50 | mAP50-95 | Params | Weights | FPS |
|---|---|---|---|---|---|---|---|
| **YOLO11s ★** | 0.927 | 0.900 | **0.948** | **0.626** | **9.4M** | **19MB** | 31 |
| YOLOv8s | 0.919 | 0.907 | 0.944 | 0.620 | 11.1M | 22.5MB | 76 |

YOLO11s gave higher accuracy **with 15% fewer parameters** (it uses the newer C2PSA attention
block), so it was selected as the final model.

### Results

**Validation split (1,581 images):**

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| helmet | 0.945 | 0.932 | 0.973 | 0.652 |
| head | 0.909 | 0.869 | 0.922 | 0.601 |
| **all** | **0.927** | **0.900** | **0.948** | **0.626** |

**Held-out test split (1,766 unseen images):**

| Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| **0.941** | **0.920** | **0.961** | 0.642 |

- **mAP50 of 0.961 on unseen test images** means the model is ~96% accurate at localizing and
  classifying helmets/heads at the standard IoU threshold.
- Test score being *higher* than validation score shows the model generalizes well and did not
  overfit.
- Speed: ~31 FPS at 640px on the RTX 3050 → suitable for near-real-time processing (the web demo
  runs on CPU, ~0.2–0.5 s per image).

## 6. Live Web Demo

Run `python server.py` and open **http://localhost:8000**. The single-page dashboard offers:

1. **Image detection** — upload any site photo; boxes are drawn with class, confidence, and
   helmet-colour role. A confidence slider re-runs detection instantly.
2. **Video analysis** — drop a video file; the server runs the **full pipeline** (detection →
   tracking → zone/dwell rule), shows a progress bar, then plays back the annotated video with
   bounding boxes, track IDs, and lists every violation with its frame number. Violations also
   appear as toast alerts.
3. **Results section** — the validation/test metrics and the YOLOv8s-vs-YOLO11s ablation table.
4. Violations counted live from the SQLite log (evidence snapshots saved under `media/snapshots/`).

Everything runs **locally** — model, database, and server; no cloud, no internet needed at demo
time.

## 7. How to Run (quick reference)

```powershell
# 1. activate the virtual environment
.\venv\Scripts\activate

# 2. start the web server
python server.py        # then open http://localhost:8000

# retraining (only if asked to reproduce):
python src/train.py --model yolo11s.pt --epochs 80 --imgsz 640 --batch 8
python src/evaluate.py --weights runs/train/yolo11s_shwd/weights/best.pt
```

## 8. Project Structure

```
server.py                 # Flask API: image / live / video detection + evidence endpoints
frontend/                 # one-page dashboard (index.html, style.css, app.js)
src/
  train.py                # model training
  evaluate.py             # mAP + FPS evaluation
  detect.py               # image / video / webcam inference
  tracking_zone.py        # ByteTrack + danger-zone + dwell-time engine  ← core logic
  helmet_color.py         # helmet colour → role classification
  db.py                   # SQLite evidence log
scripts/
  prepare_shwd.py         # dataset download/normalization
  generalization_test.py  # test on unseen images
configs/data.yaml         # dataset class config
weights/yolo11s_shwd_best.pt   # trained model (19 MB, included)
violations.db             # SQLite evidence log
media/snapshots/          # saved violation snapshots
```

## 9. Limitations & Future Scope

**Current limitations (honest points you can raise):**
- Helmet-colour → role mapping is heuristic (HSV colour ratios), so it can misjudge under unusual
  lighting or coloured filters.
- Dwell-time rule uses a fixed default danger zone per deployment; it is configured on the server
  rather than drawn interactively in the current UI.
- Runs on CPU in the demo (~2–4 FPS per stream); multiple simultaneous cameras would need a GPU
  or edge device (e.g., Jetson).

**Possible future enhancements:**
- Deep-learning-based colour/role classification instead of HSV heuristics.
- Multi-camera deployment with a central dashboard and email/SMS alerts.
- Automatic report generation (PDF) from the SQLite evidence log for safety audits.
- Detecting other PPE: reflective vests, safety shoes, scaffolding hazards.
- Edge deployment on Jetson/Raspberry-Pi class devices for on-site processing.

## 10. Likely Teacher Questions — Quick Answers

**Q: What is YOLO and why YOLO11s?**
YOLO ("You Only Look Once") is a single-shot object detector that finds all objects in one pass
over the image — very fast. The "s" (small) variant balances accuracy and speed; our ablation
study showed it beat YOLOv8s in accuracy while being smaller.

**Q: What is mAP?**
Mean Average Precision — the standard detection metric. mAP50 counts a detection correct if the
box overlaps the ground truth by ≥50%; mAP50-95 averages over stricter overlap thresholds. Our
test mAP50 = 0.961.

**Q: Why tracking? Why not just detect per frame?**
Per-frame detection can't tell whether an alarm is a new event or the same person continuing.
ByteTrack gives each worker a stable ID, so the dwell-timer can require the *same* person to stay
unprotected for N seconds — this is what kills false alarms.

**Q: What exactly is the dwell-time rule?**
A violation fires only when one tracked ID's bare head remains inside the danger zone
continuously for ≥3 seconds. A quick bend or a single blurry frame resets the timer and raises
nothing.

**Q: Where is the data stored?**
In a SQLite database (`violations.db`) — one row per violation with timestamp, track ID, reason,
confidence, and a path to the saved snapshot image. SQLite needs no server, ideal for a demo.

**Q: What is Flask?**
A lightweight Python web framework. It serves the dashboard and exposes REST endpoints
(`/api/detect`, `/api/video`, `/api/violations`) that the HTML/JS frontend calls.

**Q: Is this real-time?**
Near-real-time: 31 FPS on a GPU (faster than a CCTV's 25 FPS). The demo runs on CPU, which is
slower but fine for uploaded images/videos.

**Q: What's the novelty compared to ordinary helmet detectors?**
Most projects stop at per-frame bounding boxes. This one adds identity tracking + a
zone-and-dwell decision rule + an automatic evidence audit trail — turning a detector into a
usable safety-monitoring system with far fewer false alarms.

---

*Project: HelmetDetection_on_ConstructionSite · YOLO11s + ByteTrack + Flask · model weights
included (19 MB) — runs fully offline.*
