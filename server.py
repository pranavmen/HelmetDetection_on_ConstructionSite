"""Flask backend for the one-page helmet-detection demo (replaces the Streamlit app).

Serves the static frontend (frontend/) and exposes the detection pipeline:
  POST /api/detect   -> single-image inference (helmet / head + colour -> role)
  POST /api/live     -> webcam frame inference with ByteTrack + zone/dwell engine
  POST /api/video    -> background job: full video tracking + zone + dwell + evidence
  GET  /api/video/<id>  -> job progress / result
  GET  /api/violations  -> SQLite evidence log
  GET  /api/info         -> model + server status
  GET  /api/samples      -> demo images from outputs/
  GET  /samples/<path>   -> demo image files
  GET  /media/<path>     -> processed videos + violation snapshots
"""
import datetime
import json
import os
import shutil
import subprocess
import threading
import time
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)  # db.py writes violations.db relative to CWD

from src.db import log_violation, recent  # noqa: E402
from src.helmet_color import classify_helmet_color  # noqa: E402
from src.tracking_zone import ZoneViolationEngine, draw_zone  # noqa: E402

FRONTEND = os.path.join(ROOT, "frontend")
OUTPUTS = os.path.join(ROOT, "outputs")
MEDIA = os.path.join(ROOT, "media")
PROCESSED = os.path.join(MEDIA, "processed")
SNAPSHOTS = os.path.join(MEDIA, "snapshots")
UPLOADS = os.path.join(MEDIA, "uploads")
WEIGHTS = os.path.join(ROOT, "weights", "yolo11s_shwd_best.pt")
FALLBACK_WEIGHTS = "yolo11s.pt"  # auto-downloaded for a smoke test if best.pt is missing

for _d in (MEDIA, PROCESSED, SNAPSHOTS, UPLOADS):
    os.makedirs(_d, exist_ok=True)

MAX_FRAMES_PER_VIDEO = 3000  # CPU safety cap (~2 min @25fps)
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = Flask(__name__, static_folder=FRONTEND, static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB uploads

# ----------------------------------------------------------------------------- model
MODEL = None
MODEL_PATH = None
MODEL_LOCK = threading.Lock()  # ultralytics predictor + tracker state are not thread-safe


def load_model():
    global MODEL, MODEL_PATH
    path = WEIGHTS if os.path.exists(WEIGHTS) else FALLBACK_WEIGHTS
    MODEL = YOLO(path)
    MODEL_PATH = path
    return MODEL


def norm_cls(name: str) -> str:
    """Map raw label names to the two canonical classes (same mapping as the original app)."""
    n = str(name).lower()
    if n in ("hat", "helmet", "helmet_with", "with_helmet"):
        return "helmet"
    if n in ("person", "head", "head_without", "without_helmet", "no_helmet", "no-helmet"):
        return "head"
    return n


def build_detections(model, frame, res, track_mode=False):
    """Convert an ultralytics result into plain dicts for the frontend."""
    dets = []
    if res.boxes is None or len(res.boxes) == 0:
        return dets
    names = model.names
    for b in res.boxes:
        cls = norm_cls(names[int(b.cls)])
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        d = {
            "cls": cls,
            "conf": round(float(b.conf), 3),
            "bbox": [x1, y1, x2, y2],
            "track_id": int(b.id) if (track_mode and b.id is not None) else None,
        }
        if cls == "helmet":
            color, role = classify_helmet_color(frame, x1, y1, x2, y2)
            d["color"] = color
            d["role"] = role
        dets.append(d)
    return dets


def counts(dets):
    return {
        "helmet": sum(1 for d in dets if d["cls"] == "helmet"),
        "head": sum(1 for d in dets if d["cls"] == "head"),
    }


def zone_px(zone_norm, w, h):
    zx, zy, zw, zh = zone_norm
    return (int(zx * w), int(zy * h), int((zx + zw) * w), int((zy + zh) * h))


# ----------------------------------------------------------------------------- routes
@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.get("/health")
def health():
    return jsonify(ok=True, status="up", model=os.path.basename(MODEL_PATH))


@app.get("/api/info")
def api_info():
    device = "CUDA GPU" if _cuda() else "CPU"
    return jsonify(
        ok=True,
        model={
            "file": os.path.basename(MODEL_PATH),
            "size_mb": round(os.path.getsize(MODEL_PATH) / 1e6, 1) if os.path.exists(MODEL_PATH) else None,
            "classes": ["head", "helmet"],
            "device": device,
            "trained_weights": os.path.exists(WEIGHTS),
        },
        violations_count=_db_count(),
        active_job=_active_job(),
        server_time=datetime.datetime.now().isoformat(timespec="seconds"),
    )


def _cuda():
    import torch
    return torch.cuda.is_available()


def _db_count():
    from src.db import conn
    c = conn()
    n = c.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
    c.close()
    return n


# Gallery order curated so the demo opens on strong detections (some stored
# outputs are 0-box images kept for the report; they're skipped here).
SAMPLE_ORDER = ["gen_16.jpg", "gen_19.jpg", "gen_6.jpg", "gen_18.jpg", "capfix_frame1400.jpg",
                "gen_13.jpg", "gen_10.jpg", "gen_11.jpg", "gen_5.jpg", "gen_9.jpg",
                "gen_17.jpg", "gen_15.jpg", "gen_14.jpg", "gen_12.jpg", "gen_4.jpg",
                "gen_7.jpg", "gen_8.jpg", "gen_3.jpg"]
SAMPLE_SKIP = {"cap_grid.jpg", "test_pred_0.jpg", "test_pred_1.jpg", "test_pred_2.jpg",
               "gen_0.jpg", "gen_1.jpg", "gen_2.jpg"}


@app.get("/api/samples")
def api_samples():
    items = []
    if os.path.isdir(OUTPUTS):
        for fn in sorted(os.listdir(OUTPUTS)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")) and fn not in SAMPLE_SKIP:
                items.append({"name": fn, "url": f"/samples/{fn}"})
    gen_dir = os.path.join(OUTPUTS, "generalization")
    if os.path.isdir(gen_dir):
        for fn in sorted(os.listdir(gen_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")) and fn not in SAMPLE_SKIP:
                items.append({"name": fn, "url": f"/samples/generalization/{fn}"})
    items.sort(key=lambda s: SAMPLE_ORDER.index(s["name"]) if s["name"] in SAMPLE_ORDER else 999)
    return jsonify(ok=True, samples=items)


@app.get("/samples/<path:fname>")
def samples_file(fname):
    return send_from_directory(OUTPUTS, fname)


@app.get("/media/<path:fname>")
def media_file(fname):
    return send_from_directory(MEDIA, fname)


@app.post("/api/detect")
def api_detect():
    f = request.files.get("image")
    if f is None:
        return jsonify(ok=False, error="No image uploaded"), 400
    conf = float(request.form.get("conf", 0.5))
    arr = np.frombuffer(f.read(), np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(ok=False, error="Could not decode image"), 400

    t0 = time.time()
    with MODEL_LOCK:
        res = MODEL.predict(frame, conf=conf, imgsz=640, verbose=False)[0]
        dets = build_detections(MODEL, frame, res)
    ms = int((time.time() - t0) * 1000)
    h, w = frame.shape[:2]
    c = counts(dets)
    return jsonify(ok=True, width=w, height=h, detections=dets, counts=c, ms=ms)


# live webcam session state (single-session demo, mirrors the original Streamlit logic)
LIVE = {"engine": None, "violations": 0, "dwell": 3.0}


@app.post("/api/live")
def api_live():
    f = request.files.get("frame")
    if f is None:
        return jsonify(ok=False, error="No frame uploaded"), 400
    conf = float(request.form.get("conf", 0.5))
    dwell = float(request.form.get("dwell", 3.0))
    fps = max(1.0, float(request.form.get("fps", 8.0)))
    zone_norm = json.loads(request.form.get("zone", "[0.25,0.33,0.5,0.67]"))
    if request.form.get("reset"):
        LIVE["engine"] = None
        LIVE["violations"] = 0

    arr = np.frombuffer(f.read(), np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(ok=False, error="Could not decode frame"), 400
    h, w = frame.shape[:2]
    zone = zone_px(zone_norm, w, h)

    if LIVE["engine"] is None or abs(LIVE["dwell"] - dwell) > 1e-6:
        LIVE["engine"] = ZoneViolationEngine(dwell_sec=dwell, fps=fps)
        LIVE["dwell"] = dwell
    engine = LIVE["engine"]
    engine.dwell_frames = max(1, int(dwell * fps))  # real-seconds dwell at the client's frame rate

    fired = []
    t0 = time.time()
    with MODEL_LOCK:
        res = MODEL.track(frame, persist=True, conf=conf, imgsz=640,
                          verbose=False, tracker="bytetrack.yaml")[0]
        dets = build_detections(MODEL, frame, res, track_mode=True)
        annotated = draw_zone(res.plot(), zone)
        for d in dets:
            if d["cls"] != "head":
                continue
            x1, y1, x2, y2 = d["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            d["in_zone"] = engine.in_zone(cx, cy, zone)
            tid = d["track_id"]
            if tid is not None:
                if engine.update(tid, False, cx, cy, zone):
                    os.makedirs(SNAPSHOTS, exist_ok=True)
                    snap_rel = f"media/snapshots/live_{tid}_{int(time.time())}.jpg"
                    cv2.imwrite(os.path.join(ROOT, snap_rel), annotated)
                    reason = f"bare head (track {tid}) in danger zone for {dwell:g}s"
                    log_violation(tid, reason, snap_rel, float(d["conf"]))
                    LIVE["violations"] += 1
                    fired.append({"track_id": tid, "reason": reason, "snapshot": snap_rel})
                d["dwell_progress"] = round(min(1.0, engine.counter[tid] / engine.dwell_frames), 2)
            elif d["in_zone"]:
                d["dwell_progress"] = 0.0
    ms = int((time.time() - t0) * 1000)
    c = counts(dets)
    c["in_zone_head"] = sum(1 for d in dets if d["cls"] == "head" and d.get("in_zone"))
    return jsonify(ok=True, width=w, height=h, detections=dets, counts=c, ms=ms,
                   fired=fired, violations=LIVE["violations"], dwell=dwell)


# ----------------------------------------------------------------------------- video jobs
JOBS = {}  # job_id -> dict (in-memory job store)


def _active_job():
    for jid, j in JOBS.items():
        if j["status"] == "running":
            return jid
    return None


def _prune_jobs():
    """Keep only the 6 most recent finished jobs (and their output files)."""
    done = [jid for jid, j in JOBS.items() if j["status"] != "running"]
    for jid in sorted(done, key=lambda k: JOBS[k]["created"])[:-6]:
        j = JOBS.pop(jid, None)
        if j:
            for p in (j.get("src"), os.path.join(PROCESSED, f"{jid}.mp4")):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass


def _ffmpeg_writer(out_path, w, h, fps):
    """Return (kind, handle, closer). Prefers H.264 via the bundled ffmpeg (browser-safe)."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        err = open(out_path + ".log", "wb")
        proc = subprocess.Popen(
            [exe, "-y", "-v", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
             "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "22",
             out_path],
            stdin=subprocess.PIPE, stderr=err)
        return ("ffmpeg", proc, err)
    except Exception:
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        return ("cv2", vw, None)


def _write_frame(kind, handle, frame):
    if kind == "ffmpeg":
        handle.stdin.write(frame.tobytes())
    else:
        handle.write(frame)


def _close_writer(kind, handle, err):
    if kind == "ffmpeg":
        try:
            handle.stdin.close()
            handle.wait(timeout=30)
        finally:
            if err:
                err.close()
    else:
        handle.release()


def _run_video_job(job):
    job_id = job["id"]
    try:
        cap = cv2.VideoCapture(job["src"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not (0 < fps <= 120):
            fps = 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not (w and h):
            raise RuntimeError("Could not read video dimensions")

        zone = zone_px(job["zone"], w, h)
        engine = ZoneViolationEngine(dwell_sec=job["dwell"], fps=fps)
        model = YOLO(MODEL_PATH)  # job-local model: independent ByteTrack state
        out_path = os.path.join(PROCESSED, f"{job_id}.mp4")
        kind, handle, err = _ffmpeg_writer(out_path, w, h, fps)

        n = 0
        t_start = time.time()
        while n < min(total, MAX_FRAMES_PER_VIDEO):
            ok, frame = cap.read()
            if not ok:
                break
            res = model.track(frame, persist=True, conf=job["conf"], imgsz=640,
                              verbose=False, tracker="bytetrack.yaml")[0]
            annotated = draw_zone(res.plot(), zone)
            dets = build_detections(model, frame, res, track_mode=True)
            for d in dets:
                if d["cls"] != "head":
                    continue
                x1, y1, x2, y2 = d["bbox"]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if engine.update(d["track_id"], False, cx, cy, zone):
                    snap_rel = f"media/snapshots/{job_id}_track{d['track_id']}_{n}.jpg"
                    cv2.imwrite(os.path.join(ROOT, snap_rel), annotated)
                    reason = (f"bare head (track {d['track_id']}) in danger zone "
                              f"for {job['dwell']:g}s (video)")
                    log_violation(d["track_id"], reason, snap_rel, float(d["conf"]))
                    job["violations"].append({"track_id": d["track_id"], "reason": reason,
                                              "snapshot": snap_rel, "frame": n})
            _write_frame(kind, handle, annotated)
            n += 1
            if n % 5 == 0 or n >= total:
                job.update(frames=n, total=total,
                           progress=round(min(1.0, n / min(total, MAX_FRAMES_PER_VIDEO)), 3),
                           elapsed=round(time.time() - t_start, 1))

        cap.release()
        _close_writer(kind, handle, err)
        if n == 0:
            raise RuntimeError("No frames could be read from the video")
        job.update(status="done", progress=1.0, frames=n,
                   truncated=n >= MAX_FRAMES_PER_VIDEO,
                   video=f"/media/processed/{job_id}.mp4",
                   elapsed=round(time.time() - t_start, 1))
    except Exception as e:  # noqa: BLE001 - surface any failure to the UI
        job.update(status="error", error=str(e))
    finally:
        if job.get("src") and os.path.exists(job["src"]):
            try:
                os.remove(job["src"])  # keep uploads dir clean
            except OSError:
                pass


@app.post("/api/video")
def api_video():
    if _active_job():
        return jsonify(ok=False, error="Another video analysis is already running — please wait."), 409
    f = request.files.get("video")
    if f is None:
        return jsonify(ok=False, error="No video uploaded"), 400
    ext = os.path.splitext(f.filename or "")[1].lower()
    if ext not in ALLOWED_VIDEO_EXT:
        return jsonify(ok=False, error=f"Unsupported format '{ext}'. Use mp4 / mov / avi / mkv / webm."), 400

    job_id = uuid.uuid4().hex[:12]
    src = os.path.join(UPLOADS, f"{job_id}{ext}")
    f.save(src)
    job = {
        "id": job_id, "status": "running", "progress": 0.0, "frames": 0, "total": 0,
        "violations": [], "created": time.time(), "src": src, "elapsed": 0.0,
        "zone": json.loads(request.form.get("zone", "[0.25,0.33,0.5,0.67]")),
        "dwell": float(request.form.get("dwell", 3.0)),
        "conf": float(request.form.get("conf", 0.5)),
        "filename": f.filename,
    }
    JOBS[job_id] = job
    _prune_jobs()
    threading.Thread(target=_run_video_job, args=(job,), daemon=True).start()
    return jsonify(ok=True, job_id=job_id)


@app.get("/api/video/<job_id>")
def api_video_status(job_id):
    j = JOBS.get(job_id)
    if j is None:
        return jsonify(ok=False, error="Unknown job"), 404
    out = {k: v for k, v in j.items() if k != "src"}
    return jsonify(ok=True, job=out)


@app.get("/api/violations")
def api_violations():
    rows = recent(100)
    items = [{
        "id": r[0], "ts": r[1], "track_id": r[2], "reason": r[3],
        "snapshot": r[4], "conf": r[5],
        "snapshot_url": f"/media/{r[4].replace('media/', '', 1)}" if r[4] else None,
    } for r in rows]
    return jsonify(ok=True, violations=items)


@app.errorhandler(413)
def too_large(_e):
    return jsonify(ok=False, error="File too large (max 200 MB)"), 413


if __name__ == "__main__":
    print("\n  HARDHAT VISION — helmet detection demo")
    load_model()
    print("  model:", os.path.basename(MODEL_PATH))
    print("  device:", "CUDA GPU" if _cuda() else "CPU")
    print("  open:  http://localhost:8000\n")
    app.run(host="127.0.0.1", port=8000, threaded=True, debug=False)
