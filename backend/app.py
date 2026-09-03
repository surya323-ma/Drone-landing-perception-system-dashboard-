# dev/creator=tubakhxn
"""
Dashboard API for the Drone Landing Perception System.

Wraps main.py's process_video() / run_demo() in background jobs so the React
dashboard can:
  - upload a video and get it processed through the real perception pipeline
  - trigger the synthetic demo pipeline
  - poll job progress
  - stream the collected per-frame telemetry
  - stream/download the rendered output video

Run:
    pip install -r requirements.txt
    python app.py
Serves on http://localhost:5000
"""

import os
import sys
import time
import uuid
import threading
import traceback

from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename

# main.py lives one directory up (project root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import main as pipeline  # noqa: E402  (project's perception pipeline)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_ROOT, "uploads")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_TELEMETRY_SAMPLES = 4000  # safety cap so a very long video can't blow up memory

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload cap

# ----------------------------------------------------------------------------
# In-memory job store. Fine for a single-instance dashboard/demo deployment;
# swap for redis/db if this ever needs to run behind multiple workers.
# ----------------------------------------------------------------------------
JOBS = {}
JOBS_LOCK = threading.Lock()


def _new_job(kind, source_name=None):
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "kind": kind,             # "upload" | "demo"
            "source_name": source_name,
            "status": "queued",       # queued -> processing -> done | error
            "frame": 0,
            "total_frames": 0,
            "progress_pct": 0.0,
            "state": "SEARCHING",
            "confidence": 0.0,
            "error": None,
            "video_path": None,
            "created_at": time.time(),
            "telemetry": [],
        }
    return job_id


def _update_job(job_id, **kwargs):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


def _append_telemetry(job_id, record):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        buf = job["telemetry"]
        buf.append(record)
        # Downsample in place once we exceed the cap, keeping it bounded
        # while preserving an even spread across the whole run.
        if len(buf) > MAX_TELEMETRY_SAMPLES:
            job["telemetry"] = buf[::2]


def _run_upload_job(job_id, input_path):
    try:
        _update_job(job_id, status="processing")

        def on_progress(frame_idx, total_frames, state, conf):
            pct = (frame_idx / total_frames * 100.0) if total_frames else 0.0
            _update_job(job_id, frame=frame_idx, total_frames=total_frames,
                        progress_pct=round(pct, 1), state=state,
                        confidence=round(float(conf) * 100.0, 1))

        def on_telemetry(record):
            _append_telemetry(job_id, record)

        out_path = pipeline.process_video(
            input_path, progress_cb=on_progress, telemetry_cb=on_telemetry)
        _update_job(job_id, status="done", video_path=out_path, progress_pct=100.0)
    except SystemExit:
        _update_job(job_id, status="error", error="Could not process the uploaded video.")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _update_job(job_id, status="error", error=str(e))


def _run_demo_job(job_id):
    try:
        _update_job(job_id, status="processing")

        def on_progress(frame_idx, total_frames, state, conf):
            pct = (frame_idx / total_frames * 100.0) if total_frames else 0.0
            _update_job(job_id, frame=frame_idx, total_frames=total_frames,
                        progress_pct=round(pct, 1), state=state,
                        confidence=round(float(conf) * 100.0, 1))

        def on_telemetry(record):
            _append_telemetry(job_id, record)

        out_path = pipeline.run_demo(progress_cb=on_progress, telemetry_cb=on_telemetry)
        _update_job(job_id, status="done", video_path=out_path, progress_pct=100.0)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _update_job(job_id, status="error", error=str(e))


def _job_public(job):
    """Job dict without the (potentially large) telemetry buffer."""
    return {k: v for k, v in job.items() if k != "telemetry"}


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return jsonify({"ok": True, "service": "drone-perception-dashboard"})


@app.post("/api/jobs/upload")
def create_upload_job():
    if "video" not in request.files:
        return jsonify({"error": "No 'video' file in request."}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type .{ext}"}), 400

    job_id = _new_job("upload", source_name=f.filename)
    safe_name = f"{job_id}_{secure_filename(f.filename)}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    f.save(save_path)

    t = threading.Thread(target=_run_upload_job, args=(job_id, save_path), daemon=True)
    t.start()
    return jsonify({"job_id": job_id}), 202


@app.post("/api/jobs/demo")
def create_demo_job():
    job_id = _new_job("demo", source_name="synthetic demo")
    t = threading.Thread(target=_run_demo_job, args=(job_id,), daemon=True)
    t.start()
    return jsonify({"job_id": job_id}), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            abort(404)
        return jsonify(_job_public(job))


@app.get("/api/jobs/<job_id>/telemetry")
def job_telemetry(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            abort(404)
        return jsonify({"job_id": job_id, "count": len(job["telemetry"]),
                         "telemetry": job["telemetry"]})


@app.get("/api/jobs/<job_id>/video")
def job_video(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            abort(404)
        video_path = job.get("video_path")
    if not video_path or not os.path.isfile(video_path):
        abort(404)
    return send_file(video_path, mimetype="video/mp4", conditional=True)


@app.get("/api/jobs")
def list_jobs():
    with JOBS_LOCK:
        jobs = sorted((_job_public(j) for j in JOBS.values()),
                      key=lambda j: j["created_at"], reverse=True)
    return jsonify(jobs[:50])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
