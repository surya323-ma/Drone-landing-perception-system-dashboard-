# Perception Dashboard (new feature)

A React + Flask web dashboard bolted onto the existing perception pipeline
in `main.py`. It lets you upload drone footage (or run the built-in
synthetic demo) from a browser, watches the rendered HUD output as it's
produced, and visualizes the same telemetry the HUD draws on-frame — but
as live readouts, trend charts, and a tracking-state timeline.

```
Drone-Landing-Perception-System/
│
├── main.py                # unchanged CLI pipeline, now with optional
│                           # progress_cb / telemetry_cb hooks used by the API
├── main.cpp                # unchanged Kalman-filter HUD demo
├── requirements.txt
│
├── backend/                # Flask API — wraps main.py's process_video()/run_demo()
│   ├── app.py
│   ├── requirements.txt
│   ├── uploads/             # user-submitted videos land here
│   └── output/              # created at runtime by main.py's ensure_dirs()
│
└── frontend/                # React + Vite + Tailwind + Recharts dashboard
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── src/
        ├── App.jsx
        ├── api.js
        └── components/
            ├── Header.jsx
            ├── UploadPanel.jsx
            ├── VideoStage.jsx
            ├── HudTelemetryPanel.jsx
            ├── TelemetryCharts.jsx
            └── StateTimeline.jsx
```

## What changed in `main.py`

`process_video()` and `run_demo()` now accept two **optional** keyword
arguments — `progress_cb` and `telemetry_cb` — used only by the dashboard
API. `main()` (the CLI entry point) doesn't pass them, so
`python main.py` / `python main.py input.mp4` behave exactly as before.

- `progress_cb(frame_idx, total_frames, state, confidence)` — called
  periodically while a video is processed.
- `telemetry_cb(record)` — called once per frame with a compact,
  JSON-serializable dict (state, confidence, distance, altitude, approach
  angle, alignment, lateral/vertical offset, landing confidence, etc.) —
  the same values the on-frame HUD renders.

## Running it

**1. Backend (Flask API)**

```bash
cd backend
pip install -r requirements.txt --break-system-packages
python app.py
```

Serves on `http://localhost:5000`. Endpoints:

| Method | Path                        | Purpose                                   |
|--------|-----------------------------|--------------------------------------------|
| POST   | `/api/jobs/upload`          | multipart `video` file → starts a job      |
| POST   | `/api/jobs/demo`            | starts the synthetic demo job              |
| GET    | `/api/jobs/<id>`            | job status/progress                        |
| GET    | `/api/jobs/<id>/telemetry`  | full per-frame telemetry array so far       |
| GET    | `/api/jobs/<id>/video`      | streams the rendered output video           |

Jobs run in a background thread per request; state lives in memory (fine
for a single dashboard instance — swap in Redis/a DB if you need multiple
workers or persistence across restarts).

**2. Frontend (React dashboard)**

```bash
cd frontend
npm install
npm run dev
```

Opens on `http://localhost:5173` and proxies `/api/*` to the Flask
backend (see `vite.config.js`). Drop in a video or hit **Run Synthetic
Demo**, then watch the progress bar, the rendered HUD video, the live
telemetry panel (synced to video playback time), the confidence/alignment
and distance/altitude trend charts, and the tracking-state timeline strip
below the player (click it to seek).

**Production build:** `npm run build` in `frontend/` outputs static files
in `frontend/dist/` — serve them with any static host, or wire Flask to
serve them directly if you want a single deployable service.

## Design notes

The dashboard's palette (`tailwind.config.js` → `hud.*` colors) is lifted
directly from the perception system's own on-frame HUD palette (cyan
`#3CF0FF`, amber `#EBAA3C`, panel navy `#12181F`) so it reads as the same
instrument continued into a browser rather than a generic admin UI bolted
on top — corner-bracket frames, monospace telemetry readouts, and a
scanline texture echo the HUD's own visual language from `main.py` /
`main.cpp`.
