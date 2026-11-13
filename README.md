# TrafficMind AI

Real-time urban traffic monitoring with computer vision and reinforcement learning, built as my bachelor's thesis at the Bucharest University of Economic Studies (*Urban Traffic Monitoring Application for Bucharest*).

TrafficMind takes a video feed (camera, file, RTSP or YouTube stream), detects and counts vehicles with **YOLOv8**, works out whether the camera is looking at an intersection or a single-flow road, and — at intersections — runs a **Q-learning agent** that picks the traffic-light phase that keeps waiting times lowest. Operators follow everything from a web dashboard with a live annotated stream, congestion stats, email alerts and downloadable PDF reports.

Everything runs on a regular CPU: no GPU, no cloud services.

---

## Features

**Detection & counting**
- Vehicle and pedestrian detection with YOLOv8 (cars, motorbikes, people), with per-class confidence thresholds
- ONNX-exported `yolov8s` model for faster CPU inference, with automatic fallback to the `.pt` model
- YOLO runs on every 3rd frame; skipped frames reuse the last result to keep the stream smooth
- IoU-based box persistence to stop detections from flickering between frames
- 5-frame moving average on counts and a simple congestion score (`vehicles / 25 × 100`, capped at 100%)

**Scene classification & signal optimisation**
- Automatic scene classification: for 10 seconds after a source change, vehicle motion vectors are binned by angle. If ≥80% fall within one 135° arc the scene is `single_flow`, otherwise `intersection` (heuristic, not a trained model)
- Tabular Q-learning agent for 4 signal phases (N-S green, E-W green, N-S protected left, E-W protected left)
- State: queue level in each of the 4 directions (4 bins) + current phase → 1,024 states
- Reward: −1 per waiting vehicle, +2 per vehicle cleared, −0.5 × queue length
- Phase limits: minimum 5 s, maximum 60 s per phase

**Dashboard & administration**
- Live MJPEG video stream with colour-coded bounding boxes
- Live counts by direction, congestion level and RL agent status (phase, Q-values, epsilon, reward)
- Login with roles (`admin` / `user`), bcrypt password hashing and CSRF protection
- User management, runtime video-source switching and video upload (admin only)

**Data, reports & alerts**
- SQLite logging every 10 s
- Export to CSV and JSON
- PDF report (ReportLab + matplotlib): KPIs, congestion and vehicle-mix charts, RL agent performance, recent records
- Email alerts over SMTP when congestion stays above a threshold (default 85% for 3 consecutive readings, 5-minute cooldown)

---

## How it works

```
Video source ──► Frame capture thread ──► queue (max 2 frames, old frames dropped)
                                              │
                                              ▼
                                   YOLOv8 detection (every 3rd frame)
                                              │
                                   IoU smoothing + moving average
                                              │
             ┌────────────────────────────────┼─────────────────────────────┐
             ▼                                ▼                             ▼
   MJPEG stream to dashboard      Scene classifier → Q-learning     SQLite log (10 s) → email alerts
                                  agent picks signal phase           PDF / CSV / JSON exports
```

Video capture and inference run on separate threads with a bounded queue, so when inference is slower than the source the system drops old frames instead of falling behind.

---

## RL benchmark (simulated)

The agent was trained offline for 1,000 episodes in a simple built-in intersection simulator (Poisson vehicle arrivals per direction, capped queues, limited discharge rate), then compared against a fixed-time controller that cycles through the 4 phases every 30 steps. Results over 50 evaluation episodes per scenario:

| Scenario | Avg. wait — RL | Avg. wait — fixed-time | Reduction | Throughput — RL | Throughput — fixed-time |
|---|---:|---:|---:|---:|---:|
| Low    |   228.8 |  2,956.3 | 92.3% |   352.9 |   345.7 |
| Medium |   490.9 |  5,681.2 | 91.4% |   602.7 |   588.9 |
| Peak   | 1,184.7 | 12,300.2 | 90.4% | 1,000.5 |   975.8 |

> **Note:** these numbers come from a simplified simulator against a basic fixed-time baseline. They show that the agent learns a better policy than a fixed cycle; they are not a prediction of real-world performance. Validation with a traffic simulator such as SUMO, or against actuated signal control, would be the next step.

---

## Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.10+ |
| Web | Flask 3, Flask-Login, Flask-WTF, Flask-Bcrypt |
| Computer vision | Ultralytics YOLOv8, ONNX, OpenCV |
| Reinforcement learning | NumPy (tabular Q-learning) |
| Storage | SQLite |
| Reports | ReportLab, matplotlib |
| Video input | OpenCV, yt-dlp (YouTube streams) |
| Alerts | smtplib (SSL 465 / STARTTLS 587) |

---

## Getting started

```bash
git clone https://github.com/Raul1210/trafficmind-ai.git
cd trafficmind-ai
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Model (optional but recommended):** export the faster ONNX model once:

```bash
python export_onnx.py
```

If no ONNX file is found, the app falls back to `yolov8m.pt` in the project folder.

**Video source:** by default the app looks for `data/intersectie_romana.mp4` (not included in the repo). Either place a video there, set `VIDEO_SOURCE_PATH` (file path, camera index, RTSP or YouTube URL), or upload/switch the source from the dashboard as admin.

**Run:**

```bash
python run.py
```

Open http://127.0.0.1:5000/ in your browser.

On the first run with an empty database the app creates a default account `admin` / `admin123`. **Change this password straight away**, and set a fixed `SECRET_KEY` for anything beyond local testing.

### Configuration

All settings are read from environment variables with sensible defaults. See [`.env.example`](.env.example) for the full list (host/port, model paths, detection thresholds, congestion threshold, logging interval, etc.). The app reads real environment variables; it does not load the `.env` file by itself, so export the values in your shell or use a tool that loads `.env` files.

### Training and benchmarking the agent

```bash
# Train on a scenario (low | medium | peak)
python -m traffic_monitor.agent.q_learning --mode train --scenario medium

# Compare RL vs fixed-time on all scenarios → benchmark_results.json
python -m traffic_monitor.agent.q_learning --mode benchmark
```

---

## Project structure

```
traffic_monitor/
├── __init__.py          # create_app(): Flask setup, blueprints, background system
├── config.py            # Settings from environment variables
├── core/                # Video capture threads, system state, SQLite logging
├── detection/           # YOLO detector, IoU smoothing, scene classifier
├── agent/               # Q-learning agent, intersection simulator, benchmark
├── api/                 # Dashboard routes and JSON API
├── auth/                # Login, roles, user management
├── alerts/              # Email alerts
└── reports/             # PDF report generator
templates/               # Dashboard, login, admin and DB views
run.py                   # Entry point
export_onnx.py           # One-off YOLOv8s → ONNX export
q_table_*_final.json     # Trained Q-tables (low / medium / peak)
benchmark_results.json   # RL vs fixed-time results
```

---

## Main API endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Dashboard |
| `GET /video_feed` | Live annotated MJPEG stream |
| `GET /stats` | Smoothed counts and congestion |
| `GET /api/counts` | Counts by direction (N/S/E/W) |
| `GET /api/signal` | Current RL state: phase, Q-values, epsilon, reward |
| `GET /api/benchmark` | RL vs fixed-time results |
| `GET /api/report/pdf` | Download PDF report (admin) |
| `GET /export/csv`, `/export/json` | Export logged data (admin) |
| `POST /api/video-source/change` | Switch video source (admin) |

All routes require login.

---

## Limitations

- The Q-learning agent was trained on a simplified simulator, not on real traffic data.
- N/S/E/W directions are frame quadrants, not real compass directions.
- The congestion score is a linear function of the vehicle count; it doesn't account for speed or vehicle size.
- There is no multi-object tracking with persistent IDs (such as SORT/DeepSORT); smoothing is frame-to-frame only.
- The app runs as a single process and handles one camera at a time.

---

## Author

**Raul Mihai Pociovălișteanu** — Economic Informatics, Bucharest University of Economic Studies (2026)

Built with the help of AI-assisted development tools (Claude Code).
