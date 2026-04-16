# 🏪 Cafe Customer Tracker

A lightweight computer vision system that uses a CCTV camera feed to detect people entering a cafe, log entry events and analyse peak hours. Built as an internship project demonstrating real-world applied ML on edge hardware.

---

## 📌 Project Overview

| Feature | Details |
|---|---|
| Detection model | YOLOv8 (Ultralytics) |
| Tracking | Custom centroid tracker (no external dep) |
| Entry logic | Virtual line crossing (top → bottom = entering) |
| Input sources | Live CCTV (RTSP / webcam) **and** pre-recorded video |
| Output | Per-minute CSV log + hourly bar chart + timeline chart |
| Hardware | **CPU mode** and **GPU (CUDA) mode** — pick the right script for your setup |

---

## 🗂️ Repository Structure

```
cafe-footfall-tracker/
│
├── detect_cpu.py        ← Run this on CPU / low-power devices
├── detect_gpu.py        ← Run this if a CUDA GPU is available
├── analytics.py         ← Generate peak-hour charts from saved logs
│
├── data/
│   ├── logs/            ← Auto-generated CSV logs (gitignored by default)
│   └── recordings/      ← Place your .mp4 test videos here
│
├── outputs/             ← Chart images saved here after running analytics.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/cafe-footfall-tracker.git
cd cafe-footfall-tracker
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> The YOLOv8 model weights (`yolov8n.pt` / `yolov8s.pt`) are **downloaded automatically** on first run.

---

## 🖥️ CPU Mode — `detect_cpu.py`

Use this on **any machine** — laptops, Raspberry Pi, NUC, or any server **without** a dedicated GPU.

- Model: **YOLOv8n** (nano, ~6 MB) — fastest on CPU
- Frame skip: every 2nd frame to reduce load
- Recommended for: low-power edge devices, development/testing

### Run on a **webcam**
```bash
python detect_cpu.py --source 0 --show
```

### Run on a **live RTSP CCTV stream**
```bash
python detect_cpu.py --source "rtsp://admin:password@192.168.1.64:554/stream1" --show
```

### Run on a **pre-recorded video file**
```bash
python detect_cpu.py --source "data/recordings/cafe_footage.mp4" --show
```

### Run **headless** (no GUI window — for servers)
```bash
python detect_cpu.py --source "rtsp://..." 
```

### Adjust the entry line position
```bash
# Entry line at 40% from the top of frame (default is 50%)
python detect_cpu.py --source 0 --entry-line 0.4 --show
```

---

## 🚀 GPU Mode — `detect_gpu.py`

Use this if the client machine has an **NVIDIA GPU with CUDA**.

- Model: **YOLOv8s** (small) — better accuracy, GPU handles it comfortably
- Batch inference: multiple frames processed simultaneously
- Typically **5–10× faster** than CPU mode

### Extra install step (one-time)
```bash
# CUDA 12.1 — adjust cu version to match your driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Usage (same flags as CPU version, plus `--batch-size`)
```bash
# Webcam
python detect_gpu.py --source 0 --show

# RTSP stream
python detect_gpu.py --source "rtsp://admin:password@192.168.1.64:554/stream1"

# Pre-recorded video
python detect_gpu.py --source "data/recordings/cafe_footage.mp4" --show

# Larger batch for high-end GPUs
python detect_gpu.py --source 0 --batch-size 8 --show
```

> **No GPU?** `detect_gpu.py` automatically falls back to CPU with a warning, so it won't crash — but `detect_cpu.py` is better optimised for that case.

---

## 📊 Analytics — `analytics.py`

After running the detector (or with pre-existing logs), generate peak-hour charts:

```bash
# Use the most recent log automatically
python analytics.py

# Use a specific log file
python analytics.py --log data/logs/footfall_20240615_090000.csv

# Merge ALL logs for a combined report
python analytics.py --all
```

**Outputs saved to `outputs/`:**
- `peak_hours.png` — hourly bar chart with peak hour highlighted
- `timeline.png` — per-minute entry timeline

**Console summary example:**
```
=============================================
  CAFE FOOTFALL SUMMARY
=============================================
  Total entries logged :  347
  Peak hour            : 12:00 – 13:00  (89 entries)
  Busiest 3 hours      :
      12:00 – 13:00  →  89 entries
      08:00 – 09:00  →  61 entries
      17:00 – 18:00  →  55 entries
=============================================
```

---

## 🔧 How the Entry Line Works

```
┌──────────────────────────────────────┐
│           Camera frame               │
│                                      │
│   Person walking toward cafe door    │
│           ↓ (moving down)            │
│ - - - - - - - - - - - - - - - - - -  │  ← Virtual entry line (y = 50%)
│                                      │
│   Person counted as "entered" ✓      │
└──────────────────────────────────────┘
```

A person is counted **once** when their tracked centroid crosses the line moving **downward** (top-to-bottom). Adjust `--entry-line` to match where the cafe entrance appears in your specific camera view.

---

## 🗒️ CSV Log Format

Logs are saved to `data/logs/footfall_YYYYMMDD_HHMMSS.csv`:

| Column | Description |
|---|---|
| `timestamp` | Minute bucket (YYYY-MM-DD HH:MM) |
| `hour` | Integer hour (0–23) for easy grouping |
| `cumulative_count` | Total entries since tracking started |
| `entries_this_minute` | New entries in this 1-minute window |

---

## 📋 Requirements

- Python 3.9+
- See `requirements.txt`
- For GPU mode: NVIDIA GPU + CUDA 11.8 or 12.x + matching PyTorch build

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot open source` | Check camera index or RTSP URL; try `--source 1` |
| Model not downloading | Check internet connection; weights auto-download via ultralytics |
| Low FPS on CPU | Increase `FRAME_SKIP` constant in `detect_cpu.py` |
| CUDA not detected | Verify `nvidia-smi` works; reinstall PyTorch with correct CUDA version |
| Duplicate counts | Lower `--entry-line` ratio or increase `max_disappeared` in tracker |

---

## 🔮 Future Improvements

- [ ] Exit counting (two-way direction tracking)
- [ ] Real-time dashboard (Streamlit / Gradio)
- [ ] Occupancy limit alerts
- [ ] Multi-camera support
- [ ] Export to Google Sheets or database

---

## 👤 Author

Built by an AI/ML Intern as a practical computer vision demonstration.  
Stack: Python · OpenCV · YOLOv8 · Pandas · Matplotlib
