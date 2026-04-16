# 🏪 Cafe Customer Tracker

A lightweight computer vision system that uses a CCTV camera feed to detect people entering a cafe, log entry events and analyse peak hours. Built as an internship project demonstrating real-world applied ML on edge hardware.

---

## 📌 Project Overview

| Feature | Details |
|---|---|
| Detection model | YOLOv8 (Ultralytics) |
| Tracking | Custom centroid tracker (no external dep) |
| Entry logic | Virtual line crossing |
| Input sources | Live CCTV (RTSP / webcam) **and** pre-recorded video |
| Output | Customer count |
| Hardware | **CPU mode** and **GPU (CUDA) mode** — pick the right script for your setup |

---

## 📋 Requirements

- Python 3.9+
- See `requirements.txt`
- For GPU mode: NVIDIA GPU + CUDA 11.8 or 12.x + matching PyTorch build

---

## 👤 Author

Built by an AI/ML Intern as a practical computer vision demonstration.  
Stack: Python · OpenCV · YOLOv8 · Pandas · Matplotlib
