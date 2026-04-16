"""
Cafe Footfall Tracker — GPU Version
=====================================
Same logic as detect_cpu.py but leverages CUDA for significantly higher
throughput. Recommended for clients with an NVIDIA GPU.

Requirements (in addition to requirements.txt):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Usage:
    # Live CCTV stream
    python detect_gpu.py --source 0                          # webcam
    python detect_gpu.py --source "rtsp://ip:port/stream"   # RTSP camera

    # Pre-recorded video
    python detect_gpu.py --source "data/recordings/cafe.mp4"

    # Optional flags
    python detect_gpu.py --source 0 --entry-line 0.5 --show --batch-size 4
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
except ImportError:
    raise SystemExit(
        "PyTorch not found. Install with:\n"
        "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    )

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("ultralytics not found. Install with: pip install ultralytics")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "yolov8s.pt"           # small — better accuracy, GPU handles it easily
PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.5
FRAME_SKIP = 1                      # GPU is fast enough to process every frame


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def select_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] CUDA device: {name}  ({vram:.1f} GB VRAM)")
        return "cuda"
    print("[GPU] WARNING: CUDA not available — falling back to CPU.")
    print("      Consider running detect_cpu.py instead for optimal settings.")
    return "cpu"


# ---------------------------------------------------------------------------
# Centroid tracker (same lightweight implementation as CPU version)
# ---------------------------------------------------------------------------
class CentroidTracker:
    def __init__(self, max_disappeared: int = 30):
        self.next_id = 0
        self.objects: dict[int, np.ndarray] = {}
        self.disappeared: dict[int, int] = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, rects):
        if not rects:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        input_centroids = np.array(
            [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in rects], dtype="int"
        )

        if not self.objects:
            for c in input_centroids:
                self.register(c)
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_centroids = np.array(list(self.objects.values()))
        D = np.linalg.norm(obj_centroids[:, None] - input_centroids[None], axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            oid = obj_ids[row]
            self.objects[oid] = input_centroids[col]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(len(obj_centroids))) - used_rows:
            oid = obj_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        for col in set(range(len(input_centroids))) - used_cols:
            self.register(input_centroids[col])

        return self.objects


# ---------------------------------------------------------------------------
# Entry counter
# ---------------------------------------------------------------------------
class EntryCounter:
    def __init__(self, line_y: int):
        self.line_y = line_y
        self.prev_positions: dict[int, int] = {}
        self.counted_ids: set[int] = set()
        self.total_count = 0

    def update(self, tracked: dict) -> int:
        new_entries = 0
        for oid, centroid in tracked.items():
            cy = int(centroid[1])
            prev_cy = self.prev_positions.get(oid)
            if prev_cy is not None and oid not in self.counted_ids:
                if prev_cy < self.line_y <= cy:
                    self.counted_ids.add(oid)
                    self.total_count += 1
                    new_entries += 1
            self.prev_positions[oid] = cy

        for oid in set(self.prev_positions) - set(tracked):
            del self.prev_positions[oid]

        return new_entries


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------
class FootfallLogger:
    def __init__(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = LOG_DIR / f"footfall_gpu_{ts}.csv"
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "hour", "cumulative_count", "entries_this_minute"])
        self._bucket = ""
        self._bucket_count = 0
        print(f"[Logger] Saving to {self.path}")

    def log(self, cumulative: int, new_entries: int):
        now = datetime.now()
        bucket = now.strftime("%Y-%m-%d %H:%M")
        if bucket != self._bucket:
            if self._bucket:
                self._flush(self._bucket, cumulative - new_entries, self._bucket_count)
            self._bucket = bucket
            self._bucket_count = 0
        self._bucket_count += new_entries

    def _flush(self, bucket, cumulative, count):
        hour = int(bucket.split(" ")[1].split(":")[0])
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([bucket, hour, cumulative, count])

    def close(self, cumulative):
        if self._bucket:
            self._flush(self._bucket, cumulative, self._bucket_count)
        print(f"[Logger] Log saved → {self.path}")


# ---------------------------------------------------------------------------
# GPU batch inference helper
# ---------------------------------------------------------------------------
def detect_batch(model, frames: list, device: str) -> list[list[tuple]]:
    """
    Run inference on a list of frames simultaneously.
    Returns a list of bounding-box lists, one per frame.
    """
    results = model(
        frames,
        classes=[PERSON_CLASS_ID],
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=device,
        verbose=False,
    )
    all_boxes = []
    for r in results:
        boxes = [tuple(map(int, b.xyxy[0].tolist())) for b in r.boxes]
        all_boxes.append(boxes)
    return all_boxes


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(source, entry_line_ratio: float = 0.5, show: bool = False, batch_size: int = 4) -> None:
    device = select_device()

    print(f"[GPU] Loading YOLOv8s model on {device} …")
    model = YOLO(MODEL_NAME)
    model.to(device)

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    entry_y = int(frame_h * entry_line_ratio)

    print(f"[GPU] Stream {frame_w}×{frame_h} @ {fps_src:.1f} fps")
    print(f"[GPU] Batch size: {batch_size}  |  Entry line y={entry_y}")
    print("[GPU] Press 'q' to quit.\n")

    tracker = CentroidTracker(max_disappeared=40)
    counter = EntryCounter(line_y=entry_y)
    logger = FootfallLogger()

    frame_idx = 0
    batch_buffer: list[np.ndarray] = []
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Flush remaining buffer
                if batch_buffer:
                    all_boxes = detect_batch(model, batch_buffer, device)
                    for boxes in all_boxes:
                        tracked = tracker.update(boxes)
                        new_entries = counter.update(tracked)
                        logger.log(counter.total_count, new_entries)
                print("[GPU] Stream ended.")
                break

            frame_idx += 1
            batch_buffer.append(frame.copy())

            if len(batch_buffer) < batch_size:
                # Also display live even while buffering
                if show:
                    cv2.imshow("Cafe Footfall Tracker [GPU]", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            # ── Batch inference ────────────────────────────────────────────
            all_boxes = detect_batch(model, batch_buffer, device)

            for i, (frm, boxes) in enumerate(zip(batch_buffer, all_boxes)):
                tracked = tracker.update(boxes)
                new_entries = counter.update(tracked)
                logger.log(counter.total_count, new_entries)

                if show:
                    for x1, y1, x2, y2 in boxes:
                        cv2.rectangle(frm, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    for oid, centroid in tracked.items():
                        cx, cy = int(centroid[0]), int(centroid[1])
                        cv2.circle(frm, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(frm, str(oid), (cx - 10, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    cv2.line(frm, (0, entry_y), (frame_w, entry_y), (255, 0, 0), 2)
                    cv2.putText(frm, "ENTRY LINE", (10, entry_y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    elapsed = time.time() - t_start
                    device_label = "CUDA" if device == "cuda" else "CPU"
                    cv2.rectangle(frm, (0, 0), (310, 60), (0, 0, 0), -1)
                    cv2.putText(frm, f"Total entries: {counter.total_count}",
                                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    cv2.putText(frm, f"Device: {device_label}  |  {elapsed:.0f}s",
                                (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

                    cv2.imshow("Cafe Footfall Tracker [GPU]", frm)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise KeyboardInterrupt

            batch_buffer.clear()

            if frame_idx % 200 == 0:
                fps_actual = frame_idx / (time.time() - t_start)
                print(f"  Frame {frame_idx:6d} | Entries: {counter.total_count} | "
                      f"Throughput: {fps_actual:.1f} fps")

    except KeyboardInterrupt:
        print("[GPU] Interrupted.")
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
        logger.close(counter.total_count)
        print(f"\n[GPU] Done. Total people entered: {counter.total_count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cafe Footfall Tracker — GPU")
    parser.add_argument("--source", default="0",
                        help="Video source: webcam index, RTSP URL, or video file path")
    parser.add_argument("--entry-line", type=float, default=0.5,
                        help="Entry line position as fraction from top (0.0–1.0)")
    parser.add_argument("--show", action="store_true",
                        help="Show annotated video window")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of frames per GPU inference batch (default 4)")
    args = parser.parse_args()
    run(args.source, args.entry_line, args.show, args.batch_size)
