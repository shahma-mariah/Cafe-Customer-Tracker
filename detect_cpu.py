"""
Cafe Footfall Tracker — CPU Version
====================================
Detects people entering a cafe using a virtual entry line and logs
peak-hour analytics. Runs entirely on CPU using a lightweight YOLOv8n model.

Usage:
    # Live CCTV stream (RTSP or webcam index)
    python detect_cpu.py --source 0                          # webcam
    python detect_cpu.py --source "rtsp://ip:port/stream"   # RTSP camera

    # Pre-recorded video
    python detect_cpu.py --source "data/recordings/cafe.mp4"

    # Optional flags
    python detect_cpu.py --source 0 --entry-line 0.5 --show  # draw GUI window
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Dependency check — give a clear message if ultralytics is missing
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit(
        "ultralytics not found. Install it with:\n"
        "    pip install ultralytics"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "yolov8n.pt"          # nano — fastest on CPU (~6 MB)
PERSON_CLASS_ID = 0                 # COCO class 0 = person
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.5
FRAME_SKIP = 2                      # process every Nth frame to save CPU


# ---------------------------------------------------------------------------
# Tracker — simple centroid-based, no external dep needed
# ---------------------------------------------------------------------------
class CentroidTracker:
    """
    Lightweight centroid tracker that assigns persistent IDs to detections.
    Objects that disappear for > max_disappeared frames are de-registered.
    """

    def __init__(self, max_disappeared: int = 30):
        self.next_id = 0
        self.objects: dict[int, np.ndarray] = {}      # id -> centroid
        self.disappeared: dict[int, int] = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid: np.ndarray) -> None:
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, obj_id: int) -> None:
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, rects: list[tuple]) -> dict[int, np.ndarray]:
        """
        rects: list of (x1, y1, x2, y2) bounding boxes.
        Returns dict of {id: centroid}.
        """
        if not rects:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        input_centroids = np.array(
            [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in rects],
            dtype="int",
        )

        if not self.objects:
            for c in input_centroids:
                self.register(c)
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_centroids = np.array(list(self.objects.values()))

        # Compute pairwise distances
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

        unused_rows = set(range(len(obj_centroids))) - used_rows
        unused_cols = set(range(len(input_centroids))) - used_cols

        for row in unused_rows:
            oid = obj_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects


# ---------------------------------------------------------------------------
# Entry counter
# ---------------------------------------------------------------------------
class EntryCounter:
    """
    Counts a person as 'entered' when their centroid crosses the entry line
    moving in a downward direction (top-of-frame → bottom = entering cafe).
    Adjust direction logic based on your camera orientation.
    """

    def __init__(self, line_y: int):
        self.line_y = line_y
        self.prev_positions: dict[int, int] = {}   # id -> previous y centroid
        self.counted_ids: set[int] = set()
        self.total_count = 0

    def update(self, tracked: dict[int, np.ndarray]) -> int:
        """Returns number of NEW entries detected this frame."""
        new_entries = 0
        for oid, centroid in tracked.items():
            cy = int(centroid[1])
            prev_cy = self.prev_positions.get(oid)

            if prev_cy is not None and oid not in self.counted_ids:
                # Crossed line downward (entering)
                if prev_cy < self.line_y <= cy:
                    self.counted_ids.add(oid)
                    self.total_count += 1
                    new_entries += 1

            self.prev_positions[oid] = cy

        # Clean up lost tracks
        active_ids = set(tracked.keys())
        stale = set(self.prev_positions) - active_ids
        for oid in stale:
            del self.prev_positions[oid]

        return new_entries


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------
class FootfallLogger:
    def __init__(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = LOG_DIR / f"footfall_{ts}.csv"
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "hour", "cumulative_count", "entries_this_minute"])
        self._minute_bucket: str = ""
        self._minute_count: int = 0
        print(f"[Logger] Saving to {self.path}")

    def log(self, cumulative: int, new_entries: int) -> None:
        now = datetime.now()
        bucket = now.strftime("%Y-%m-%d %H:%M")
        if bucket != self._minute_bucket:
            if self._minute_bucket:
                self._flush(self._minute_bucket, cumulative - new_entries, self._minute_count)
            self._minute_bucket = bucket
            self._minute_count = 0
        self._minute_count += new_entries

    def _flush(self, bucket: str, cumulative: int, minute_count: int) -> None:
        hour = int(bucket.split(" ")[1].split(":")[0])
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([bucket, hour, cumulative, minute_count])

    def close(self, cumulative: int) -> None:
        if self._minute_bucket:
            self._flush(self._minute_bucket, cumulative, self._minute_count)
        print(f"[Logger] Log saved → {self.path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(source, entry_line_ratio: float = 0.5, show: bool = False) -> None:
    print("[CPU] Loading YOLOv8n model …")
    model = YOLO(MODEL_NAME)
    model.to("cpu")

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    entry_y = int(frame_h * entry_line_ratio)

    print(f"[CPU] Stream {frame_w}×{frame_h} @ {fps_src:.1f} fps")
    print(f"[CPU] Entry line at y={entry_y} ({entry_line_ratio*100:.0f}% from top)")
    print("[CPU] Press 'q' to quit.\n")

    tracker = CentroidTracker(max_disappeared=40)
    counter = EntryCounter(line_y=entry_y)
    logger = FootfallLogger()

    frame_idx = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[CPU] Stream ended.")
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            # ── Detection ──────────────────────────────────────────────────
            results = model(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False,
            )

            boxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2))

            # ── Tracking & counting ────────────────────────────────────────
            tracked = tracker.update(boxes)
            new_entries = counter.update(tracked)
            logger.log(counter.total_count, new_entries)

            # ── Overlay ────────────────────────────────────────────────────
            if show:
                for x1, y1, x2, y2 in boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for oid, centroid in tracked.items():
                    cx, cy = int(centroid[0]), int(centroid[1])
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(oid), (cx - 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Entry line
                cv2.line(frame, (0, entry_y), (frame_w, entry_y), (255, 0, 0), 2)
                cv2.putText(frame, "ENTRY LINE", (10, entry_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Stats HUD
                elapsed = time.time() - t_start
                cv2.rectangle(frame, (0, 0), (280, 60), (0, 0, 0), -1)
                cv2.putText(frame, f"Total entries: {counter.total_count}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(frame, f"People visible: {len(boxes)}  |  {elapsed:.0f}s",
                            (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                cv2.imshow("Cafe Footfall Tracker [CPU]", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[CPU] Quit requested.")
                    break

            # Console heartbeat every 100 processed frames
            if (frame_idx // FRAME_SKIP) % 100 == 0:
                print(f"  Frame {frame_idx:6d} | Entries so far: {counter.total_count}")

    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
        logger.close(counter.total_count)
        print(f"\n[CPU] Done. Total people entered: {counter.total_count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cafe Footfall Tracker — CPU")
    parser.add_argument("--source", default="0",
                        help="Video source: webcam index (0,1…), RTSP URL, or video file path")
    parser.add_argument("--entry-line", type=float, default=0.5,
                        help="Fractional position of entry line from top (0.0–1.0). Default 0.5")
    parser.add_argument("--show", action="store_true",
                        help="Show live annotated window (disable for headless servers)")
    args = parser.parse_args()
    run(args.source, args.entry_line, args.show)
