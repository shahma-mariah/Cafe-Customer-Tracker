import cv2
from ultralytics import YOLO

# Load the Nano model (fastest for CPU)
model = YOLO('yolov8n.pt') 

# CHANGE THIS: Use the RTSP URL of the camera for webcam or "video.mp4" for your file 
video_source = "cafe_video.mp4" 
cap = cv2.VideoCapture(video_source)

# Performance: Downscale video for smoother CPU processing
FRAME_WIDTH = 640

# --- CALIBRATION ---
# 0.7 = 70% down the screen. Change to 0.5 for middle.
LINE_Y = 0.7 
# -------------------

counter = 0
tracked_ids = set()

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Resize for speed on AMD Ryzen 3
    frame = cv2.resize(frame, (FRAME_WIDTH, int(frame.shape[0] * (FRAME_WIDTH / frame.shape[1]))))
    h, w, _ = frame.shape
    line_pos = int(h * LINE_Y)

    # Run Tracking (Force CPU)
    results = model.track(frame, persist=True, classes=0, device='cpu', verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int(y2) # Track the feet

            # Crossing Logic
            if cy > line_pos and obj_id not in tracked_ids:
                tracked_ids.add(obj_id)
                counter += 1

            # Visuals
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Draw Line and Count
    cv2.line(frame, (0, line_pos), (w, line_pos), (0, 0, 255), 2)
    cv2.putText(frame, f"CPU Count: {counter}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Cafe Tracker (CPU)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
