import cv2
from ultralytics import YOLO

# Use the more accurate Medium model for GPU users
model = YOLO('yolov8m.pt') 
# CHANGE THIS: Use the RTSP URL of the camera for webcam or "video.mp4" for your file
video_source = "cafe_video.mp4"
cap = cv2.VideoCapture(video_source)

# GPU can handle full resolution
counter = 0
tracked_ids = set()

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    h, w, _ = frame.shape
    line_pos = int(h * 0.7)

    # Run Tracking (Force GPU - device 0)
    results = model.track(frame, persist=True, classes=0, device=0, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int(y2)

            if cy > line_pos and obj_id not in tracked_ids:
                tracked_ids.add(obj_id)
                counter += 1
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

    cv2.line(frame, (0, line_pos), (w, line_pos), (0, 255, 0), 3)
    cv2.putText(frame, f"GPU Count: {counter}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    cv2.imshow("Cafe Tracker (GPU)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
