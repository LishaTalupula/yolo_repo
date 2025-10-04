from ultralytics import YOLO
import cv2
from datetime import datetime, timedelta
 
# Load model
model = YOLO('RPi_device_agent/camera_stream/models/construction-model.pt')
 
# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 20)
 
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Requested FPS: 20, Actual FPS: {actual_fps:.2f}")
print("Press 'q' to quit.")
 
# === Configuration ===
CONFIDENCE_THRESHOLD = 0.5
ENTRY_CONFIRM_DURATION = 1.0    # seconds to confirm initial entry
EXIT_GRACE_DURATION = 5.0       # seconds to wait before logging exit
 
# === State Variables ===
session_active = False          # Is a person session ongoing?
entry_time = None               # When the session started
last_seen_time = None           # Last time person was confidently detected
person_detected_start = None    # For initial entry debounce
 
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame.")
        break
 
    results = model(frame, verbose=False)
    
    # Check for valid 'Person' detection (confidence > 0.5)
    current_person_detected = False
    for box in results[0].boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        if cls_name == 'Person' and conf > CONFIDENCE_THRESHOLD:
            current_person_detected = True
            break
 
    current_time = datetime.now()
 
    # === Handle ongoing session ===
    if session_active:
        if current_person_detected:
            # Person is still present → update last seen
            last_seen_time = current_time
        else:
            # Person not seen → check if grace period expired
            if (current_time - last_seen_time).total_seconds() >= EXIT_GRACE_DURATION:
                # Grace period over → log exit
                exit_time = current_time
                print(f"[EXIT] Person exited at: {exit_time}")
                print(f"  → Duration: {exit_time - entry_time}")
                print(f"  → Entry: {entry_time}, Exit: {exit_time}")
                session_active = False
                entry_time = None
                last_seen_time = None
                person_detected_start = None
            # else: still in grace period → do nothing (session continues)
 
    else:
        # No active session → check for new entry
        if current_person_detected:
            if person_detected_start is None:
                person_detected_start = current_time
            else:
                if (current_time - person_detected_start).total_seconds() >= ENTRY_CONFIRM_DURATION:
                    # Confirm new session
                    session_active = True
                    entry_time = current_time
                    last_seen_time = current_time
                    person_detected_start = None  # reset
                    print(f"[ENTRY] Person confirmed at: {entry_time}")
        else:
            # Reset debounce if no detection
            person_detected_start = None
 
    # Show frame
    annotated_frame = results[0].plot()
    cv2.imshow("Person Detection", annotated_frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Final exit on quit
if session_active:
    exit_time = datetime.now()
    print(f"[FINAL EXIT] Person exited at: {exit_time}")
    print(f"  → Entry: {entry_time}, Exit: {exit_time}")
    print(f"  → Duration: {exit_time - entry_time}")
 
cap.release()
cv2.destroyAllWindows()