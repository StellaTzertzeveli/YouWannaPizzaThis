import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LIMB_INDICES = {
    "left_wrist": 16,
    "right_wrist": 15,
    # "left_elbow": 13,
    # "right_elbow": 14,
    # "left_knee": 25,
    # "right_knee": 26,
    # "left_ankle": 27,q
    # "right_ankle": 28,
}

# 1. Initialize Global Variable for Storing Latest Landmarks
latest_landmarks = None


# 2. Callback Function for MediaPipe Live Stream Mode
def print_result(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_landmarks
    if result.pose_landmarks:
        # Save the list of 33 pose landmarks for the first detected person
        latest_landmarks = result.pose_landmarks[0]


# 3. Configure Pose Landmarker
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_result
)

# 4. Open Webcam Capture
cap = cv2.VideoCapture(0)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    print("Camera running! Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # A. Preprocessing: Mirror image horizontally & convert BGR to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # B. Convert NumPy frame to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # C. Send frame asynchronously with timestamp in milliseconds
        frame_timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # D. Process Detected Body Landmarks (Right Wrist Index = 16)
        if latest_landmarks is not None:
            for limb_str, limb_idx in LIMB_INDICES.items():
                limb = latest_landmarks[limb_idx]

                # Map normalized coordinates (0.0 - 1.0) to screen frame dimensions
                h, w, _ = frame.shape
                cx, cy = int(limb.x * w), int(limb.y * h)

                # Draw a circle on top of the right wrist in OpenCV window
                cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)
                cv2.putText(frame, f"{limb_str}: ({cx}, {cy})", (cx + 20, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # E. Display Camera Feed
        cv2.imshow("MediaPipe Pose Tracking", frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()