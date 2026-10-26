import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import motion_detectors.ChopDetector as ChopDetector
import motion_detectors.CrossDetector as CrossDetector
import motion_detectors.GrabDetector as GrabDetector
import motion_detectors.ThumbsupDetector as ThumbsupDetector

display_pose = False
chop_amount = 0
hand_results = []
LIMB_INDICES = {15: 'left wrist', 16: 'right wrist'}

def hand_callback(result, output_image, timestamp_ms):
    global latest_hand_result
    latest_hand_result = result


def pose_callback(result, output_image, timestamp_ms):
    global latest_pose_result
    latest_pose_result = result


# ---------------------HAND LANDMARKER----------------------
hand_model_path = "hand_landmarker.task"

BaseOptions = python.BaseOptions
# 3. Configure Pose Landmarker

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=hand_model_path
    ),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=hand_callback
)
hand_landmarker = vision.HandLandmarker.create_from_options(
    hand_options
)

#-------------------------------pose landmarker--------------------------------
pose_model_path = "pose_landmarker.task"

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=pose_model_path
    ),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=pose_callback,
)

pose_landmarker = vision.PoseLandmarker.create_from_options(
    pose_options
)

# 4. Open Webcam Capture
cap = cv2.VideoCapture(0)
chopDetector = ChopDetector.ChopDetector()
crossDetector = CrossDetector.CrossDetector()
grabDetector = GrabDetector.GrabDetector('grab')
thumbsupDetector = ThumbsupDetector.ThumbsupDetector()
latest_hand_result = None
latest_pose_result = None
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

    # SEND RESULTS TO BOTH MODELS
    frame_timestamp_ms = int(time.time() * 1000)
    hand_landmarker.detect_async(
        mp_image,
        frame_timestamp_ms
    )

    pose_landmarker.detect_async(
        mp_image,
        frame_timestamp_ms
    )
    #HAND RESULTS
    if latest_hand_result is not None:
        for hand in latest_hand_result.hand_landmarks:
            chopDetector.update(hand)
            grabDetector.update(hand, should_print=True)
            thumbsupDetector.update(hand, should_print=True)
        # frame = grabDetector.display_bar_graph(frame)

    if latest_pose_result is not None:
        for body in latest_pose_result.pose_landmarks:
            crossDetector.update(body, should_print=True)

    frame = thumbsupDetector.display_bar_graph(frame)
    key_input = cv2.waitKey(1) & 0xFF
    if  key_input == ord('v'):
        thumbsupDetector.start_detection()
        print('started detection chopDetector')
    # E. Display Camera Feed
    cv2.imshow("MediaPipe Pose Tracking", frame)

    # Exit loop if 'q' is pressed
    if key_input == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()